import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Configuration
PREPROCESSED_FILE = 'data/preprocessed_mean.pkl'
PLOT_DIR = f"single_pass_plots_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
os.makedirs(PLOT_DIR, exist_ok=True)

# Best Hyperparameters (LGSTM, neg1) from final_model_comparison_all.csv
HP = {
    'batch_size': 4,
    'dropout': 0.4,
    'final_activation': 'elu',
    'loss': 'mse',
    'lr': 0.0001,
    'optimizer': 'adam',
    'u1': 128,
    'u2': 16
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# --- Data Loading ---
def get_data(path):
    if not os.path.exists(path):
        logging.error(f"File {path} not found!")
        sys.exit(1)

    logging.info(f"Loading data from {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    logging.info("Data loaded.")
    return data

# --- Model Definition ---
class LGSTMModel(nn.Module):
    def __init__(self, hp, in_dim):
        super().__init__()
        self.lstm1 = nn.LSTM(in_dim, hp['u1'], batch_first=True)
        self.drop1 = nn.Dropout(hp['dropout'])
        self.lstm2 = nn.LSTM(hp['u1'], hp['u2'], batch_first=True)
        self.drop2 = nn.Dropout(hp['dropout'])
        self.fc = nn.Linear(hp['u2'], 1)
        self.final_act = hp.get('final_activation', 'sigmoid')

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.drop1(x)
        x, _ = self.lstm2(x)
        x = self.drop2(x)
        out = self.fc(x)

        if self.final_act == 'sigmoid':
            return torch.sigmoid(out)
        elif self.final_act == 'tanh':
            return torch.tanh(out)
        elif self.final_act == 'relu':
            return torch.relu(out)
        elif self.final_act == 'leakyrelu':
            return torch.nn.functional.leaky_relu(out)
        elif self.final_act == 'elu':
            return torch.nn.functional.elu(out)
        return out

# --- Helpers ---
def get_loss_fn(loss_name):
    if loss_name == 'binary_crossentropy':
        return nn.BCELoss()
    elif loss_name == 'mse':
        return nn.MSELoss()
    elif loss_name == 'hinge':
        return lambda y_pred, y_true: torch.mean(torch.clamp(1 - y_true * y_pred, min=0))
    return nn.BCELoss()

def map_output_to_prob(preds, final_act):
    if final_act == 'tanh':
        return (preds + 1.0) / 2.0
    if final_act in ['relu', 'leakyrelu', 'elu']:
        return np.clip(preds, 0.0, 1.0)
    return preds

def predict_batched(model, X, batch_size=32):
    model.eval()
    preds_list = []
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for xb, in dataloader:
            xb = xb.to(device)
            out = model(xb)
            preds_list.append(out.cpu())

    return torch.cat(preds_list, dim=0)

# --- Utility Score Function ---
def compute_prediction_utility(labels, predictions, dt_early=-12, dt_optimal=-6, dt_late=3.0,
                             max_u_tp=1, min_u_fn=-2, u_fp=-0.05, u_tn=0):
    n_patients = len(labels)
    observed_utility = 0.0
    best_utility = 0.0
    inactive_utility = 0.0

    for i in range(n_patients):
        label = labels[i]
        pred = predictions[i]
        n = len(label)

        if np.any(label):
            t_sepsis = np.argmax(label == 1)
            is_septic = True
        else:
            t_sepsis = float('inf')
            is_septic = False

        u_vector = np.zeros(n)
        if is_septic:
            t_diff = np.arange(n) - t_sepsis
            mask_slope = (t_diff >= dt_early) & (t_diff <= dt_optimal)
            u_vector[mask_slope] = max_u_tp * (t_diff[mask_slope] - dt_early) / (dt_optimal - dt_early)
            mask_plateau = (t_diff > dt_optimal) & (t_diff <= dt_late)
            u_vector[mask_plateau] = max_u_tp

        term1 = pred * u_vector
        term2 = (pred == 1) & (u_vector == 0)
        p_utility = np.sum(term1) + np.sum(term2.astype(float) * u_fp)
        observed_utility += p_utility

        best_pred = (u_vector > 0).astype(int)
        term1_best = best_pred * u_vector
        term2_best = (best_pred == 1) & (u_vector == 0)
        b_utility = np.sum(term1_best) + np.sum(term2_best.astype(float) * u_fp)
        best_utility += b_utility

    if best_utility == inactive_utility:
        normalized_utility = 0.0
    else:
        normalized_utility = (observed_utility - inactive_utility) / (best_utility - inactive_utility)

    return normalized_utility

def evaluate_metrics(model, X_seq, indices, y_seq_list, max_seq_len, final_act):
    model.eval()

    # Prediction
    preds_tensor = predict_batched(model, X_seq, batch_size=32)
    preds_flat = preds_tensor.numpy().flatten()
    probs_mapped = map_output_to_prob(preds_flat, final_act)

    # Re-shape for utility calculation
    probs_shaped = probs_mapped.reshape(preds_tensor.shape[:2])
    preds_padded = (probs_shaped > 0.5).astype(int)

    # Utility Calculation
    labels_list_util = []
    preds_list_util = []

    # AUC Calculation
    y_true_flat = []

    for k, idx_val in enumerate(indices):
        true_seq = y_seq_list[idx_val]
        pat_len = min(len(true_seq), max_seq_len)

        p_seq = preds_padded[k, :pat_len]
        labels_list_util.append(true_seq[:pat_len])
        preds_list_util.append(p_seq)

        y_true_flat.extend(true_seq[:pat_len])

    # Utility
    util_score = compute_prediction_utility(labels_list_util, preds_list_util)

    # AUC
    # We need flattened probs and labels for AUC, but we need to mask out padding
    # Ideally, we should reconstruct the flat arrays based on actual lengths
    # However, for simplicity and speed (and consistency with previous script if it did this):
    # The previous script did:
    #   auc_score = roc_auc_score(y_val.numpy().flatten(), val_preds_prob)
    # This included padding. We will stick to the previous method for AUC to be consistent,
    # OR we can be more precise. The user said "see training-Official.py".
    # training-Official.py used: roc_auc_score(y_val.numpy().flatten(), val_preds_prob)
    # We will do the same for AUC.

    # But wait, y_seq_list and indices are for the unpadded/variable length Utility calculation.
    # For AUC, we have the tensor X_seq. We need the corresponding y_seq tensor.
    # We will pass y_seq tensor to this function or handle it outside.

    return util_score

# --- Main Execution ---
def main():
    # 1. Load Data
    data = get_data(PREPROCESSED_FILE)
    X_train = data['X_train_seq']
    y_train = data['y_train_seq']
    X_test = data['X_test_seq']
    y_test = data['y_test_seq']
    y_seq_list = data['y_seq_list']
    idx = data['idx']
    n_train = data['n_train']
    n_val = data['n_val']

    # Test indices are those after train and val
    test_indices = idx[n_train+n_val:]

    in_dim = X_train.shape[2]
    max_seq_len = X_train.shape[1]

    logging.info(f"Training Data Shape: {X_train.shape}")
    logging.info(f"Test Data Shape: {X_test.shape}")

    # 2. Initialize Model
    model = LGSTMModel(HP, in_dim).to(device)

    # 3. Setup Training
    lr = HP['lr']
    if HP['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = get_loss_fn(HP['loss'])

    ds_train = TensorDataset(X_train, y_train)
    dl_train = DataLoader(ds_train, batch_size=HP['batch_size'], shuffle=True)

    # 4. Training Loop (25 to 250, step 25)
    epochs_log = []
    auc_log = []
    util_log = []

    current_epoch = 0
    target_epochs = list(range(25, 251, 25))

    logging.info("Starting Single Pass Training (LGSTM - neg1)...")

    for target in target_epochs:
        epochs_to_run = target - current_epoch
        logging.info(f"Training for {epochs_to_run} epochs (Target: {target})...")

        model.train()
        for _ in range(epochs_to_run):
            for xb, yb in dl_train:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb.unsqueeze(-1))
                loss.backward()
                optimizer.step()

        current_epoch = target

        # Evaluation
        logging.info(f"Evaluating at epoch {current_epoch}...")

        # Utility
        util = evaluate_metrics(model, X_test, test_indices, y_seq_list, max_seq_len, HP['final_activation'])

        # AUC (Standard calculation as per training-Official.py)
        model.eval()
        preds_tensor = predict_batched(model, X_test, batch_size=32)
        preds_prob = map_output_to_prob(preds_tensor.numpy().flatten(), HP['final_activation'])
        y_test_flat = y_test.numpy().flatten()

        try:
            auc = roc_auc_score(y_test_flat, preds_prob)
        except Exception as e:
            logging.warning(f"AUC calculation failed: {e}")
            auc = 0.5

        logging.info(f"Epoch {current_epoch}: AUC = {auc:.4f}, Utility = {util:.4f}")

        epochs_log.append(current_epoch)
        auc_log.append(auc)
        util_log.append(util)

    # 5. Confusion Matrix
    logging.info("Generating Confusion Matrix...")
    # Re-calculate predictions for the final model (unpadded)
    model.eval()
    preds_tensor = predict_batched(model, X_test, batch_size=32)
    preds_flat_tensor = preds_tensor.numpy().flatten()
    probs_mapped = map_output_to_prob(preds_flat_tensor, HP['final_activation'])
    probs_shaped = probs_mapped.reshape(preds_tensor.shape[:2])
    preds_bin = (probs_shaped > 0.5).astype(int)

    all_true = []
    all_pred = []

    for k, idx_val in enumerate(test_indices):
        true_seq = y_seq_list[idx_val]
        pat_len = min(len(true_seq), max_seq_len)

        # Extract valid part
        p_seq = preds_bin[k, :pat_len]
        t_seq = true_seq[:pat_len]

        all_true.extend(t_seq)
        all_pred.extend(p_seq)

    cm = confusion_matrix(all_true, all_pred)

    plt.figure(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix (Unpadded Test Data)")
    plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix.png"))
    plt.close()
    logging.info(f"Confusion Matrix saved to {os.path.join(PLOT_DIR, 'confusion_matrix.png')}")

    # 6. Plotting
    logging.info("Plotting results...")

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_log, util_log, 'o-', label='Utility Score', color='green')
    plt.plot(epochs_log, auc_log, 's-', label='AUC', color='blue', alpha=0.5)
    plt.title(f"LGSTM (neg1) Single Pass Training\nBatch={HP['batch_size']}, LR={HP['lr']}, Loss={HP['loss']}")
    plt.xlabel("Epochs")
    plt.ylabel("Utility Score")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(PLOT_DIR, "LGSTM_neg1_SinglePass_Metrics.png")
    plt.savefig(plot_path)
    logging.info(f"Plot saved to {plot_path}")

    # Save raw metrics to CSV
    df_results = pd.DataFrame({
        'epoch': epochs_log,
        'auc': auc_log,
        'utility': util_log
    })
    csv_path = os.path.join(PLOT_DIR, "single_pass_metrics.csv")
    df_results.to_csv(csv_path, index=False)
    logging.info(f"Metrics saved to {csv_path}")

if __name__ == "__main__":
    main()
