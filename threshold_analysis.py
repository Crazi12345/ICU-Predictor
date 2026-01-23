import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    f1_score
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# --- Models Definitions ---

class LGSTMModel(nn.Module):
    def __init__(self, hp, in_dim):
        super().__init__()
        self.lstm1 = nn.LSTM(in_dim, int(hp['u1']), batch_first=True)
        self.drop1 = nn.Dropout(hp['dropout'])
        self.lstm2 = nn.LSTM(int(hp['u1']), int(hp['u2']), batch_first=True)
        self.drop2 = nn.Dropout(hp['dropout'])
        self.fc = nn.Linear(int(hp['u2']), 1)
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
        return out

# --- Helper Functions ---

def get_loss_fn(loss_name):
    if loss_name == 'binary_crossentropy':
        return nn.BCELoss()
    elif loss_name == 'mse':
        return nn.MSELoss()
    return nn.BCELoss()

def map_output_to_prob(preds, final_act):
    if final_act == 'tanh':
        return (preds + 1.0) / 2.0
    if final_act == 'relu':
        return np.clip(preds, 0.0, 1.0)
    return preds

def predict_in_batches(model, X, batch_size=32):
    model.eval()
    preds_list = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            out = model(batch)
            preds_list.append(out.cpu())
    return torch.cat(preds_list)

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

        inactive_utility += 0.0

    if best_utility == inactive_utility:
        normalized_utility = 0.0
    else:
        normalized_utility = (observed_utility - inactive_utility) / (best_utility - inactive_utility)

    return normalized_utility

def load_data(method):
    filename = f'data/preprocessed_{method}.pkl'
    if not os.path.exists(filename):
        logging.warning(f"{filename} not found. Trying preprocessed_data.pkl")
        filename = 'data/preprocessed_data.pkl'

    logging.info(f"Loading data from {filename}...")
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

# --- Main Training Logic ---

def train_and_analyze_thresholds():
    models_config = [
        {
            'name': 'Default_LGSTM_BestUtil_199',
            'dataset_method': 'neg1',
            'batch_size': 8, 'dropout': 0.4, 'final_activation': 'elu', 'loss': 'mse',
            'lr': 0.001, 'optimizer': 'adamw', 'u1': 128.0, 'u2': 64.0
        },
        {
            'name': 'Default_LGSTM_BestAUC_134',
            'dataset_method': 'mean',
            'batch_size': 8, 'dropout': 0.4, 'final_activation': 'sigmoid', 'loss': 'binary_crossentropy',
            'lr': 0.001, 'optimizer': 'adamw', 'u1': 128.0, 'u2': 32.0
        },
        {
            'name': 'Default_LGSTM_BestAUC_14',
            'dataset_method': 'mean',
            'batch_size': 8, 'dropout': 0.2, 'final_activation': 'sigmoid', 'loss': 'binary_crossentropy',
            'lr': 0.0005, 'optimizer': 'adamw', 'u1': 128.0, 'u2': 32.0
        }
    ]

    results_history = []

    # Thresholds to test
    thresholds_to_test = [0.001,0.005,0.01,0.1,0.25,0.33]

    # Epoch checkpoints
    checkpoints = [75]

    # Store the best model info for final Confusion Matrix
    best_overall_util = -float('inf')
    best_model_name = ""
    best_model_details = ""
    best_model_data = None # (y_true, y_pred_bin)

    for config in models_config:
        logging.info(f"\n{'='*20}\nStarting: {config['name']} (Method: {config['dataset_method']})\n{'='*20}")

        # 1. Load Data
        data = load_data(config['dataset_method'])
        X_train = data['X_train_seq'].to(device)
        y_train = data['y_train_seq'].to(device)
        X_test = data['X_test_seq'].to(device)
        y_test = data['y_test_seq']

        has_orig_lists = 'y_seq_list' in data and 'idx' in data and 'n_train' in data and 'n_val' in data

        # 2. Setup Model
        in_dim = X_train.shape[2]
        model = LGSTMModel(config, in_dim).to(device)

        if config['optimizer'] == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
        else:
            optimizer = optim.Adam(model.parameters(), lr=config['lr'])

        criterion = get_loss_fn(config['loss'])

        # 3. Training Loop
        ds_train = TensorDataset(X_train, y_train)
        dl_train = DataLoader(ds_train, batch_size=config['batch_size'], shuffle=True)

        current_epoch = 0

        for target_epoch in checkpoints:
            epochs_to_train = target_epoch - current_epoch
            if epochs_to_train <= 0:
                continue

            # Train
            model.train()
            for _ in range(epochs_to_train):
                for xb, yb in dl_train:
                    optimizer.zero_grad()
                    out = model(xb)
                    loss = criterion(out, yb.unsqueeze(-1))
                    loss.backward()
                    optimizer.step()

            current_epoch = target_epoch

            # Evaluate at this checkpoint
            model.eval()
            with torch.no_grad():
                preds_tensor = predict_in_batches(model, X_test, config['batch_size'])
                preds_np = preds_tensor.numpy().flatten()

            probs = map_output_to_prob(preds_np, config['final_activation'])
            y_test_flat = y_test.numpy().flatten()

            # AUC (Threshold independent)
            try:
                auc = roc_auc_score(y_test_flat, probs)
            except:
                auc = 0.5

            # Compute for each threshold
            for th in thresholds_to_test:
                preds_bin = (probs > th).astype(int)
                f1 = f1_score(y_test_flat, preds_bin)

                # Utility
                if has_orig_lists:
                    probs_reshaped = probs.reshape(X_test.shape[0], X_test.shape[1])
                    preds_reshaped = (map_output_to_prob(predict_in_batches(model, X_test, config['batch_size']).numpy(), config['final_activation']) > th).astype(int).reshape(X_test.shape[0], X_test.shape[1])

                    idx = data['idx']
                    n_train = data['n_train']
                    n_val = data['n_val']
                    y_seq_list = data['y_seq_list']
                    test_indices = idx[n_train+n_val:]

                    final_labels_list = []
                    final_preds_list = []

                    for k, original_idx in enumerate(test_indices):
                        if k >= len(preds_reshaped): break
                        true_seq = y_seq_list[original_idx]
                        pat_len = len(true_seq)
                        max_len = preds_reshaped.shape[1]
                        eff_len = min(pat_len, max_len)

                        pred_seq = preds_reshaped[k, :eff_len]
                        final_labels_list.append(true_seq[:eff_len])
                        final_preds_list.append(pred_seq)

                    util = compute_prediction_utility(final_labels_list, final_preds_list)
                else:
                    util = 0.0

                logging.info(f"Epoch {current_epoch} | Thresh {th}: AUC={auc:.4f}, F1={f1:.4f}, Util={util:.4f}")

                results_history.append({
                    'model': config['name'],
                    'epoch': current_epoch,
                    'threshold': th,
                    'auc': auc,
                    'f1': f1,
                    'utility': util
                })

                if util > best_overall_util:
                    best_overall_util = util
                    best_model_name = config['name']
                    best_model_details = f"{config['name']} (Epoch {current_epoch}, Thresh {th})"
                    best_model_data = (y_test_flat, preds_bin)

    # --- Plotting ---
    df_res = pd.DataFrame(results_history)
    df_res.to_csv('threshold_analysis_results.csv', index=False)

    # Plot Utility vs Threshold
    plt.figure(figsize=(12, 8))

    # Create a unique identifier for each Model+Epoch combination
    models = df_res['model'].unique()
    epochs = df_res['epoch'].unique()

    # We want to plot Utility (Y) vs Threshold (X)
    # Different lines for each Model-Epoch pair?
    # Or maybe just group by Model and show different Epochs with different line styles?

    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    model_color_map = dict(zip(models, colors))

    # Line styles for epochs
    linestyles = ['-', '--', ':']
    # Ensure we have enough styles if epochs > 3, though here we have 3 (50, 75, 100)
    epoch_style_map = {e: linestyles[i % len(linestyles)] for i, e in enumerate(sorted(epochs))}

    for model_name in models:
        for epoch_val in epochs:
            sub = df_res[(df_res['model'] == model_name) & (df_res['epoch'] == epoch_val)]
            if sub.empty:
                continue

            # Sort by threshold to ensure correct plotting
            sub = sub.sort_values('threshold')

            label = f"{model_name} (E{epoch_val})"
            plt.plot(sub['threshold'], sub['utility'],
                     color=model_color_map[model_name],
                     linestyle=epoch_style_map[epoch_val],
                     marker='o',
                     label=label)

    plt.title('Utility vs Threshold (at Epochs 50, 75, 100)')
    plt.xlabel('Threshold')
    plt.ylabel('Normalized Utility')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('threshold_utility_analysis.png')
    plt.close()

    # Confusion Matrix for Best Model (Best util across all epochs/thresholds)
    if best_model_data:
        y_true, y_pred = best_model_data
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'CM: {best_model_details}\n(Best Util: {best_overall_util:.4f})')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Neg', 'Pos'])
        plt.yticks(tick_marks, ['Neg', 'Pos'])

        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig('threshold_best_model_confusion_matrix.png')
        plt.close()

    logging.info("Done. Results saved to 'threshold_analysis_results.csv' and plots generated.")

if __name__ == "__main__":
    train_and_analyze_thresholds()
