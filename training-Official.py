import os
import sys
import logging
import argparse
import pickle
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate timestamp for plot directory
run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
plot_dir = f"plotsOff_{run_timestamp}"
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_log_official.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# Argument Parsing
parser = argparse.ArgumentParser(description='Train ICU Prediction Model (Official Score)')
parser.add_argument('--use-cache', action='store_true', help='Use cached preprocessed data if available')
parser.add_argument('--use-last-final', action='store_true', help='Use the last known best configurations instead of grid search')
parser.add_argument('--use-all-pp', action='store_true', help='Run training on all preprocessing methods (neg1, mean, median, linear)')
args, unknown = parser.parse_known_args()

def get_last_final_configs():
    """Returns the best configurations from the last run (JSON), or falls back to hardcoded."""
    json_path = "best_model_configs.json"
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                configs = json.load(f)
            
            # Infer model_type if missing
            valid_configs = []
            for c in configs:
                if 'model_type' not in c:
                    if 'RNN' in c['name']:
                        c['model_type'] = 'RNN'
                    elif 'CNN' in c['name']:
                        c['model_type'] = 'CNN'
                    elif 'LGSTM' in c['name']:
                        c['model_type'] = 'LGSTM'
                    else:
                        logging.warning(f"Could not infer model_type for {c['name']}, skipping.")
                        continue
                valid_configs.append(c)

            logging.info(f"Loaded {len(valid_configs)} configurations from {json_path}")
            return valid_configs
        except Exception as e:
            logging.warning(f"Failed to load {json_path}: {e}")

    logging.info("Using HARDCODED fallback configurations.")
    return [
        {
            'name': 'RNN_BestUtil_82',
            'model_type': 'RNN',
            'hp': {'batch_size': 32, 'dropout': 0.4, 'final_activation': 'leakyrelu', 'loss': 'mse', 'lr': 0.001, 'optimizer': 'adamw', 'units': 64},
            'score_type': 'utility',
            'val_score': 0.0980
        },
        {
            'name': 'RNN_BestUtil_144',
            'model_type': 'RNN',
            'hp': {'batch_size': 64, 'dropout': 0.2, 'final_activation': 'leakyrelu', 'loss': 'mse', 'lr': 0.001, 'optimizer': 'adam', 'units': 64},
            'score_type': 'utility',
            'val_score': 0.0792
        },
        {
            'name': 'RNN_BestAUC_65',
            'model_type': 'RNN',
            'hp': {'batch_size': 32, 'dropout': 0.4, 'final_activation': 'sigmoid', 'loss': 'binary_crossentropy', 'lr': 0.001, 'optimizer': 'adam', 'units': 128},
            'score_type': 'auc',
            'val_score': 0.9895
        },
        {
            'name': 'RNN_BestAUC_3',
            'model_type': 'RNN',
            'hp': {'batch_size': 32, 'dropout': 0.2, 'final_activation': 'sigmoid', 'loss': 'binary_crossentropy', 'lr': 0.001, 'optimizer': 'adamw', 'units': 128},
            'score_type': 'auc',
            'val_score': 0.9888
        },
        {
            'name': 'CNN_BestUtil_1462',
            'model_type': 'CNN',
            'hp': {'batch_size': 64, 'dropout': 0.2, 'f1': 64, 'f2': 128, 'final_activation': 'leakyrelu', 'kernel_size': 5, 'loss': 'mse', 'lr': 0.0005, 'optimizer': 'adamw', 'stride': 1},
            'score_type': 'utility',
            'val_score': 0.0794
        },
        {
            'name': 'CNN_BestUtil_386',
            'model_type': 'CNN',
            'hp': {'batch_size': 32, 'dropout': 0.2, 'f1': 64, 'f2': 128, 'final_activation': 'sigmoid', 'kernel_size': 3, 'loss': 'binary_crossentropy', 'lr': 0.001, 'optimizer': 'adamw', 'stride': 1},
            'score_type': 'utility',
            'val_score': 0.0744
        },
        {
            'name': 'CNN_BestAUC_658',
            'model_type': 'CNN',
            'hp': {'batch_size': 32, 'dropout': 0.4, 'f1': 32, 'f2': 128, 'final_activation': 'sigmoid', 'kernel_size': 5, 'loss': 'binary_crossentropy', 'lr': 0.001, 'optimizer': 'adamw', 'stride': 1},
            'score_type': 'auc',
            'val_score': 0.9919
        },
        {
            'name': 'CNN_BestAUC_1138',
            'model_type': 'CNN',
            'hp': {'batch_size': 64, 'dropout': 0.2, 'f1': 32, 'f2': 64, 'final_activation': 'tanh', 'kernel_size': 5, 'loss': 'mse', 'lr': 0.001, 'optimizer': 'adamw', 'stride': 1},
            'score_type': 'auc',
            'val_score': 0.9915
        },
        {
            'name': 'LGSTM_BestUtil_0',
            'model_type': 'LGSTM',
            'hp': {'batch_size': 8, 'dropout': 0.2, 'final_activation': 'sigmoid', 'loss': 'binary_crossentropy', 'lr': 0.001, 'optimizer': 'adam', 'u1': 64, 'u2': 32},
            'score_type': 'utility',
            'val_score': 0.1044
        },
        {
            'name': 'LGSTM_BestUtil_1',
            'model_type': 'LGSTM',
            'hp': {'batch_size': 8, 'dropout': 0.2, 'final_activation': 'sigmoid', 'loss': 'binary_crossentropy', 'lr': 0.001, 'optimizer': 'adam', 'u1': 64, 'u2': 64},
            'score_type': 'utility',
            'val_score': -999.0000
        }
    ]

def get_data(use_cache=False, cache_path='preprocessed_data.pkl'):
    if use_cache and os.path.exists(cache_path):
        logging.info(f"Loading preprocessed data from {cache_path}...")
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            logging.info("✓ Data loaded from cache.")
            return data
        except Exception as e:
            logging.info(f"⚠ Failed to load cache ({e}). Re-running preprocessing...")

    # If we are here, and a specific cache_path was requested (e.g., preprocessed_mean.pkl),
    # but it doesn't exist, we can't just fall back to the default pipeline easily because
    # the default pipeline in this script does the "neg1" strategy (the one we hardcoded earlier).
    # Ideally, preprocessing.py should be used to generate these files.

    if cache_path != 'preprocessed_data.pkl':
         logging.error(f"Cache file {cache_path} not found! Please run preprocessing.py first.")
         sys.exit(1)

    # --- Default Data Loading (Fallback) ---
    try:
        import kagglehub
        path = kagglehub.dataset_download("salikhussaini49/prediction-of-sepsis")
        logging.info(f"Dataset downloaded to: {path}")
        dataset_path = path
    except Exception as e:
        logging.info(f"Download failed ({e}). Using local path or expecting Dataset.csv in working dir.")
        dataset_path = os.getcwd()

    possible_paths = [
        os.path.join(dataset_path, "Dataset.csv"),
        os.path.join(dataset_path, "dataset.csv"),
        os.path.join(os.path.expanduser("~"), "Downloads", "Dataset.csv"),
        os.path.join(os.path.expanduser("~"), "Downloads", "dataset.csv"),
        "Dataset.csv",
        "dataset.csv",
    ]

    csv_file = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_file = path
            logging.info(f"✓ Found dataset at: {csv_file}")
            break

    df_full = None
    if csv_file:
        df_full = pd.read_csv(csv_file)
        logging.info(f"✓ Loaded Dataset: {df_full.shape[0]} rows, {df_full.shape[1]} columns")
    else:
        # Synthetic fallback
        logging.info("\n⚠ Creating synthetic dataset...")
        n_patients = 100
        max_time_steps = 200
        n_features = 15
        data_list = []
        for p_id in range(n_patients):
            n_steps = np.random.randint(50, max_time_steps)
            for t in range(n_steps):
                features = np.random.randn(n_features) * 0.5
                has_sepsis = np.random.rand() > 0.7
                if has_sepsis:
                    features[:3] += t / max_time_steps * 2
                sepsis_label = 1 if (has_sepsis and t > n_steps * 0.6) else 0
                data_list.append({'patient_id': p_id, **{f'feature_{i}': features[i] for i in range(n_features)}, 'SepsisLabel': sepsis_label})
        df_full = pd.DataFrame(data_list)

    # --- Preprocessing ---
    if df_full is not None:
        if 'Unnamed: 0' in df_full.columns:
            df_full = df_full.drop(columns=['Unnamed: 0'])
        if 'Patient_ID' in df_full.columns:
            df_full.rename(columns={'Patient_ID': 'patient_id'}, inplace=True)

        logging.info(f"Shape: {df_full.shape}")

        df_imputed = df_full.copy()
        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns.tolist()
        if 'patient_id' in numeric_cols: numeric_cols.remove('patient_id')
        if 'SepsisLabel' in numeric_cols: numeric_cols.remove('SepsisLabel')

        # Per-patient imputation and scaling
        logging.info("⏳ Imputing (ffill/bfill) and Scaling per-patient (ignoring NaNs)...")
        df_scaled = df_imputed.copy()

        # 1. Temporal Imputation
        for col in numeric_cols:
            df_scaled[col] = df_scaled.groupby('patient_id')[col].transform(lambda x: x.ffill().bfill())

        # 2. Scaling ignoring NaNs
        def robust_scale_ignore_nan(x):
            if x.isna().all():
                return x
            mean = np.nanmean(x)
            std = np.nanstd(x)
            if np.isnan(std) or std == 0:
                return x - mean if not np.isnan(mean) else x
            return (x - mean) / std

        for col in numeric_cols:
             df_scaled[col] = df_scaled.groupby('patient_id')[col].transform(robust_scale_ignore_nan)

        # 3. Fill remaining NaNs with -1 (as requested to indicate missingness)
        missing_count = df_scaled[numeric_cols].isna().sum().sum()
        logging.info(f"  Filling {missing_count} remaining NaNs with -1.0")
        df_scaled[numeric_cols] = df_scaled[numeric_cols].fillna(-1.0)

        logging.info("✓ Scaling and Imputation complete.")

    # --- Sequence Creation ---
    max_seq_len = 256
    X_seq_list = []
    y_seq_list = []
    patient_ids = []

    all_patient_ids = sorted(df_scaled['patient_id'].unique())
    half_n = max(1, len(all_patient_ids) // 1)
    patient_ids_to_use = all_patient_ids[:half_n]

    for patient_id in patient_ids_to_use:
        mask = df_scaled['patient_id'] == patient_id
        X_pat = df_scaled.loc[mask, numeric_cols].values
        y_pat = df_scaled.loc[mask, 'SepsisLabel'].values if 'SepsisLabel' in df_scaled.columns else np.ones(mask.sum())

        if len(X_pat) > max_seq_len:
            X_pat = X_pat[-max_seq_len:, :]
            y_pat = y_pat[-max_seq_len:]
        else:
            pad_len = max_seq_len - len(X_pat)
            X_pat = np.vstack([X_pat, np.zeros((pad_len, X_pat.shape[1]))])
            y_pat = np.concatenate([y_pat, np.zeros(pad_len)])

        X_seq_list.append(X_pat)
        y_seq_list.append(y_pat)
        patient_ids.append(patient_id)

    X_seq = np.array(X_seq_list)
    y_seq = np.array(y_seq_list)

    n = len(X_seq)
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    X_train_seq = torch.tensor(X_seq[idx[:n_train]], dtype=torch.float32)
    y_train_seq = torch.tensor(y_seq[idx[:n_train]], dtype=torch.float32)
    X_val_seq = torch.tensor(X_seq[idx[n_train:n_train+n_val]], dtype=torch.float32)
    y_val_seq = torch.tensor(y_seq[idx[n_train:n_train+n_val]], dtype=torch.float32)
    X_test_seq = torch.tensor(X_seq[idx[n_train+n_val:]], dtype=torch.float32)
    y_test_seq = torch.tensor(y_seq[idx[n_train+n_val:]], dtype=torch.float32)

    data_bundle = {
        'X_train_seq': X_train_seq, 'y_train_seq': y_train_seq,
        'X_val_seq': X_val_seq, 'y_val_seq': y_val_seq,
        'X_test_seq': X_test_seq, 'y_test_seq': y_test_seq,
        'y_seq_list': y_seq_list, 'idx': idx,
        'n_train': n_train, 'n_val': n_val
    }

    if use_cache:
        logging.info(f"Saving preprocessed data to {cache_path}...")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data_bundle, f)
            logging.info("✓ Data saved to cache.")
        except Exception as e:
            logging.info(f"⚠ Failed to save cache: {e}")

    return data_bundle

# --- PyTorch Models ---
class RNNModel(nn.Module):
    def __init__(self, hp, in_dim):
        super().__init__()
        self.rnn = nn.RNN(in_dim, hp['units'], batch_first=True)
        self.drop = nn.Dropout(hp['dropout'])
        self.fc = nn.Linear(hp['units'], 1)
        self.final_act = hp.get('final_activation', 'sigmoid')

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.drop(x)
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

class CNNModel(nn.Module):
    def __init__(self, hp, in_dim):
        super().__init__()
        ks = hp.get('kernel_size', 3)
        st = hp.get('stride', 1)
        self.conv1 = nn.Conv1d(in_dim, hp['f1'], ks, stride=st, padding=ks//2)
        self.conv2 = nn.Conv1d(hp['f1'], hp['f2'], ks, stride=st, padding=ks//2)
        self.drop = nn.Dropout(hp['dropout'])
        self.fc = nn.Linear(hp['f2'], 1)
        self.final_act = hp.get('final_activation', 'sigmoid')

    def forward(self, x):
        # x: (N, L, C) -> (N, C, L)
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = self.drop(x)
        x = torch.relu(self.conv2(x))
        x = self.drop(x)
        # (N, C, L) -> (N, L, C)
        x = x.transpose(1, 2)
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

# --- Training & Evaluation Functions ---

def get_loss_fn(loss_name):
    if loss_name == 'binary_crossentropy':
        return nn.BCELoss()
    elif loss_name == 'mse':
        return nn.MSELoss()
    elif loss_name == 'hinge':
        return lambda y_pred, y_true: torch.mean(torch.clamp(1 - y_true * y_pred, min=0))
    return nn.BCELoss()

def predict_batched(model, X, batch_size=32):
    """
    Perform inference in batches to avoid OOM.
    """
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

def map_output_to_prob(preds, final_act):
    if final_act == 'tanh':
        return (preds + 1.0) / 2.0
    if final_act == 'relu' or final_act == 'leakyrelu' or final_act == 'elu':
        return np.clip(preds, 0.0, 1.0)
    return preds

# Score Utility Function
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

    return normalized_utility, observed_utility

def evaluate_utility_val(model, X_val, val_indices, y_seq_list, max_seq_len, final_act='sigmoid'):
    model.eval()
    probs_tensor = predict_batched(model, X_val, batch_size=32)
    probs_flat = probs_tensor.numpy().flatten()
    # Map to prob
    probs_mapped = map_output_to_prob(probs_flat, final_act).reshape(probs_tensor.shape[:2])

    # Threshold 0.5
    preds_padded = (probs_mapped > 0.5).astype(int)

    labels_list = []
    preds_list = []

    for k, idx_val in enumerate(val_indices):
        true_seq = y_seq_list[idx_val]
        pat_len = min(len(true_seq), max_seq_len)

        p_seq = preds_padded[k, :pat_len]
        labels_list.append(true_seq[:pat_len])
        preds_list.append(p_seq)

    norm, raw = compute_prediction_utility(labels_list, preds_list)
    return norm

def train_model_pyt(model_class, hp, in_dim, X_train, y_train, X_val, y_val, epochs):
    model = model_class(hp, in_dim).to(device)
    lr = hp.get('lr', 1e-3)
    opt_name = hp.get('optimizer', 'adam').lower()

    if opt_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif opt_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    elif opt_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif opt_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_name = hp.get('loss', 'binary_crossentropy')
    criterion = get_loss_fn(loss_name)

    if loss_name == 'hinge':
        y_train_target = (y_train.clone() * 2.0) - 1.0
    else:
        y_train_target = y_train

    num_workers = 0
    use_pin_memory = (device.type == 'cuda')

    ds_train = TensorDataset(X_train, y_train_target)
    dl_train = DataLoader(ds_train, batch_size=hp['batch_size'], shuffle=True,
                          num_workers=num_workers, pin_memory=use_pin_memory)

    history = {'loss': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb.unsqueeze(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        history['loss'].append(train_loss / len(dl_train))

    return model, history

def grid_search_pyt(model_class, grid, name, in_dim, X_train, y_train, X_val, y_val,
                   val_indices, y_seq_list, max_seq_len, epochs=10):
    results = []

    os.makedirs('/tmp/icu_tune', exist_ok=True)

    for hp in ParameterGrid(grid):
        if hp['loss'] == 'binary_crossentropy' and hp['final_activation'] != 'sigmoid': continue
        if hp['loss'] == 'hinge' and hp['final_activation'] == 'sigmoid': continue

        logging.info(f'[{name}] trying {hp}')
        model = None
        try:
            model, hist = train_model_pyt(model_class, hp, in_dim, X_train, y_train, X_val, y_val, epochs)
            # Evaluate Utility
            util_score = evaluate_utility_val(model, X_val, val_indices, y_seq_list, max_seq_len, hp.get('final_activation', 'sigmoid'))

            # Evaluate AUC on Val
            model.eval()
            val_out = predict_batched(model, X_val, batch_size=32)
            val_preds_np = val_out.numpy().flatten()
            val_preds_prob = map_output_to_prob(val_preds_np, hp.get('final_activation', 'sigmoid'))
            try:
                auc_score = roc_auc_score(y_val.numpy().flatten(), val_preds_prob)
            except:
                    auc_score = 0.5

        except Exception as e:
            logging.info(f'  fit failed: {e}')
            util_score = -999.0
            auc_score = 0.5

        logging.info(f'  Utility={util_score:.4f}, AUC={auc_score:.4f}')
        results.append({**hp, 'utility': util_score, 'auc': auc_score})

    df = pd.DataFrame(results)
    return df

def evaluate_sequence_model(model, X_seq, y_seq, threshold=0.5, name='model', final_act='sigmoid'):
    model.eval()
    preds = predict_batched(model, X_seq, batch_size=32)

    preds_flat = preds.numpy().flatten()
    preds_prob = map_output_to_prob(preds_flat, final_act)
    y_flat = y_seq.numpy().flatten()

    auc = roc_auc_score(y_flat, preds_prob)
    avg_prec = average_precision_score(y_flat, preds_prob)
    y_pred_bin = (preds_prob >= threshold).astype(int)
    prec = precision_score(y_flat, y_pred_bin, zero_division=0)
    rec = recall_score(y_flat, y_pred_bin, zero_division=0)
    f1 = f1_score(y_flat, y_pred_bin, zero_division=0)
    cm = confusion_matrix(y_flat, y_pred_bin)

    return {
        'auc': auc, 'ap': avg_prec, 'precision': prec, 'recall': rec, 'f1': f1,
        'confusion_matrix': cm, 'y_true': y_flat, 'y_pred': preds_prob, 'name': name
    }

def plot_patient_prediction(model, X_seq, y_seq_list, original_indices, max_seq_len, final_act='sigmoid', prefix=''):
    """
    Plots the prediction score for each hour for a few sample patients.
    """
    os.makedirs(plot_dir, exist_ok=True)
    model.eval()
    preds = predict_batched(model, X_seq, batch_size=32)

    preds_np = preds.numpy()

    # Pick 2 random patients (one positive, one negative if possible)
    np.random.seed(99) # Fixed seed for reproducibility
    sample_indices = np.random.choice(len(original_indices), size=min(3, len(original_indices)), replace=False)

    for i, idx_in_batch in enumerate(sample_indices):
        true_pat_idx = original_indices[idx_in_batch]
        true_seq = y_seq_list[true_pat_idx]
        pat_len = min(len(true_seq), max_seq_len)

        pred_seq_raw = preds_np[idx_in_batch, :pat_len].flatten()
        pred_seq_prob = map_output_to_prob(pred_seq_raw, final_act)

        plt.figure(figsize=(10, 4))
        plt.plot(range(pat_len), true_seq[:pat_len], label='True Label', color='black', linestyle='--')
        plt.plot(range(pat_len), pred_seq_prob, label='Prediction Score', color='blue', alpha=0.7)
        plt.axhline(0.5, color='red', linestyle=':', label='Threshold')
        plt.ylim(-0.1, 1.1)
        plt.xlabel('Hours (Time Steps)')
        plt.ylabel('Sepsis Probability / Label')
        plt.title(f'Patient {true_pat_idx} - Hourly Predictions ({prefix})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/{prefix}_patient_{true_pat_idx}_hourly.png')
        plt.close()

def plot_all_results(eval_metrics_list, model_dfs, final_epoch_results, best_model_name):
    # Ensure plots directory exists
    os.makedirs(plot_dir, exist_ok=True)

    # 1. Detailed Hyperparam plots for ALL models
    for name, df in model_dfs.items():
        if df is not None and not df.empty:
            # Identify params (exclude metrics and identifiers)
            exclude = ['utility', 'auc', 'score']
            params = [c for c in df.columns if c not in exclude and df[c].nunique() > 1]

            if not params:
                continue

            # Layout: Grid of subplots
            ncols = 3
            nrows = (len(params) + ncols - 1) // ncols
            fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))

            # Ensure axs is iterable even if only 1 subplot
            if nrows * ncols == 1:
                axs = np.array([axs])
            axs = axs.ravel()

            for i, param in enumerate(params):
                # Get max score per param value
                best_scores = df.groupby(param)['utility'].max()

                # Plot
                x_vals = best_scores.index.astype(str)
                y_vals = best_scores.values

                axs[i].bar(x_vals, y_vals, color='skyblue', edgecolor='black')
                axs[i].set_title(f'Effect of {param}')
                axs[i].set_ylabel('Best Utility Score')
                axs[i].tick_params(axis='x', rotation=45)
                axs[i].grid(axis='y', linestyle='--', alpha=0.7)

            # Turn off unused subplots
            for j in range(i + 1, len(axs)):
                axs[j].axis('off')

            plt.suptitle(f'{name} Hyperparameter Impact (Utility)', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'{plot_dir}/{name}_hyperparams.png')
            plt.close()

    # 2. Final Epochs Performance Line Plot
    if final_epoch_results:
        epochs = [r['epoch'] for r in final_epoch_results]
        utils = [r['utility'] for r in final_epoch_results]
        aucs = [r['auc'] for r in final_epoch_results]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, utils, 'o-', label='Utility Score', color='green')
        plt.plot(epochs, aucs, 's-', label='AUC', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.title(f'{best_model_name} Scores vs Epochs')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/{best_model_name}_epochs_trend.png')
        plt.close()

def run_training_pipeline(data_bundle, result_prefix=""):
    """
    Executes the training pipeline (Grid Search -> Candidate Selection -> Final Training)
    for a given dataset bundle.
    """
    X_train_seq = data_bundle['X_train_seq']
    y_train_seq = data_bundle['y_train_seq']
    X_val_seq = data_bundle['X_val_seq']
    y_val_seq = data_bundle['y_val_seq']
    X_test_seq = data_bundle['X_test_seq']
    y_test_seq = data_bundle['y_test_seq']
    y_seq_list = data_bundle['y_seq_list']
    idx = data_bundle['idx']
    n_train = data_bundle['n_train']
    n_val = data_bundle['n_val']

    max_seq_len = X_train_seq.shape[1]
    in_dim = X_train_seq.shape[2]

    logging.info(f"[{result_prefix}] Train: {X_train_seq.shape}, Val: {X_val_seq.shape}, Test: {X_test_seq.shape}")

    candidate_configs = []

    if args.use_last_final:
        logging.info("=== USING LAST FINAL CONFIGURATIONS (SKIPPING GRID SEARCH) ===")
        configs = get_last_final_configs()
        for c in configs:
            if c['model_type'] == 'RNN':
                model_cls = RNNModel
            elif c['model_type'] == 'CNN':
                model_cls = CNNModel
            elif c['model_type'] == 'LGSTM':
                model_cls = LGSTMModel
            else:
                logging.warning(f"Unknown model type: {c['model_type']}")
                continue

            candidate_configs.append({
                'name': c['name'],
                'model_cls': model_cls,
                'hp': c['hp'],
                'score_type': c['score_type'],
                'val_score': c['val_score']
            })

    else:
        logging.info(f"=== [{result_prefix}] FULL GRID SEARCH ON ALL MODELS (PYTORCH) - OFFICIAL SCORE & AUC ===")

        big_rnn_grid = {
            'units': [64, 128],
            'dropout': [0.2, 0.4],
            'lr': [1e-3, 5e-4],
            'batch_size': [32, 64],
            #'batch_size': [32],
            'final_activation': ['sigmoid', 'leakyrelu', 'elu', 'tanh'],
            #'final_activation': ['sigmoid', 'elu' ],
            'loss': ['binary_crossentropy', 'mse', 'hinge'],
            #'loss': ['binary_crossentropy', 'mse'],
            'optimizer': ['adam', 'adamw']
            #'optimizer': ['adam']
        }

        big_cnn_grid = {
            'f1': [32, 64],
            #'f2': [64],
            'f2': [64, 128],
            'kernel_size': [3, 5],
            #'kernel_size': [3],
            'stride': [1, 2],
            #'stride': [1],
            'dropout': [0.2, 0.4],
            'lr': [1e-3, 5e-4],
            #'batch_size': [32],
            'batch_size': [32, 64],
            'final_activation': ['sigmoid', 'leakyrelu', 'elu', 'tanh'],
            #'final_activation': ['sigmoid','elu'],
            'loss': ['binary_crossentropy', 'mse', 'hinge'],
            #'loss': ['binary_crossentropy', 'mse'],
            'optimizer': ['adam', 'adamw']
            #'optimizer': ['adam']
        }

        big_lgstm_grid = {
                'u1': [64, 128],
            #'u1': [64],
            'u2': [32, 64],
            'dropout': [0.2, 0.4],
            'lr': [1e-3, 5e-4],
            'batch_size': [8, 16], # Updated as requested
            #'batch_size': [16], # Updated as requested
            'final_activation': ['sigmoid', 'leakyrelu', 'elu', 'tanh'],
            #'final_activation': ['sigmoid','elu'],
            'loss': ['binary_crossentropy', 'mse', 'hinge'],
            #'loss': ['binary_crossentropy', 'mse'],
            'optimizer': ['adam', 'adamw']
            #'optimizer': ['adam']
        }

        # Indices for Validation Eval
        val_indices = idx[n_train:n_train+n_val]

        # Run Grid Search
        logging.info(f"--- Tuning RNN ({result_prefix}) ---")
        rnn_df = grid_search_pyt(RNNModel, big_rnn_grid, f'{result_prefix}_RNN', in_dim,
                                X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                                val_indices, y_seq_list, max_seq_len, epochs=5)

        logging.info(f"\n--- Tuning CNN ({result_prefix}) ---")
        cnn_df = grid_search_pyt(CNNModel, big_cnn_grid, f'{result_prefix}_CNN', in_dim,
                                X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                                val_indices, y_seq_list, max_seq_len, epochs=5)

        logging.info(f"\n--- Tuning LGSTM ({result_prefix}) ---")
        lgstm_df = grid_search_pyt(LGSTMModel, big_lgstm_grid, f'{result_prefix}_LGSTM', in_dim,
                                X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                                val_indices, y_seq_list, max_seq_len, epochs=5)

        # Plot Hyperparameters
        logging.info("Generating Hyperparameter Plots...")
        model_dfs = {
            f'{result_prefix}_RNN': rnn_df,
            f'{result_prefix}_CNN': cnn_df,
            f'{result_prefix}_LGSTM': lgstm_df
        }
        plot_all_results([], model_dfs, [], "")

        # --- Select Top 2 by Utility and Top 2 by AUC for EACH model ---

        for name, df, model_cls in [('RNN', rnn_df, RNNModel), ('CNN', cnn_df, CNNModel), ('LGSTM', lgstm_df, LGSTMModel)]:
            if df.empty: continue

            # Top 2 by Utility
            top_util = df.sort_values('utility', ascending=False).head(2)
            for _, row in top_util.iterrows():
                hp = row.drop(['utility', 'auc']).to_dict()
                candidate_configs.append({
                    'name': f'{result_prefix}_{name}_BestUtil_{_}',
                    'model_cls': model_cls,
                    'model_type': name,
                    'hp': hp,
                    'score_type': 'utility',
                    'val_score': row['utility']
                })

            # Top 2 by AUC
            top_auc = df.sort_values('auc', ascending=False).head(2)
            for _, row in top_auc.iterrows():
                hp = row.drop(['utility', 'auc']).to_dict()
                is_dup = False
                for c in candidate_configs:
                    if c['hp'] == hp and c['name'].startswith(f'{result_prefix}_{name}'):
                        is_dup = True
                        break
                if not is_dup:
                    candidate_configs.append({
                        'name': f'{result_prefix}_{name}_BestAUC_{_}',
                        'model_cls': model_cls,
                        'model_type': name,
                        'hp': hp,
                        'score_type': 'auc',
                        'val_score': row['auc']
                    })

    logging.info(f"\n=== [{result_prefix}] Selected {len(candidate_configs)} Candidate Configurations for Final Training ===")
    
    # Save candidate configs to JSON for future use
    json_path = "best_model_configs.json"
    # Load existing if any (to append/merge if running multiple prefixes)
    existing_configs = []
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                existing_configs = json.load(f)
        except:
            pass
    
    # Simple merge: add new ones, replacing if name exists (though names should be unique per run usually)
    # Actually, let's just append and filter duplicates by name later if needed, 
    # but since names include prefix, they should be unique.
    
    # We can't save 'model_cls' class object to JSON, so we remove it for saving
    configs_to_save = []
    for c in candidate_configs:
        c_save = c.copy()
        if 'model_cls' in c_save:
            del c_save['model_cls']
        configs_to_save.append(c_save)
    
    # Update existing list
    # Remove old entries with same names as new ones to avoid dups
    new_names = {c['name'] for c in configs_to_save}
    existing_configs = [c for c in existing_configs if c['name'] not in new_names]
    final_save_list = existing_configs + configs_to_save
    
    try:
        with open(json_path, 'w') as f:
            json.dump(final_save_list, f, indent=4)
        logging.info(f"✓ Saved {len(final_save_list)} configurations to {json_path}")
    except Exception as e:
        logging.error(f"Failed to save configs to JSON: {e}")

    for c in candidate_configs:
        logging.info(f"  {c['name']} (Val {c['score_type'].title()}: {c['val_score']:.4f}) - HP: {c['hp']}")

    # --- Final Training of Candidates ---
    test_indices = idx[n_train+n_val:]
    final_results = []

    all_candidates_epoch_trends = {}
    final_epochs_list = [10, 25, 50, 100, 150]

    for config in candidate_configs:
        name = config['name']
        model_cls = config['model_cls']
        hp = config['hp']

        logging.info(f"\nTraining {name} (Epoch Analysis)...")

        candidate_trend = []
        model = None

        for n_ep in final_epochs_list:
            logging.info(f"  > Epochs: {n_ep}")
            # Train Fresh Model
            model, hist = train_model_pyt(model_cls, hp, in_dim, X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=n_ep)

            # Evaluate
            metrics = evaluate_sequence_model(model, X_test_seq, y_test_seq, threshold=0.5, name=name, final_act=hp.get('final_activation', 'sigmoid'))
            util_score = evaluate_utility_val(model, X_test_seq, test_indices, y_seq_list, max_seq_len, hp.get('final_activation', 'sigmoid'))

            logging.info(f"    -> Ep {n_ep}: Util={util_score:.4f}, AUC={metrics['auc']:.4f}")

            candidate_trend.append({
                'epoch': n_ep,
                'utility': util_score,
                'auc': metrics['auc'],
                'f1': metrics['f1']
            })

        all_candidates_epoch_trends[name] = candidate_trend

        # Use the result from the max epochs (150) for the final summary
        last_res = candidate_trend[-1]
        
        # Flatten HP for CSV
        res_entry = {
            'name': name,
            'dataset_method': result_prefix,
            'utility': last_res['utility'],
            'auc': last_res['auc'],
            'f1': last_res['f1']
        }
        # Add hyperparams
        for k, v in hp.items():
            res_entry[k] = v
            
        final_results.append(res_entry)

        # Plot Patient Predictions (using the model from the last iteration)
        plot_patient_prediction(model, X_test_seq, y_seq_list, test_indices, max_seq_len, hp.get('final_activation', 'sigmoid'), prefix=name)

    # --- Plot Combined Trends ---
    os.makedirs(plot_dir, exist_ok=True)

    # Utility Plot
    plt.figure(figsize=(10, 6))
    for name, trend in all_candidates_epoch_trends.items():
        eps = [t['epoch'] for t in trend]
        uts = [t['utility'] for t in trend]
        plt.plot(eps, uts, 'o-', label=name)
    plt.xlabel('Epochs')
    plt.ylabel('Utility Score')
    plt.title(f'Utility Score vs Epochs ({result_prefix})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/{result_prefix}_All_Candidates_Epochs_Utility.png')
    plt.close()

    # AUC Plot
    plt.figure(figsize=(10, 6))
    for name, trend in all_candidates_epoch_trends.items():
        eps = [t['epoch'] for t in trend]
        aucs = [t['auc'] for t in trend]
        plt.plot(eps, aucs, 's-', label=name)
    plt.xlabel('Epochs')
    plt.ylabel('AUC Score')
    plt.title(f'AUC Score vs Epochs ({result_prefix})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/{result_prefix}_All_Candidates_Epochs_AUC.png')
    plt.close()

    return final_results

def plot_preprocessing_comparison(df_results):

    """

    Plots a comparison of best utility and AUC scores for each preprocessing method.

    """

    if df_results.empty or 'dataset_method' not in df_results.columns:

        return



    os.makedirs(plot_dir, exist_ok=True)



    # Filter to get the best model per preprocessing method

    best_per_method = df_results.loc[df_results.groupby('dataset_method')['utility'].idxmax()]



    methods = best_per_method['dataset_method'].unique()



    fig, ax1 = plt.subplots(figsize=(10, 6))



    x = np.arange(len(methods))

    width = 0.35



    # Utility Bars

    rects1 = ax1.bar(x - width/2, best_per_method['utility'], width, label='Best Utility', color='skyblue', edgecolor='black')

    ax1.set_xlabel('Preprocessing Method')

    ax1.set_ylabel('Utility Score', color='blue')

    ax1.tick_params(axis='y', labelcolor='blue')

    ax1.set_xticks(x)

    ax1.set_xticklabels(methods)



    # AUC Bars (secondary axis)

    ax2 = ax1.twinx()

    rects2 = ax2.bar(x + width/2, best_per_method['auc'], width, label='Best AUC', color='lightgreen', edgecolor='black')

    ax2.set_ylabel('AUC Score', color='green')

    ax2.tick_params(axis='y', labelcolor='green')

    ax2.set_ylim(0.5, 1.0) # AUC usually > 0.5



    plt.title('Best Performance by Preprocessing Method')



    # Combined Legend

    lines1, labels1 = ax1.get_legend_handles_labels()

    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')



    plt.tight_layout()

    plt.savefig(f'{plot_dir}/preprocessing_comparison.png')

    plt.close()



if __name__ == "__main__":



    all_final_results = []



    if args.use_all_pp:

        logging.info("=== RUNNING TRAINING ON ALL PREPROCESSING METHODS ===")

        methods = ['neg1', 'mean', 'median', 'linear']



        for method in methods:

            pkl_file = f"preprocessed_{method}.pkl"

            if not os.path.exists(pkl_file):

                logging.warning(f"File {pkl_file} not found. Skipping...")

                continue



            logging.info(f"\n\n>>> PROCESSING DATASET: {method} <<<")

            data = get_data(use_cache=True, cache_path=pkl_file)

            results = run_training_pipeline(data, result_prefix=method)

            all_final_results.extend(results)



    else:

        # Default single-run behavior

        logging.info("=== RUNNING SINGLE DATASET TRAINING ===")

        data = get_data(use_cache=args.use_cache)

        results = run_training_pipeline(data, result_prefix="Default")

        all_final_results.extend(results)



    # Summary

    if all_final_results:

        res_df = pd.DataFrame(all_final_results).sort_values('utility', ascending=False)

        logging.info("\n=== FINAL TEST RESULTS SUMMARY (ALL METHODS) ===")

        logging.info(res_df.to_string(index=False))
        res_df.to_csv('final_model_comparison_all.csv', index=False)
        # Generate Preprocessing Comparison Plot
        plot_preprocessing_comparison(res_df)


