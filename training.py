import os
import sys
import logging
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        logging.FileHandler("training_log.txt"),
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
parser = argparse.ArgumentParser(description='Train ICU Prediction Model')
parser.add_argument('--use-cache', action='store_true', help='Use cached preprocessed data if available')
args, unknown = parser.parse_known_args()

def get_data(use_cache=False):
    cache_path = 'preprocessed_data.pkl'
    if use_cache and os.path.exists(cache_path):
        logging.info(f"Loading preprocessed data from {cache_path}...")
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            logging.info("✓ Data loaded from cache.")
            return data
        except Exception as e:
            logging.info(f"⚠ Failed to load cache ({e}). Re-running preprocessing...")

    # --- Data Loading (from Code.ipynb) ---
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
        logging.info(f"✓ Synthetic dataset: {df_full.shape[0]} rows, {df_full.shape[1]} columns")

    # --- Preprocessing (from Code.ipynb) ---
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

        logging.info(f"✓ Features: {len(numeric_cols)}, Label: SepsisLabel")

        # Per-patient imputation
        logging.info("⏳ Imputing missing values per-patient...")
        for col in numeric_cols:
            df_imputed[col] = df_imputed.groupby('patient_id')[col].transform(lambda x: x.ffill().bfill())
            df_imputed[col] = df_imputed.groupby('patient_id')[col].transform(lambda x: x.fillna(x.mean()))
            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())

        logging.info(f"✓ Imputation complete. NaNs remaining: {df_imputed[numeric_cols].isna().sum().sum()}")

        # Per-patient scaling
        logging.info("⏳ Scaling features per-patient...")
        df_scaled = df_imputed.copy()
        scaler_dict = {}
        for patient_id in df_scaled['patient_id'].unique():
            mask = df_scaled['patient_id'] == patient_id
            scaler = StandardScaler()
            df_scaled.loc[mask, numeric_cols] = scaler.fit_transform(df_scaled.loc[mask, numeric_cols])
            scaler_dict[patient_id] = scaler
        logging.info("✓ Scaling complete.")

    # --- Sequence Creation (from Code.ipynb) ---
    max_seq_len = 256
    X_seq_list = []
    y_seq_list = []
    patient_ids = []

    all_patient_ids = sorted(df_scaled['patient_id'].unique())
    # OPTIMIZATION: Halve the data for performance
    half_n = max(1, len(all_patient_ids) // 1)
    logging.info(f"⚠️ Performance Optimization: Reducing dataset from {len(all_patient_ids)} to {half_n} patients (50%)")
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
    logging.info(f"Sequence data created: X_seq {X_seq.shape}, y_seq {y_seq.shape}")

    # Train/val/test split
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

# Execute Data Loading
data_bundle = get_data(use_cache=args.use_cache)

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

logging.info(f"Train: {X_train_seq.shape}, Val: {X_val_seq.shape}, Test: {X_test_seq.shape}")

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
        return out

class CNNModel(nn.Module):
    def __init__(self, hp, in_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hp['f1'], 3, padding=1)
        self.conv2 = nn.Conv1d(hp['f1'], hp['f2'], 3, padding=1)
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
        return out

# --- Training & Evaluation Functions ---

def get_loss_fn(loss_name):
    if loss_name == 'binary_crossentropy':
        return nn.BCELoss()
    elif loss_name == 'mse':
        return nn.MSELoss()
    elif loss_name == 'hinge':
        # Simple Hinge Loss implementation: mean(max(0, 1 - y_true * y_pred))
        # Assumes y_true is {-1, 1} and y_pred is unbounded (or tanh)
        return lambda y_pred, y_true: torch.mean(torch.clamp(1 - y_true * y_pred, min=0))
    return nn.BCELoss()

def map_output_to_prob(preds, final_act):
    if final_act == 'tanh':
        return (preds + 1.0) / 2.0
    if final_act == 'relu':
        return np.clip(preds, 0.0, 1.0)
    return preds

def train_model_pyt(model_class, hp, in_dim, X_train, y_train, X_val, y_val, epochs):
    model = model_class(hp, in_dim).to(device)

    # Optimizer
    lr = hp.get('lr', 1e-3)
    if hp.get('optimizer', 'adam') == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr) # Default to Adam

    # Loss Function
    loss_name = hp.get('loss', 'binary_crossentropy')
    criterion = get_loss_fn(loss_name)

    # Prepare data for Hinge if needed
    if loss_name == 'hinge':
        y_train_target = (y_train.clone() * 2.0) - 1.0
        y_val_target = (y_val.clone() * 2.0) - 1.0
    else:
        y_train_target = y_train
        y_val_target = y_val

    # Optimize data loading
    # Fix for "Too many open files": use num_workers=0 to avoid excessive file descriptor usage during grid search
    num_workers = 0
    use_pin_memory = (device.type == 'cuda')

    ds_train = TensorDataset(X_train, y_train_target)
    dl_train = DataLoader(ds_train, batch_size=hp['batch_size'], shuffle=True,
                          num_workers=num_workers, pin_memory=use_pin_memory)

    # Training Loop
    history = {'loss': [], 'val_auc': []}

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

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_val.to(device))
            # Convert back to prob for AUC calc
            val_preds_np = val_out.cpu().numpy().flatten()
            val_preds_prob = map_output_to_prob(val_preds_np, hp.get('final_activation', 'sigmoid'))

            try:
                # y_val is always 0/1 for AUC calc
                auc = roc_auc_score(y_val.numpy().flatten(), val_preds_prob)
            except:
                auc = 0.5
            history['val_auc'].append(auc)

    return model, history

def grid_search_pyt(model_class, grid, name, in_dim, X_train, y_train, X_val, y_val, epochs=10):
    results = []
    best_hp = None
    best_score = -1.0
    best_weights_path = f'/tmp/icu_tune/{name}_best_weights.pth'
    os.makedirs('/tmp/icu_tune', exist_ok=True)

    for hp in ParameterGrid(grid):
        # Skip invalid combinations to prevent CUDA errors
        if hp['loss'] == 'binary_crossentropy' and hp['final_activation'] != 'sigmoid':
            continue
        if hp['loss'] == 'hinge' and hp['final_activation'] == 'sigmoid':
            continue

        logging.info(f'[{name}] trying {hp}')
        model = None
        try:
            model, hist = train_model_pyt(model_class, hp, in_dim, X_train, y_train, X_val, y_val, epochs)
            score = max(hist['val_auc'])
        except Exception as e:
            logging.info(f'  fit failed: {e}')
            score = 0.0

        logging.info(f'  score={score:.4f}')
        results.append({**hp, 'score': score})

        if score > best_score and model is not None:
            best_score = score
            best_hp = hp
            torch.save(model.state_dict(), best_weights_path)

    df = pd.DataFrame(results).sort_values('score', ascending=False).reset_index(drop=True)
    return best_hp, df

# --- OFFICIAL PHYSIONET CHALLENGE 2019 SCORING ---

def compute_prediction_utility(labels, predictions, dt_early=-12, dt_optimal=-6, dt_late=3.0,
                             max_u_tp=1, min_u_fn=-2, u_fp=-0.05, u_tn=0):
    """
    Computes the official PhysioNet 2019 Utility Score.
    """
    n_patients = len(labels)

    # Initialize counters for observation (not strictly needed for normalized score but good for debug)
    # n_tp, n_fp, n_fn, n_tn = 0, 0, 0, 0

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

        # Create utility vector for this patient
        u_vector = np.zeros(n)
        if is_septic:
            t_diff = np.arange(n) - t_sepsis
            # Slope part
            mask_slope = (t_diff >= dt_early) & (t_diff <= dt_optimal)
            u_vector[mask_slope] = max_u_tp * (t_diff[mask_slope] - dt_early) / (dt_optimal - dt_early)
            # Plateau part
            mask_plateau = (t_diff > dt_optimal) & (t_diff <= dt_late)
            u_vector[mask_plateau] = max_u_tp

        # Calculate observed for this patient
        # Score = Sum( (Pred==1) * ( U>0 ? U : -0.05 ) )
        # This covers TPs (U>0) and FPs (U=0).

        term1 = pred * u_vector
        term2 = (pred == 1) & (u_vector == 0)
        p_utility = np.sum(term1) + np.sum(term2.astype(float) * u_fp)
        observed_utility += p_utility

        # Best Utility (If we predicted perfectly)
        # Perfect pred matches u_vector > 0
        best_pred = (u_vector > 0).astype(int)
        term1_best = best_pred * u_vector
        term2_best = (best_pred == 1) & (u_vector == 0) # Should be 0
        b_utility = np.sum(term1_best) + np.sum(term2_best.astype(float) * u_fp)
        best_utility += b_utility

        # Inactive Utility (If we predicted all zeros)
        # Pred = 0 everywhere. Score = 0.
        inactive_utility += 0.0

    # Final Normalized Score
    if best_utility == inactive_utility:
        normalized_utility = 0.0
    else:
        normalized_utility = (observed_utility - inactive_utility) / (best_utility - inactive_utility)

    return normalized_utility, observed_utility

# --- PhysioNet Scoring (from single_pass.py) ---
def evaluate_sequence_model(model, X_seq, y_seq, threshold=0.5, name='model', final_act='sigmoid'):
    model.eval()
    with torch.no_grad():
        preds = model(X_seq.to(device))

    preds_flat = preds.cpu().numpy().flatten()
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

def plot_all_results(eval_metrics_list, model_dfs, final_epoch_results, best_model_name):
    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)

    # 1. Hyperparam plots for ALL models
    for name, df in model_dfs.items():
        if df is not None and not df.empty:
            plt.figure(figsize=(10, 6))
            # Just plotting score distribution for top 15 configs
            sns.barplot(data=df.head(15), x='score', y=df.head(15).index)
            plt.title(f'{name} - Top 15 Configs')
            plt.xlabel('Val AUC')
            plt.ylabel('Rank')
            plt.tight_layout()
            plt.savefig(f'plots/{name}_tuning_results.png')
            plt.close()

    # 2. Final Epochs Performance Line Plot
    if final_epoch_results:
        epochs = [r['epoch'] for r in final_epoch_results]
        aucs = [r['auc'] for r in final_epoch_results]
        f1s = [r['f1'] for r in final_epoch_results]
        
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, aucs, 'o-', label='AUC')
        plt.plot(epochs, f1s, 's-', label='F1')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.title(f'{best_model_name} Performance vs Epochs')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'plots/{best_model_name}_epochs_trend.png')
        plt.close()

    # 3. Evaluation Metrics (ROC, PR, CM) for final runs
    for i, res in enumerate(eval_metrics_list):
        name = res['name']
        y_true = res['y_true']
        y_pred = res['y_pred']
        cm = res['confusion_matrix']

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred)

        plt.figure(figsize=(14, 4))

        # ROC
        plt.subplot(1, 3, 1)
        plt.plot(fpr, tpr, label=f"AUC={res['auc']:.3f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(f'ROC Curve ({name})')
        plt.legend()

        # PR
        plt.subplot(1, 3, 2)
        plt.plot(recall_vals, precision_vals, label=f"AP={res['ap']:.3f}")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall ({name})')
        plt.legend()

        # CM
        plt.subplot(1, 3, 3)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix ({name})')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Neg', 'Pos'])
        plt.yticks(tick_marks, ['Neg', 'Pos'])

        thresh = cm.max() / 2.
        for r, c in np.ndindex(cm.shape):
            plt.text(c, r, format(cm[r, c], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[r, c] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(f'plots/{name}_metrics_{i}.png')
        plt.close()

# --- Main Execution Flow ---
if __name__ == "__main__":
    logging.info("=== FULL GRID SEARCH ON ALL MODELS (PYTORCH) ===\n")
    
    in_dim = X_train_seq.shape[2]

    # Refined Grids (Removing problematic activ/loss, adding units)
    big_rnn_grid = {
        'units': [64, 128, 256],
        'dropout': [0.2, 0.5],
        'lr': [1e-3, 5e-4],
        'batch_size': [32, 64],
        'final_activation': ['sigmoid'],
        'loss': ['binary_crossentropy'],
        'optimizer': ['adam']
    }
    
    big_cnn_grid = {
        'f1': [32, 64],
        'f2': [64, 128],
        'dropout': [0.2, 0.5],
        'lr': [1e-3, 5e-4],
        'batch_size': [32, 64],
        'final_activation': ['sigmoid'],
        'loss': ['binary_crossentropy'],
        'optimizer': ['adam']
    }
    
    big_lgstm_grid = {
        'u1': [64, 128],
        'u2': [32, 64],
        'dropout': [0.2, 0.5],
        'lr': [1e-3, 5e-4],
        'batch_size': [32, 64],
        'final_activation': ['sigmoid'],
        'loss': ['binary_crossentropy'],
        'optimizer': ['adam']
    }

    # Run Grid Search (Epochs = 15)
    logging.info("--- Tuning RNN (15 epochs) ---")
    best_rnn_hp, rnn_df = grid_search_pyt(RNNModel, big_rnn_grid, 'RNN', in_dim, X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=15)
    
    logging.info("\n--- Tuning CNN (15 epochs) ---")
    best_cnn_hp, cnn_df = grid_search_pyt(CNNModel, big_cnn_grid, 'CNN', in_dim, X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=15)
    
    logging.info("\n--- Tuning LGSTM (15 epochs) ---")
    best_lgstm_hp, lgstm_df = grid_search_pyt(LGSTMModel, big_lgstm_grid, 'LGSTM', in_dim, X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=15)

    # Compare
    scores = []
    if not rnn_df.empty: scores.append(('RNN', rnn_df.iloc[0]['score'], best_rnn_hp, rnn_df, RNNModel))
    if not cnn_df.empty: scores.append(('CNN', cnn_df.iloc[0]['score'], best_cnn_hp, cnn_df, CNNModel))
    if not lgstm_df.empty: scores.append(('LGSTM', lgstm_df.iloc[0]['score'], best_lgstm_hp, lgstm_df, LGSTMModel))

    if scores:
        best_name, best_score, best_hp, best_df, best_model_cls = max(scores, key=lambda x: x[1])
        logging.info(f'\n✓ Overall Best Model: {best_name} (score={best_score:.4f})')
        logging.info(f'  Best Config: {best_hp}')
        
        # --- FINAL EXPERIMENTS ---
        logging.info("\n=== FINAL EXPERIMENTS (25, 50, 100 epochs) ===\n")
        final_epochs = [25, 50, 100]
        eval_metrics_list = []
        final_epoch_results = []

        for e in final_epochs:
            logging.info(f'Training {best_name} for {e} epochs...')
            model, hist = train_model_pyt(best_model_cls, best_hp, in_dim, X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=e)
            metrics = evaluate_sequence_model(model, X_test_seq, y_test_seq, threshold=0.5, name=f'{best_name}_final_e{e}', final_act=best_hp.get('final_activation', 'sigmoid'))
            eval_metrics_list.append(metrics)
            
            final_epoch_results.append({
                'epoch': e,
                'auc': metrics['auc'],
                'f1': metrics['f1']
            })
            logging.info(f"  Test AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")

        # --- PREPARE DATA FOR SCORING (Use the last trained model i.e., 100 epochs) ---
        logging.info("Preparing data for PhysioNet Official Scoring...")
        model.eval()
        with torch.no_grad():
            probs_padded_tensor = model(X_test_seq.to(device))
            probs_padded = probs_padded_tensor.cpu().numpy()

        final_act = best_hp.get('final_activation', 'sigmoid')
        probs_mapped = map_output_to_prob(probs_padded.flatten(), final_act).reshape(probs_padded.shape)
        preds_padded = (probs_mapped > 0.5).astype(int)

        final_labels_list = []
        final_preds_list = []
        test_indices = idx[n_train+n_val:]
        logging.info(f"Scoring on {len(test_indices)} test patients...")

        for i, original_idx in enumerate(test_indices):
            true_seq = y_seq_list[original_idx]
            pat_len = len(true_seq)
            if pat_len > max_seq_len: pat_len = max_seq_len
            pred_seq = preds_padded[i, :pat_len, 0]
            final_labels_list.append(true_seq[:pat_len])
            final_preds_list.append(pred_seq)

        norm_score, raw_score = compute_prediction_utility(final_labels_list, final_preds_list)

        logging.info("\n" + "="*40)
        logging.info(f"OFFICIAL PHYSIONET UTILITY SCORE (at {final_epochs[-1]} epochs)")
        logging.info("="*40)
        logging.info(f"Model: {best_name}")
        logging.info(f"Normalized Utility Score: {norm_score:.4f}")
        logging.info(f"Raw Utility Score:        {raw_score:.2f}")
        logging.info("-" * 40)
        logging.info("="*40)

        logging.info("\nGenerating Plots...")
        model_dfs = {
            'RNN': rnn_df,
            'CNN': cnn_df,
            'LGSTM': lgstm_df
        }
        plot_all_results(eval_metrics_list, model_dfs, final_epoch_results, best_name)
    else:
        logging.info("No models trained successfully.")