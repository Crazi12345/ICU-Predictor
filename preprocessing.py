import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def load_raw_data():
    """Loads the dataset from CSV, downloading if necessary."""
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

    if csv_file:
        df = pd.read_csv(csv_file)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        if 'Patient_ID' in df.columns:
            df.rename(columns={'Patient_ID': 'patient_id'}, inplace=True)
        return df
    else:
        logging.error("Could not find Dataset.csv")
        return None

def process_and_save(df_input, method_name, imputation_fn):
    logging.info(f"\n=== Running Preprocessing Method: {method_name} ===")
    df = df_input.copy()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'patient_id' in numeric_cols: numeric_cols.remove('patient_id')
    if 'SepsisLabel' in numeric_cols: numeric_cols.remove('SepsisLabel')

    # Apply specific imputation strategy
    df_clean = imputation_fn(df, numeric_cols)
    
    # Standard Scaling (Robust to NaNs if any remain, though they shouldn't)
    logging.info("⏳ Scaling features per-patient...")
    
    # Helper for scaling
    def robust_scale(x):
        # If all -1 or all same, std is 0
        if len(x) < 2: return x
        mean = np.mean(x)
        std = np.std(x)
        if std == 0: return x - mean
        return (x - mean) / std

    # For methods that might leave -1s, we should decide if we scale them. 
    # Usually -1 is a flag, scaling it mixes it. 
    # But for neural nets, valid inputs must be scaled. 
    # Strategy: Scale everything. The model will learn that the scaled value of -1 is the "missing" signal.
    
    for col in numeric_cols:
        df_clean[col] = df_clean.groupby('patient_id')[col].transform(robust_scale)

    # Sequence Creation
    logging.info("⏳ Creating sequences...")
    max_seq_len = 256
    X_seq_list = []
    y_seq_list = []
    patient_ids = []
    
    all_patient_ids = sorted(df_clean['patient_id'].unique())
    # Full dataset
    patient_ids_to_use = all_patient_ids

    for patient_id in patient_ids_to_use:
        mask = df_clean['patient_id'] == patient_id
        X_pat = df_clean.loc[mask, numeric_cols].values
        y_pat = df_clean.loc[mask, 'SepsisLabel'].values if 'SepsisLabel' in df_clean.columns else np.ones(mask.sum())

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

    # Train/Val/Test Split (Deterministic)
    np.random.seed(42)
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

    filename = f"data/preprocessed_{method_name}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(data_bundle, f)
    logging.info(f"✓ Saved to {filename}")

# --- Imputation Strategies ---

def strategy_neg1(df, cols):
    """
    1. Per-patient ffill/bfill.
    2. Fill remaining NaNs with -1.0.
    """
    df_out = df.copy()
    logging.info("  [Strategy: Neg1] ffill/bfill -> fillna(-1)")
    for col in cols:
        df_out[col] = df_out.groupby('patient_id')[col].transform(lambda x: x.ffill().bfill())
    
    df_out[cols] = df_out[cols].fillna(-1.0)
    return df_out

def strategy_mean(df, cols):
    """
    1. Per-patient Mean.
    2. If patient has NO values, fill with -1.0.
    """
    df_out = df.copy()
    logging.info("  [Strategy: Mean] Patient Mean -> fillna(-1)")
    
    for col in cols:
        # Fill with patient mean
        df_out[col] = df_out.groupby('patient_id')[col].transform(lambda x: x.fillna(x.mean()))
    
    # Remaining are patients who had ALL NaNs for that col
    df_out[cols] = df_out[cols].fillna(-1.0)
    return df_out

def strategy_median(df, cols):
    """
    1. Per-patient Median.
    2. If patient has NO values, fill with -1.0.
    """
    df_out = df.copy()
    logging.info("  [Strategy: Median] Patient Median -> fillna(-1)")
    
    for col in cols:
        df_out[col] = df_out.groupby('patient_id')[col].transform(lambda x: x.fillna(x.median()))
    
    df_out[cols] = df_out[cols].fillna(-1.0)
    return df_out

def strategy_linear(df, cols):
    """
    1. Linear Interpolation.
    2. ffill/bfill for edges.
    3. Fill remaining with -1.0.
    """
    df_out = df.copy()
    logging.info("  [Strategy: Linear] Interpolate(linear) -> ffill/bfill -> fillna(-1)")
    
    # Interpolation needs to be done per group, but pandas groupby.apply is slow.
    # We can rely on the fact that patient_id sorts the data usually, but safer to group.
    # To speed up, we might iterate if dataset is small, or use transform.
    # Linear interpolation requires index to be time-aware or just incremental.
    
    for col in cols:
        df_out[col] = df_out.groupby('patient_id')[col].transform(
            lambda x: x.interpolate(method='linear').ffill().bfill()
        )
        
    df_out[cols] = df_out[cols].fillna(-1.0)
    return df_out

if __name__ == "__main__":
    df = load_raw_data()
    if df is not None:
        process_and_save(df, "neg1", strategy_neg1)
        process_and_save(df, "mean", strategy_mean)
        process_and_save(df, "median", strategy_median)
        process_and_save(df, "linear", strategy_linear)
        logging.info("\nAll preprocessing methods complete.")
