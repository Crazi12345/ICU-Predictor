# Setup: imports and kagglehub download
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import SimpleRNN, LSTM, Conv1D, TimeDistributed, Dense, Dropout
from tensorflow.keras.models import Sequential
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, confusion_matrix,
                             precision_score, recall_score, f1_score)

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Try to download dataset
try:
    import kagglehub
    path = kagglehub.dataset_download("salikhussaini49/prediction-of-sepsis")
    print(f"Dataset downloaded to: {path}")
    dataset_path = path
except Exception as e:
    print(f"Download failed ({e}). Using local path or expecting Dataset.csv in working dir.")
    dataset_path = os.getcwd()

# Try multiple possible paths for Dataset.csv
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
        print(f"✓ Found dataset at: {csv_file}")
        break

if csv_file is None:
    print("✗ Dataset.csv not found in any of these locations:")
    for p in possible_paths:
        print(f"  - {p}")
    print("\nSearching for any .csv files in current dir and home...")
    for root, dirs, files in os.walk(os.getcwd()):
        for file in files:
            if file.endswith('.csv'):
                print(f"  Found: {os.path.join(root, file)}")
        break  # only check top level
    df_full = None
else:
    df_full = pd.read_csv(csv_file)
    print(f"✓ Loaded Dataset: {df_full.shape[0]} rows, {df_full.shape[1]} columns")
    # Show first 10 column names
    print(f"Columns: {df_full.columns.tolist()[:10]}...")
    print(f"First 3 rows:\n{df_full.head(3)}")
# Data preprocessing: FAST global imputation and scaling (for initial exploration)
# Note: Full grid search version uses per-patient imputation
if df_full is None:
    print("\n⚠ Creating synthetic dataset...")
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
            data_list.append({'Patient_ID': p_id, **{f'feature_{i}': features[i] for i in range(n_features)}, 'SepsisLabel': sepsis_label})
    df_full = pd.DataFrame(data_list)
    print(f"✓ Synthetic dataset: {df_full.shape[0]} rows, {df_full.shape[1]} columns")

if df_full is not None:
    # Drop unnamed index and standardize patient ID column
    if 'Unnamed: 0' in df_full.columns:
        df_full = df_full.drop(columns=['Unnamed: 0'])
    if 'Patient_ID' in df_full.columns:
        df_full.rename(columns={'Patient_ID': 'patient_id'}, inplace=True)

    print(f"Shape: {df_full.shape}")

    # Get numeric feature columns (exclude patient_id and SepsisLabel)
    df_imputed = df_full.copy()
    numeric_cols = df_imputed.select_dtypes(
        include=[np.number]).columns.tolist()
    if 'patient_id' in numeric_cols:
        numeric_cols.remove('patient_id')
    if 'SepsisLabel' in numeric_cols:
        numeric_cols.remove('SepsisLabel')

    print(f"✓ Features: {len(numeric_cols)}, Label: SepsisLabel")

    # FAST global imputation: ffill, bfill, then global mean
    print("⏳ Imputing missing values (global)...")
    for col in numeric_cols:
        df_imputed[col] = df_imputed[col].ffill(
        ).bfill().fillna(df_imputed[col].mean())

    print(f"✓ Imputation complete. NaNs remaining: {df_imputed[numeric_cols].isna().sum().sum()}")

    # Global scaling (StandardScaler on all data)
    print("⏳ Scaling features (global)...")
    df_scaled = df_imputed.copy()
    scaler = StandardScaler()
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

    print("✓ Scaling complete.")
    scaler_dict = {'global': scaler}  # For compatibility
else:
    print("ERROR: df_full is None.")
# Sequence creation: variable-length sequences per patient (FAST: using 33% of patients for testing)
if df_full is not None:
    # Extract sequences per patient
    max_seq_len = 256

    # Sample only 50% of patients for faster testing
    all_patients = sorted(df_scaled['patient_id'].unique())
    sample_size = max(1, len(all_patients) // 10)  # At least 1 patient
    sampled_patients = np.random.choice(
        all_patients, size=sample_size, replace=False)
    print(f"⏳ Using {sample_size} out of {len(all_patients)} patients for faster testing...")

    X_seq_list = []
    y_seq_list = []
    patient_ids = []

    for patient_id in sorted(sampled_patients):
        mask = df_scaled['patient_id'] == patient_id
        # shape: (time_steps, features)
        X_pat = df_scaled.loc[mask, numeric_cols].values
        y_pat = df_scaled.loc[mask, 'SepsisLabel'].values if 'SepsisLabel' in df_scaled.columns else np.ones(
            mask.sum())  # shape: (time_steps,)

        # Post-pad or truncate to max_seq_len
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

    # shape: (n_patients, max_seq_len, n_features)
    X_seq = np.array(X_seq_list)
    y_seq = np.array(y_seq_list)  # shape: (n_patients, max_seq_len)

    print(f"Sequence data created: X_seq {X_seq.shape}, y_seq {y_seq.shape}")

    # Train/val/test split (80/10/10)
    n = len(X_seq)
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    X_train_seq = X_seq[idx[:n_train]]
    y_train_seq = y_seq[idx[:n_train]]
    X_val_seq = X_seq[idx[n_train:n_train+n_val]]
    y_val_seq = y_seq[idx[n_train:n_train+n_val]]
    X_test_seq = X_seq[idx[n_train+n_val:]]
    y_test_seq = y_seq[idx[n_train+n_val:]]

    # Also keep unpadded sequences for test set (for PhysioNet scoring)
    y_test_list = [y_seq_list[i][:np.sum(y_seq_list[i] != 0) + 1] if np.sum(
        y_seq_list[i] != 0) > 0 else y_seq_list[i] for i in range(len(patient_ids))]

    print(f"Train: {X_train_seq.shape}, Val: {X_val_seq.shape}, Test: {X_test_seq.shape}")
    print(f"✓ Note: Using 33% of patients for faster iteration during testing")
else:
    print("ERROR: Cannot create sequences. df_full is None.")
# Evaluation functions (MUST COME BEFORE TUNING CELL)


def plot_confusion_matrix(cm, classes=['Neg', 'Pos'], title='Confusion matrix', cmap=plt.cm.Blues):
    """Plot confusion matrix."""
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def evaluate_sequence_model(model, X_seq, y_seq, threshold=0.5, name='model'):
    """Evaluate sequence model on flattened predictions."""
    preds = model.predict(X_seq, verbose=0)
    preds_flat = preds.reshape(-1)
    y_flat = y_seq.reshape(-1)

    auc = roc_auc_score(y_flat, preds_flat)
    avg_prec = average_precision_score(y_flat, preds_flat)
    y_pred_bin = (preds_flat >= threshold).astype(int)
    prec = precision_score(y_flat, y_pred_bin, zero_division=0)
    rec = recall_score(y_flat, y_pred_bin, zero_division=0)
    f1 = f1_score(y_flat, y_pred_bin, zero_division=0)
    cm = confusion_matrix(y_flat, y_pred_bin)

    print(f"Evaluation for {name}:")
    print(f"  AUC: {auc:.4f}, AP: {avg_prec:.4f}")
    print(f"  Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print('  Confusion matrix:\n', cm)

    fpr, tpr, _ = roc_curve(y_flat, preds_flat)
    precision_vals, recall_vals, _ = precision_recall_curve(y_flat, preds_flat)

    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1)
    plt.plot(fpr, tpr, label=f'AUC={auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC Curve ({name})')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(recall_vals, precision_vals, label=f'AP={avg_prec:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall ({name})')
    plt.legend()

    plt.subplot(1, 3, 3)
    plot_confusion_matrix(
        cm, classes=['Neg', 'Pos'], title=f'Confusion Matrix ({name})')

    plt.tight_layout()
    plt.show()

    return {'auc': auc, 'ap': avg_prec, 'precision': prec, 'recall': rec, 'f1': f1, 'confusion_matrix': cm}

# Model builders for two-stage tuning


input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
os.makedirs('/tmp/icu_tune', exist_ok=True)


def get_optimizer(opt_name, lr):
    """Get optimizer by name."""
    if (opt_name or '').lower() == 'adam':
        return Adam(lr)
    return Adam(lr)


def map_output_to_prob(preds, final_act):
    """Map model outputs to probability-like [0,1] range based on activation."""
    preds = np.asarray(preds)
    if final_act == 'tanh':
        return (preds + 1.0) / 2.0
    if final_act == 'relu':
        return np.clip(preds, 0.0, 1.0)
    return preds


def safe_roc_auc(y_true_flat, pred_flat):
    """Safe AUC computation."""
    try:
        return float(roc_auc_score(y_true_flat, pred_flat))
    except Exception:
        return float('nan')


def build_rnn_model(hp):
    """Build RNN model with hyperparams hp."""
    final_act = hp.get('final_activation', 'sigmoid')
    loss_fn = hp.get('loss', 'binary_crossentropy')
    m = Sequential([
        tf.keras.Input(shape=input_shape),
        SimpleRNN(hp['units'], return_sequences=True),
        Dropout(hp['dropout']),
        TimeDistributed(Dense(1, activation=final_act))
    ])
    opt = get_optimizer(hp.get('optimizer', 'adam'), hp.get('lr', 1e-3))
    m.compile(optimizer=opt, loss=loss_fn, metrics=[
              'accuracy', Precision(), Recall(), AUC(name='auc')])
    return m


def build_cnn_model(hp):
    """Build CNN model with hyperparams hp."""
    final_act = hp.get('final_activation', 'sigmoid')
    loss_fn = hp.get('loss', 'binary_crossentropy')
    m = Sequential([
        tf.keras.Input(shape=input_shape),
        Conv1D(hp['f1'], 3, activation='relu', padding='same'),
        Dropout(hp['dropout']),
        Conv1D(hp['f2'], 3, activation='relu', padding='same'),
        Dropout(hp['dropout']),
        TimeDistributed(Dense(1, activation=final_act))
    ])
    opt = get_optimizer(hp.get('optimizer', 'adam'), hp.get('lr', 1e-3))
    m.compile(optimizer=opt, loss=loss_fn, metrics=[
              'accuracy', Precision(), Recall(), AUC(name='auc')])
    return m


def build_lgstm_model(hp):
    """Build stacked LSTM model with hyperparams hp."""
    final_act = hp.get('final_activation', 'sigmoid')
    loss_fn = hp.get('loss', 'binary_crossentropy')
    m = Sequential([
        tf.keras.Input(shape=input_shape),
        LSTM(hp['u1'], return_sequences=True),
        Dropout(hp['dropout']),
        LSTM(hp['u2'], return_sequences=True),
        Dropout(hp['dropout']),
        TimeDistributed(Dense(1, activation=final_act))
    ])
    opt = get_optimizer(hp.get('optimizer', 'adam'), hp.get('lr', 1e-3))
    m.compile(optimizer=opt, loss=loss_fn, metrics=[
              'accuracy', Precision(), Recall(), AUC(name='auc')])
    return m


# SINGLE PASS: Train 3 models with fixed hyperparams, pick best
print("=== SINGLE PASS MODEL TRAINING ===\n")

# Fixed hyperparams for fast single-pass testing
rnn_hp = {
    'units': 64, 'dropout': 0.2, 'lr': 1e-3, 'batch_size': 64,
    'final_activation': 'relu', 'loss': 'binary_crossentropy', 'optimizer': 'adam'
}

cnn_hp = {
    'f1': 32, 'f2': 64, 'dropout': 0.2, 'lr': 1e-3, 'batch_size': 64,
    'final_activation': 'relu', 'loss': 'binary_crossentropy', 'optimizer': 'adam'
}

lgstm_hp = {
    'u1': 64, 'u2': 32, 'dropout': 0.2, 'lr': 1e-3, 'batch_size': 64,
    'final_activation': 'relu', 'loss': 'binary_crossentropy', 'optimizer': 'adam'
}

os.makedirs('/tmp/icu_tune', exist_ok=True)


def train_and_evaluate(build_fn, hp, name, epochs=5):
    """Train model and return validation AUC."""
    print(f"\n[{name}] Training for {epochs} epochs...")
    m = build_fn(hp)

    try:
        hist = m.fit(X_train_seq, y_train_seq,
                     validation_data=(X_val_seq, y_val_seq),
                     epochs=epochs, batch_size=hp['batch_size'], verbose=0)

        # Get validation AUC
        if 'val_auc' in hist.history:
            val_auc = max(hist.history['val_auc'])
        else:
            preds = m.predict(X_val_seq, verbose=0)[:, :, 0]
            probs = map_output_to_prob(
                preds, hp.get('final_activation', 'sigmoid'))
            val_auc = safe_roc_auc(y_val_seq.reshape(-1), probs)

        print(f"[{name}] Validation AUC: {val_auc:.4f}")

        # Save weights
        try:
            m.save_weights(f'/tmp/icu_tune/{name}_weights.h5')
        except Exception:
            pass

        return m, val_auc, hist
    except Exception as e:
        print(f"[{name}] Training failed: {e}")
        return None, float('nan'), None


# Train all 3 models
print("Training models...")
rnn_model, rnn_auc, rnn_hist = train_and_evaluate(
    build_rnn_model, rnn_hp, 'RNN', epochs=3)
cnn_model, cnn_auc, cnn_hist = train_and_evaluate(
    build_cnn_model, cnn_hp, 'CNN', epochs=3)
lgstm_model, lgstm_auc, lgstm_hist = train_and_evaluate(
    build_lgstm_model, lgstm_hp, 'LGSTM', epochs=3)

# Find best model
models_results = []
if rnn_model is not None:
    models_results.append(('RNN', rnn_model, rnn_auc, rnn_hp))
if cnn_model is not None:
    models_results.append(('CNN', cnn_model, cnn_auc, cnn_hp))
if lgstm_model is not None:
    models_results.append(('LGSTM', lgstm_model, lgstm_auc, lgstm_hp))

if models_results:
    best_name, best_model, best_auc, best_hp = max(
        models_results, key=lambda x: x[2])
    print(f"\n✓ Best model: {best_name} (val AUC: {best_auc:.4f})")
else:
    print("\n✗ All models failed!")
    best_model = None
    best_name = None
    best_hp = None

# Summary
print("\n=== SINGLE PASS RESULTS ===")
for name, auc in [('RNN', rnn_auc), ('CNN', cnn_auc), ('LGSTM', lgstm_auc)]:
    status = "✓" if auc == auc else "✗"  # NaN check
    print(f"{status} {name}: val_auc={auc:.4f}")

# Optional: Train best model for more epochs and evaluate on test set
if best_model is not None:
    print("\n=== FINAL EVALUATION ===\n")
    print(f"Evaluating {best_name} on test set...")

    # Evaluate on test set
    metrics = evaluate_sequence_model(best_model, X_test_seq, y_test_seq,
                                      threshold=0.5, name=f'{best_name}_test')

    # Save results
    results_df = pd.DataFrame([{
        'model': best_name,
        'auc': metrics['auc'],
        'ap': metrics['ap'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1']
    }])
    results_df.to_csv('/tmp/icu_tune/single_pass_results.csv', index=False)
    print(f"\n✓ Results saved to /tmp/icu_tune/single_pass_results.csv")
else:
    print("\n✗ No best model available for evaluation.")
