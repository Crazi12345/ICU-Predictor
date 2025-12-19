import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Data Loading (Reused from single_pass.py) ---
try:
    import kagglehub
    path = kagglehub.dataset_download("salikhussaini49/prediction-of-sepsis")
    dataset_path = path
except Exception as e:
    dataset_path = os.getcwd()

possible_paths = [
    os.path.join(dataset_path, "Dataset.csv"),
    os.path.join(dataset_path, "dataset.csv"),
    "Dataset.csv",
    "dataset.csv",
]

csv_file = None
for path in possible_paths:
    if os.path.exists(path):
        csv_file = path
        break

if csv_file:
    df_full = pd.read_csv(csv_file)
else:
    # Synthetic dataset fallback
    n_patients, max_time_steps, n_features = 100, 200, 15
    data_list = []
    for p_id in range(n_patients):
        n_steps = np.random.randint(50, max_time_steps)
        for t in range(n_steps):
            features = np.random.randn(n_features) * 0.5
            has_sepsis = np.random.rand() > 0.7
            if has_sepsis: features[:3] += t / max_time_steps * 2
            sepsis_label = 1 if (has_sepsis and t > n_steps * 0.6) else 0
            data_list.append({'patient_id': p_id, **{f'f{i}': features[i] for i in range(n_features)}, 'SepsisLabel': sepsis_label})
    df_full = pd.DataFrame(data_list)

# Preprocessing
if 'Unnamed: 0' in df_full.columns: df_full.drop(columns=['Unnamed: 0'], inplace=True)
if 'Patient_ID' in df_full.columns: df_full.rename(columns={'Patient_ID': 'patient_id'}, inplace=True)

numeric_cols = df_full.select_dtypes(include=[np.number]).columns.tolist()
for col in ['patient_id', 'SepsisLabel']:
    if col in numeric_cols: numeric_cols.remove(col)

df_full[numeric_cols] = df_full[numeric_cols].ffill().bfill().fillna(0)
scaler = StandardScaler()
df_full[numeric_cols] = scaler.fit_transform(df_full[numeric_cols])

# Sequence creation
max_seq_len = 256
all_patients = sorted(df_full['patient_id'].unique())
sample_size = max(1, len(all_patients) // 1)
sampled_patients = np.random.choice(all_patients, size=sample_size, replace=False)

X_seq_list, y_seq_list = [], []
for pid in sorted(sampled_patients):
    pat_data = df_full[df_full['patient_id'] == pid]
    X_pat = pat_data[numeric_cols].values
    y_pat = pat_data['SepsisLabel'].values

    if len(X_pat) > max_seq_len:
        X_pat, y_pat = X_pat[-max_seq_len:], y_pat[-max_seq_len:]
    else:
        pad_len = max_seq_len - len(X_pat)
        X_pat = np.vstack([X_pat, np.zeros((pad_len, len(numeric_cols)))])
        y_pat = np.concatenate([y_pat, np.zeros(pad_len)])
    X_seq_list.append(X_pat)
    y_seq_list.append(y_pat)

X_seq = torch.tensor(np.array(X_seq_list), dtype=torch.float32)
y_seq = torch.tensor(np.array(y_seq_list), dtype=torch.float32).unsqueeze(-1)

# Splits
n = len(X_seq)
n_train, n_val = int(0.8 * n), int(0.1 * n)
indices = torch.randperm(n)

train_ds = TensorDataset(X_seq[indices[:n_train]], y_seq[indices[:n_train]])
val_ds = TensorDataset(X_seq[indices[n_train:n_train+n_val]], y_seq[indices[n_train:n_train+n_val]])
test_ds = TensorDataset(X_seq[indices[n_train+n_val:]], y_seq[indices[n_train+n_val:]])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)
test_loader = DataLoader(test_ds, batch_size=64)

# --- Model Architectures ---

class RNNModel(nn.Module):
    def __init__(self, in_dim, units, dropout):
        super().__init__()
        self.rnn = nn.RNN(in_dim, units, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(units, 1)
    def forward(self, x):
        x, _ = self.rnn(x)
        return torch.sigmoid(self.fc(self.drop(x)))

class CNNModel(nn.Module):
    def __init__(self, in_dim, f1, f2, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, f1, 3, padding=1)
        self.conv2 = nn.Conv1d(f1, f2, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(f2, 1)
    def forward(self, x):
        x = x.transpose(1, 2) # (N, L, C) -> (N, C, L)
        x = torch.relu(self.conv1(x))
        x = self.drop(x)
        x = torch.relu(self.conv2(x))
        x = self.drop(x)
        x = x.transpose(1, 2) # (N, C, L) -> (N, L, C)
        return torch.sigmoid(self.fc(x))

class LSTMModel(nn.Module):
    def __init__(self, in_dim, u1, u2, dropout):
        super().__init__()
        self.lstm1 = nn.LSTM(in_dim, u1, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(u1, u2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc = nn.Linear(u2, 1)
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.drop1(x)
        x, _ = self.lstm2(x)
        x = self.drop2(x)
        return torch.sigmoid(self.fc(x))

# --- Training Utilities ---

def train_model(model, loader, val_loader, epochs=3, lr=1e-3):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            out = model(xb)
            all_preds.append(out.cpu().numpy().flatten())
            all_true.append(yb.numpy().flatten())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_preds)
    return roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.5

# --- Execution ---

print("=== STARTING PYTORCH SINGLE PASS ===")
in_dim = len(numeric_cols)

models_meta = [
    (RNNModel(in_dim, 64, 0.2), "RNN"),
    (CNNModel(in_dim, 32, 64, 0.2), "CNN"),
    (LSTMModel(in_dim, 64, 32, 0.2), "LSTM")
]

results = []
best_auc = -1
best_model_name = ""
best_model_obj = None

for model, name in models_meta:
    print(f"Training {name}...")
    auc = train_model(model, train_loader, val_loader, epochs=3)
    print(f"{name} Val AUC: {auc:.4f}")
    results.append((name, auc))
    if auc > best_auc:
        best_auc = auc
        best_model_name = name
        best_model_obj = model

print("\n=== FINAL RESULTS ===")
for name, auc in results:
    print(f"✓ {name}: {auc:.4f}")

# Final Evaluation on Test Set
if best_model_obj:
    print(f"\nBest Model: {best_model_name}. Evaluating on test set...")
    best_model_obj.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = best_model_obj(xb.to(device))
            all_preds.append(out.cpu().numpy().flatten())
            all_true.append(yb.numpy().flatten())

    y_true, y_pred = np.concatenate(all_true), np.concatenate(all_preds)
    test_auc = roc_auc_score(y_true, y_pred)
    print(f"Test AUC: {test_auc:.4f}")

    # Plotting logic simplified
    plt.figure(figsize=(10, 4))
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, label=f'AUC={test_auc:.3f}')
    plt.title(f'Test ROC - {best_model_name}')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend()
    plt.show()
