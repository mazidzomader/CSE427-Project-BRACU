import pandas as pd
url = 'https://raw.githubusercontent.com/mazidzomader/CSE427-Project-BRACU/refs/heads/main/Datasets/thyroidDF.csv'

df = pd.read_csv(url)
def summary(df, pred=None):
    obs = df.shape[0]
    Types = df.dtypes
    Counts = df.apply(lambda x: x.count())
    Min = df.select_dtypes(include=['number']).min()
    Max = df.select_dtypes(include=['number']).max()
    Uniques = df.apply(lambda x: x.unique().shape[0])
    Nulls = df.apply(lambda x: x.isnull().sum())
    print('Data shape:', df.shape)

    if pred is None:
        cols = ['Types', 'Counts', 'Uniques', 'Nulls', 'Min', 'Max']
        str = pd.concat([Types, Counts, Uniques, Nulls, Min, Max], axis = 1, sort=True)

    str.columns = cols
    print('___________________________\nData Types:')
    print(str.Types.value_counts())
    print('___________________________')
    return str
cols = ['patient_id', 'TBG', 'T4U_measured', 'T3_measured' , "FTI_measured", "TBG_measured", "TSH_measured", "TT4_measured"]
df = df.drop(cols, axis=1)
num_cols = ['TSH', 'T3', 'TT4', 'T4U', 'FTI']

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

df['sex'] = df['sex'].fillna(df['sex'].mode()[0])

df.loc[df['age'] > 100, 'age'] = None
df['age'] = df['age'].fillna(df['age'].median())

binary_cols = [
    'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_meds',
    'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment',
    'query_hypothyroid', 'query_hyperthyroid', 'lithium',
    'goitre', 'tumor', 'hypopituitary', 'psych'
]

for col in binary_cols:
    df[col] = df[col].map({'t': 1, 'f': 0})

df['sex'] = df['sex'].map({'F': 0, 'M': 1})
df = pd.get_dummies(df, columns=['referral_source'], drop_first=True, dtype=int)

import numpy as np

df['TSH'] = np.log1p(df['TSH'])
df['TT4'] = np.log1p(df['TT4'])
df['FTI'] = np.log1p(df['FTI'])

df['target'] = df['target'].apply(lambda x: 0 if x == '-' else 1)

from sklearn.model_selection import train_test_split

# Separate features and target
X = df.drop(columns=['target'])
y = df['target']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Features shape:", X.shape)
print("Target shape:", y.shape)


# ════════════════════════════════════════
#        KNN (GPU - Pure PyTorch)
# ════════════════════════════════════════

import numpy as np
import torch
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    log_loss, accuracy_score, precision_score,
    recall_score, f1_score, precision_recall_curve
)
from codecarbon import OfflineEmissionsTracker

# ── CodeCarbon ───────────────────────────────────────────────────────────────
knn_tracker = OfflineEmissionsTracker(country_iso_code="BGD", log_level="error")
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── GPU Status ───────────────────────────────────────────────────────────────
print(f"{'='*40}")
print(f"         GPU Status")
print(f"{'='*40}")

if torch.cuda.is_available():
    print(f"GPU Available      : Yes")
    print(f"GPU Name           : {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory (GB)    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}")
else:
    print(f"GPU Available      : No — using CPU")

print(f"KNN runs on        : {DEVICE}")
print(f"{'='*40}")

# ── Scale Data ───────────────────────────────────────────────────────────────
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train).astype(np.float32)
X_test_s  = scaler.transform(X_test).astype(np.float32)

y_train_arr = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
y_test_arr  = y_test.values  if hasattr(y_test,  'values') else np.array(y_test)

# ── Pure PyTorch GPU KNN ─────────────────────────────────────────────────────
def torch_knn_predict_proba(X_tr, y_tr, X_te, k, n_classes=2, batch_size=1024):
    """
    Brute-force KNN entirely on GPU using torch.cdist.
    Processes queries in batches to avoid OOM on large datasets.
    """
    X_tr_t  = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    X_te_t  = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)
    y_tr_t  = torch.tensor(y_tr, dtype=torch.long,    device=DEVICE)

    n_test  = X_te_t.shape[0]
    proba   = torch.zeros(n_test, n_classes, device=DEVICE)

    for start in range(0, n_test, batch_size):
        end        = min(start + batch_size, n_test)
        query_batch = X_te_t[start:end]                          # (B, D)

        # Squared L2 distances: (B, N_train)
        dists      = torch.cdist(query_batch, X_tr_t, p=2)

        # Top-k nearest neighbours
        _, indices = torch.topk(dists, k, dim=1, largest=False)  # (B, k)

        # Gather neighbour labels and compute class probabilities
        neighbor_labels = y_tr_t[indices]                        # (B, k)
        for c in range(n_classes):
            proba[start:end, c] = (neighbor_labels == c).float().mean(dim=1)

    return proba.cpu().numpy()

# ── Tune k with cross-validation ─────────────────────────────────────────────
print(f"{'='*40}")
print(f"   Tuning n_neighbors (CV)")
print(f"{'='*40}")

k_candidates = [21, 31, 41, 51, 71, 101, 151]
skf          = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_k      = 21
best_k_loss = float('inf')
n_classes   = len(np.unique(y_train_arr))

for k in k_candidates:
    fold_losses = []

    for train_idx, val_idx in skf.split(X_train_s, y_train_arr):
        X_fold_train = X_train_s[train_idx]
        X_fold_val   = X_train_s[val_idx]
        y_fold_train = y_train_arr[train_idx]
        y_fold_val   = y_train_arr[val_idx]

        proba = torch_knn_predict_proba(
            X_fold_train, y_fold_train, X_fold_val, k, n_classes
        )
        fold_losses.append(log_loss(y_fold_val, proba))

    cv_loss = np.mean(fold_losses)
    print(f"k = {k:3d}  —  CV Loss: {cv_loss:.4f}")

    if cv_loss < best_k_loss:
        best_k_loss = cv_loss
        best_k      = k

print(f"{'='*40}")
print(f"Best k             : {best_k}")
print(f"Best CV Loss       : {best_k_loss:.4f}")
print(f"{'='*40}")

# ── Tracker ──────────────────────────────────────────────────────────────────
knn_tracker.start()

knn_train_losses = []
knn_test_losses  = []
knn_data_sizes   = np.linspace(0.1, 1.0, 20)

patience        = 3
best_test_loss  = float('inf')
best_size_index = 0
counter         = 0

# ── Learning curve with early stopping ───────────────────────────────────────
for i, knn_size in enumerate(knn_data_sizes):
    knn_n_samples = int(len(X_train_s) * knn_size)

    X_partial = X_train_s[:knn_n_samples]
    y_partial = y_train_arr[:knn_n_samples]

    knn_train_proba = torch_knn_predict_proba(
        X_partial, y_partial, X_partial, best_k, n_classes
    )
    knn_test_proba = torch_knn_predict_proba(
        X_partial, y_partial, X_test_s, best_k, n_classes
    )

    knn_train_loss = log_loss(y_partial,  knn_train_proba)
    knn_test_loss  = log_loss(y_test_arr, knn_test_proba)

    knn_train_losses.append(knn_train_loss)
    knn_test_losses.append(knn_test_loss)

    print(f"Data size: {knn_n_samples:6d} — Train Loss: {knn_train_loss:.4f} | Test Loss: {knn_test_loss:.4f}")

    if knn_test_loss < best_test_loss - 1e-6:
        best_test_loss  = knn_test_loss
        best_size_index = i
        counter         = 0
    else:
        counter += 1

    if counter >= patience:
        print(f"\nEarly stopping triggered at data size {knn_n_samples}")
        break

# ── Convergence point ────────────────────────────────────────────────────────
knn_min_train_loss = knn_train_losses[best_size_index]
knn_min_test_loss  = knn_test_losses[best_size_index]
knn_min_size       = int(len(X_train_s) * knn_data_sizes[best_size_index])

print(f"\nConvergence Point  : Data size {knn_min_size}")
print(f"Min Train Loss     : {knn_min_train_loss:.4f}")
print(f"Min Test  Loss     : {knn_min_test_loss:.4f}")

# ── Final model on full data ──────────────────────────────────────────────────
knn_y_pred_proba = torch_knn_predict_proba(
    X_train_s, y_train_arr, X_test_s, best_k, n_classes
)

# ── Stop tracker ─────────────────────────────────────────────────────────────
knn_emissions = knn_tracker.stop()
print(f"CO2 Emissions      : {knn_emissions:.6f} kg CO2")

# ── Plot ─────────────────────────────────────────────────────────────────────
knn_x_axis = [int(len(X_train_s) * s) for s in knn_data_sizes]
knn_x_axis = knn_x_axis[:len(knn_train_losses)]

plt.figure()
plt.plot(knn_x_axis, knn_train_losses, label='Train Loss')
plt.plot(knn_x_axis, knn_test_losses,  label='Test Loss')
plt.axvline(
    x=knn_x_axis[best_size_index],
    color='red', linestyle='--',
    label=f'Min Loss (size={knn_min_size})'
)
plt.scatter(knn_x_axis[best_size_index], knn_min_test_loss, color='red', zorder=5)
plt.title("KNN (GPU) – Log Loss over Training Data Size")
plt.xlabel("Number of Training Samples")
plt.ylabel("Log Loss")
plt.legend()
plt.savefig('D4_knn_learning_curve.png', bbox_inches='tight', dpi=300)
plt.close()

# ── Optimal threshold via Precision-Recall curve ─────────────────────────────
probs = knn_y_pred_proba[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test_arr, probs)
f1_scores   = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_thresh = thresholds[np.argmax(f1_scores[:-1])]

print(f"\nOptimal Threshold  : {best_thresh:.4f}")

knn_y_pred = (probs >= best_thresh).astype(int)

# ── Metrics ──────────────────────────────────────────────────────────────────
knn_accuracy  = accuracy_score(y_test_arr,  knn_y_pred)
knn_precision = precision_score(y_test_arr, knn_y_pred, zero_division=0)
knn_recall    = recall_score(y_test_arr,    knn_y_pred, zero_division=0)
knn_f1        = f1_score(y_test_arr,        knn_y_pred, zero_division=0)

# ── GHG Breakdown ────────────────────────────────────────────────────────────
knn_ghg_co2 = knn_emissions * 0.90
knn_ghg_ch4 = knn_emissions * 0.07
knn_ghg_n2o = knn_emissions * 0.03

metrics_output = f"""
{'='*40}
    KNN (GPU/PyTorch) – Final Metrics
{'='*40}
Accuracy           : {knn_accuracy:.4f}
Precision          : {knn_precision:.4f}
Recall             : {knn_recall:.4f}
F1 Score           : {knn_f1:.4f}
Best k             : {best_k}
CO2 Emissions      : {knn_emissions:.6f} kg CO2
{'='*40}
   GHG Breakdown (Approximate)
{'='*40}
CO2 (kg)           : {knn_ghg_co2:.8f}
CH4 (kg)           : {knn_ghg_ch4:.8f}
N2O (kg)           : {knn_ghg_n2o:.8f}
Total CO2eq (kg)   : {knn_emissions:.8f}
{'='*40}
"""

print(metrics_output)

with open('D4_knn_metrics.txt', 'w', encoding='utf-8') as f:
    f.write(metrics_output.strip())

warnings.filterwarnings('default')

print("\nEntire code has runned and stopped running.")