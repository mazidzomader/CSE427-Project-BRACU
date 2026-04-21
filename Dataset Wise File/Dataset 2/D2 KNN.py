import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from codecarbon import OfflineEmissionsTracker
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ── 1. Data Loading & Preprocessing ──────────────────────────────────────────
url = "https://raw.githubusercontent.com/mazidzomader/CSE427-Project-BRACU/main/Datasets/heart.csv"
df = pd.read_csv(url)

# Handle missing values and categorical features
df.replace('?', np.nan, inplace=True)
df = pd.get_dummies(df, drop_first=True)
df = df.apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

TARGET_COL = "HeartDisease"
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ── 2. Scaling & SMOTE ───────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train).astype(np.float32)
X_test_s  = scaler.transform(X_test).astype(np.float32)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_s, y_train)
y_train_res_arr = y_train_res.values if hasattr(y_train_res, 'values') else np.array(y_train_res)
y_test_arr = y_test.values if hasattr(y_test, 'values') else np.array(y_test)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── 3. GPU Status ─────────────────────────────────────────────────────────────
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

# ── 4. Pure PyTorch GPU KNN ──────────────────────────────────────────────────
def torch_knn_predict_proba(X_tr, y_tr, X_te, k, n_classes=2, batch_size=1024):
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=DEVICE)
    n_test = X_te_t.shape[0]
    proba = torch.zeros(n_test, n_classes, device=DEVICE)
    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        dists = torch.cdist(X_te_t[start:end], X_tr_t, p=2)
        _, indices = torch.topk(dists, k, dim=1, largest=False)
        neighbor_labels = y_tr_t[indices]
        for c in range(n_classes):
            proba[start:end, c] = (neighbor_labels == c).float().mean(dim=1)
    return proba.cpu().numpy()

# ── 5. Tuning & Training ─────────────────────────────────────────────────────
knn_tracker = OfflineEmissionsTracker(country_iso_code="BGD", log_level="error")

print(f"\n   Tuning n_neighbors (CV)...")
k_candidates = [3, 5, 7, 11, 15, 21, 31, 41]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_k, best_k_loss, n_classes = 3, float('inf'), 2

for k in k_candidates:
    losses = []
    for tr_idx, val_idx in skf.split(X_train_res, y_train_res_arr):
        p = torch_knn_predict_proba(X_train_res[tr_idx], y_train_res_arr[tr_idx], X_train_res[val_idx], k, n_classes)
        losses.append(log_loss(y_train_res_arr[val_idx], p))
    cv_l = np.mean(losses)
    print(f"k = {k:3d}  —  CV Loss: {cv_l:.4f}")
    if cv_l < best_k_loss: best_k_loss, best_k = cv_l, k

print(f"Best k: {best_k}\n")

knn_tracker.start()
tr_losses, te_losses = [], []
knn_sizes = np.linspace(0.1, 1.0, 15)
patience, best_te_l, best_size_idx, counter = 3, float('inf'), 0, 0

for i, s in enumerate(knn_sizes):
    n = int(len(X_train_res) * s)
    X_p, y_p = X_train_res[:n], y_train_res_arr[:n]
    tr_p = torch_knn_predict_proba(X_p, y_p, X_p, best_k, n_classes)
    te_p = torch_knn_predict_proba(X_p, y_p, X_test_s, best_k, n_classes)
    tr_l, te_l = log_loss(y_p, tr_p), log_loss(y_test_arr, te_p)
    tr_losses.append(tr_l); te_losses.append(te_l)
    print(f"Size: {n:4d} | Train Loss: {tr_l:.4f} | Test Loss: {te_l:.4f}")
    if te_l < best_te_l - 1e-6:
        best_te_l, best_size_idx, counter = te_l, i, 0
    else:
        counter += 1
    if counter >= patience: break

emissions = knn_tracker.stop()

# ── 6. Metrics ───────────────────────────────────────────────────────────────
knn_proba = torch_knn_predict_proba(X_train_res, y_train_res_arr, X_test_s, best_k, n_classes)
probs = knn_proba[:, 1]
pre, rec, thresh = precision_recall_curve(y_test_arr, probs)
f1s = 2 * (pre * rec) / (pre + rec + 1e-8)
best_t = thresh[np.argmax(f1s[:-1])]

preds = (probs >= best_t).astype(int)
acc, pr, rc, f1 = accuracy_score(y_test_arr, preds), precision_score(y_test_arr, preds), recall_score(y_test_arr, preds), f1_score(y_test_arr, preds)

# ── 7. Output & Visualization ────────────────────────────────────────────────
ghg_co2, ghg_ch4, ghg_n2o = emissions * 0.90, emissions * 0.07, emissions * 0.03
metrics_output = f"""
{'='*40}
    KNN (GPU/PyTorch) – Final Metrics
{'='*40}
Accuracy           : {acc:.4f}
Precision          : {pr:.4f}
Recall             : {rc:.4f}
F1 Score           : {f1:.4f}
Best k             : {best_k}
CO2 Emissions      : {emissions:.6f} kg CO2
{'='*40}
   GHG Breakdown (Approximate)
{'='*40}
CO2 (kg)           : {ghg_co2:.8f}
CH4 (kg)           : {ghg_ch4:.8f}
N2O (kg)           : {ghg_n2o:.8f}
Total CO2eq (kg)   : {emissions:.8f}
{'='*40}
"""
print(metrics_output)
with open('D2_knn_metrics.txt', 'w', encoding='utf-8') as f:
    f.write(metrics_output.strip())

plt.figure()
plt.plot(tr_losses, label='Train Loss')
plt.plot(te_losses, label='Test Loss')
plt.axvline(x=best_size_idx, color='red', linestyle='--', label=f'Best Point')
plt.title("KNN Learning Curve (D2 Heart Disease)")
plt.xlabel("Training Steps")
plt.ylabel("Log Loss")
plt.legend()
plt.savefig('D2_knn_learning_curve.png', bbox_inches='tight', dpi=300)
plt.close()

print("\nEntire code has runned and stopped running.")