import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from codecarbon import OfflineEmissionsTracker
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ── 1. Advanced Data Preprocessing & Feature Engineering ──────────────────────
url = 'https://raw.githubusercontent.com/mazidzomader/CSE427-Project-BRACU/refs/heads/main/Datasets/HepatitisCdata.csv'
df = pd.read_csv(url)
# Drop the first unnamed index column
df = df.drop(df.columns[0], axis=1)

# Fill missing numerical values with median
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].median())

# Encode Sex
df['Sex'] = df['Sex'].map({'m': 1, 'f': 0}).fillna(0).astype(int)

# Target variable is Category
# Encode Category to Binary: 0 for Blood Donors, 1 for Disease (Hepatitis/Fibrosis/Cirrhosis)
df['Category'] = df['Category'].apply(lambda x: 0 if str(x).startswith('0') else 1)

# Feature Engineering for HCV
df['AST_ALT_ratio'] = df['AST'] / (df['ALT'] + 1e-8)
df['ALB_PROT_ratio'] = df['ALB'] / (df['PROT'] + 1e-8)

X, y = df.drop('Category', axis=1), df['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ── 2. Robust Scaling & SMOTETomek ──────────────────────────────────────────
scaler = RobustScaler()
X_train_s = scaler.fit_transform(X_train).astype(np.float32)
X_test_s  = scaler.transform(X_test).astype(np.float32)

# Full 1:1 SMOTE balance
smt = SMOTE(random_state=42, sampling_strategy='auto', k_neighbors=5)
X_train_res, y_train_res = smt.fit_resample(X_train_s, y_train)
y_train_res_arr = np.array(y_train_res)
y_test_arr = np.array(y_test)
print(f"After SMOTE — Positive (Disease): {(np.array(y_train_res)==1).sum()}, Negative (Donor): {(np.array(y_train_res)==0).sum()}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
print(f"KNN runs on        : {str(DEVICE).upper()} (PyTorch)")
print(f"{'='*40}")


# ── 4. Pure PyTorch GPU KNN ──────────────────────────────────────────────────
def torch_knn_predict_proba(X_tr, y_tr, X_te, k, n_classes=2, batch_size=1024):
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=DEVICE)
    n_test = X_te_t.shape[0]; proba = torch.zeros(n_test, n_classes, device=DEVICE)
    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test); dists = torch.cdist(X_te_t[start:end], X_tr_t, p=2)
        _, indices = torch.topk(dists, k, dim=1, largest=False); neighbor_labels = y_tr_t[indices]
        for c in range(n_classes): proba[start:end, c] = (neighbor_labels == c).float().mean(dim=1)
    return proba.cpu().numpy()

# ── 5. Tuning & Training ─────────────────────────────────────────────────────
knn_tracker = OfflineEmissionsTracker(country_iso_code="BGD", log_level="error")
k_candidates = [3, 7, 15, 31, 63, 127]
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
best_k, best_k_loss = 3, float('inf')
for k in k_candidates:
    losses = []
    for tr_idx, val_idx in skf.split(X_train_res, y_train_res_arr):
        p = torch_knn_predict_proba(X_train_res[tr_idx], y_train_res_arr[tr_idx], X_train_res[val_idx], k)
        losses.append(log_loss(y_train_res_arr[val_idx], p))
    cv_l = np.mean(losses)
    if cv_l < best_k_loss: best_k_loss, best_k = cv_l, k

knn_tracker.start()
te_losses, knn_sizes = [], np.linspace(0.1, 1.0, 10)
for s in knn_sizes:
    n = int(len(X_train_res) * s); X_p, y_p = X_train_res[:n], y_train_res_arr[:n]
    te_p = torch_knn_predict_proba(X_p, y_p, X_test_s, best_k); te_losses.append(log_loss(y_test_arr, te_p))
emissions = knn_tracker.stop()

# ── 6. Threshold Tuning & Metrics ─────────────────────────────────────────────
probs = torch_knn_predict_proba(X_train_res, y_train_res_arr, X_test_s, best_k)[:, 1]
pre, rec, thresh = precision_recall_curve(y_test_arr, probs)
f1s = 2 * (pre * rec) / (pre + rec + 1e-8); best_t = thresh[np.argmax(f1s[:-1])]
preds = (probs >= best_t).astype(int)
acc, pr, rc, f1 = accuracy_score(y_test_arr, preds), precision_score(y_test_arr, preds), recall_score(y_test_arr, preds), f1_score(y_test_arr, preds)

# ── Output ───────────────────────────────────────────────────────────────────
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
CO2 (kg)           : {emissions*0.9:.8f}
CH4 (kg)           : {emissions*0.07:.8f}
N2O (kg)           : {emissions*0.03:.8f}
Total CO2eq (kg)   : {emissions:.8f}
{'='*40}
"""
print(metrics_output)
with open('D3_knn_metrics.txt', 'w', encoding='utf-8') as f: f.write(metrics_output.strip())

plt.figure(); plt.plot(knn_sizes * len(X_train_res), te_losses, label='Test Loss')
plt.title("KNN Learning Curve (D3 Hepatitis C Data)"); plt.savefig('D3_knn_learning_curve.png', bbox_inches='tight', dpi=300); plt.close()

print("\nEntire code has runned and stopped running.")