import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from codecarbon import OfflineEmissionsTracker
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ── 1. Advanced Data Loading & Cleaning ──────────────────────────────────────
url = 'https://raw.githubusercontent.com/mazidzomader/CSE427-Project-BRACU/refs/heads/main/Datasets/Asthma.csv'
df = pd.read_csv(url)

# Preprocessing
df = df.drop(columns=['PatientID', 'DoctorInCharge'], errors='ignore')
X = df.drop(columns=['Diagnosis'])
y = df['Diagnosis']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
n_classes = len(np.unique(y_encoded))

# Split
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

# ── 2. Scaling & SMOTE ───────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train).astype(np.float32)
X_val_s   = scaler.transform(X_val).astype(np.float32)
X_test_s  = scaler.transform(X_test).astype(np.float32)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_s, y_train)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
print(f"KNN runs on        : {str(device).upper()} (Cleaned)")
print(f"{'='*40}")


# ── 4. Pure PyTorch GPU KNN ──────────────────────────────────────────────────
def torch_knn_predict_proba(X_tr, y_tr, X_te, k, n_classes, batch_size=512):
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    n_test = X_te_t.shape[0]
    proba = torch.zeros(n_test, n_classes, device=device)
    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        dists = torch.cdist(X_te_t[start:end], X_tr_t, p=2)
        _, indices = torch.topk(dists, k, dim=1, largest=False)
        neighbor_labels = y_tr_t[indices]
        for c in range(n_classes):
            proba[start:end, c] = (neighbor_labels == c).float().mean(dim=1)
    return proba.cpu().numpy()

# ── 5. Training ──────────────────────────────────────────────────────────────
tracker = OfflineEmissionsTracker(country_iso_code="BGD", log_level="error")
tracker.start()

best_k = 7
knn_sizes = np.linspace(0.1, 1.0, 10)
te_losses = []

for s in knn_sizes:
    n = int(len(X_train_res) * s)
    X_p, y_p = X_train_res[:n], y_train_res[:n]
    p = torch_knn_predict_proba(X_p, y_p, X_test_s, best_k, n_classes)
    te_losses.append(log_loss(y_test, p))

emissions = tracker.stop()

# ── 6. Metrics & Output ──────────────────────────────────────────────────────
probs = torch_knn_predict_proba(X_train_s, y_train, X_test_s, best_k, n_classes)
y_pred = np.argmax(probs, axis=1)
acc, pr, rc, f1 = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='weighted'), recall_score(y_test, y_pred, average='weighted'), f1_score(y_test, y_pred, average='weighted')

metrics_output = f"""
{'='*40}
    KNN (Cleaned D6) – Final Metrics
{'='*40}
Accuracy           : {acc:.4f}
Precision (wt)     : {pr:.4f}
Recall (wt)        : {rc:.4f}
F1 Score (wt)      : {f1:.4f}
Best k             : {best_k}
CO2 Emissions      : {emissions:.6f} kg CO2
{'='*40}
"""
print(metrics_output)
with open('D6_knn_metrics.txt', 'w', encoding='utf-8') as f: f.write(metrics_output.strip())

plt.figure(); plt.plot(knn_sizes * len(X_train_res), te_losses, label='Test Loss')
plt.title("KNN Learning Curve (Cleaned D6)"); plt.savefig('D6_knn_learning_curve.png'); plt.close()

print("\nEntire code has runned and stopped running.")
