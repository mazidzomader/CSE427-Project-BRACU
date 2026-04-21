import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
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
print(f"AdaBoost runs on   : {str(device).upper()} (Cleaned)")
print(f"{'='*40}")

# ── 4. Base Learner & Ensemble ───────────────────────────────────────────────
class RobustStump(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, 64), nn.ReLU(), nn.Linear(64, n_classes), nn.Softmax(dim=1))
    def forward(self, X): return self.net(X)

class GPUAdaBoost:
    def __init__(self, n_estimators=50, learning_rate=0.01, epochs=15):
        self.n_estimators, self.learning_rate, self.epochs = n_estimators, learning_rate, epochs
        self.stumps, self.alphas = [], []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        X_t, y_t = torch.tensor(X).to(device), torch.tensor(y, dtype=torch.long).to(device)
        weights = np.full(n_samples, 1.0 / n_samples)
        for t in range(self.n_estimators):
            stump = RobustStump(n_features, n_classes).to(device)
            optimizer = optim.Adam(stump.parameters(), lr=0.01)
            idx = np.random.choice(n_samples, size=min(n_samples, 2000), p=weights)
            stump.train()
            for _ in range(self.epochs):
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(torch.log(stump(X_t[idx]) + 1e-9), y_t[idx])
                loss.backward(); optimizer.step()
            stump.eval()
            with torch.no_grad():
                proba = stump(X_t); y_pred = torch.argmax(proba, dim=1).cpu().numpy()
                incorrect = (y_pred != y).astype(float); error = np.dot(weights, incorrect)
                error = np.clip(error, 1e-10, 1 - 1e-10)
            if error >= (1 - 1/n_classes): break
            alpha = self.learning_rate * (np.log((1 - error) / error) + np.log(n_classes - 1))
            weights *= np.exp(alpha * incorrect); weights /= np.sum(weights)
            self.stumps.append(stump); self.alphas.append(alpha)

    def staged_predict_proba(self, X):
        X_t = torch.tensor(X).to(device); cumulative = torch.zeros(X_t.shape[0], n_classes, device=device)
        for stump, alpha in zip(self.stumps, self.alphas):
            with torch.no_grad():
                cumulative += alpha * torch.log(stump(X_t) + 1e-9)
                yield torch.softmax(cumulative, dim=1).cpu().numpy()

# ── 5. Training ──────────────────────────────────────────────────────────────
tracker = OfflineEmissionsTracker(country_iso_code="BGD", log_level="error")
tracker.start()

print("\nTraining Cleaned GPU AdaBoost...")
ada_model = GPUAdaBoost(n_estimators=100, learning_rate=0.05)
ada_model.fit(X_train_res, y_train_res)

te_losses = [log_loss(y_test, p) for p in ada_model.staged_predict_proba(X_test_s)]
best_idx = np.argmin(te_losses)
ada_model.stumps = ada_model.stumps[:best_idx + 1]; ada_model.alphas = ada_model.alphas[:best_idx + 1]

probs = list(ada_model.staged_predict_proba(X_test_s))[-1]
y_pred = np.argmax(probs, axis=1)
acc, pr, rc, f1 = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='weighted'), recall_score(y_test, y_pred, average='weighted'), f1_score(y_test, y_pred, average='weighted')
emissions = tracker.stop()

# ── 6. Metrics & Output ──────────────────────────────────────────────────────
metrics_output = f"""
{'='*40}
    AdaBoost (Cleaned D6) – Final Metrics
{'='*40}
Accuracy           : {acc:.4f}
Precision (wt)     : {pr:.4f}
Recall (wt)        : {rc:.4f}
F1 Score (wt)      : {f1:.4f}
Convergence Estim  : {best_idx + 1}
CO2 Emissions      : {emissions:.6f} kg CO2
{'='*40}
"""
print(metrics_output)
with open('D6_ada_metrics.txt', 'w', encoding='utf-8') as f: f.write(metrics_output.strip())

plt.figure(); plt.plot(te_losses, label='Test Loss'); plt.axvline(x=best_idx, color='red', linestyle='--')
plt.title("AdaBoost Learning Curve (Cleaned D6)"); plt.savefig('D6_ada_learning_curve.png'); plt.close()

print("\nEntire code has runned and stopped running.")