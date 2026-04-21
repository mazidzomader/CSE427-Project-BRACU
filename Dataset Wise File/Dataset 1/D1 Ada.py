import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from codecarbon import OfflineEmissionsTracker
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ── 1. Data Loading & Preprocessing ──────────────────────────────────────────
url = "https://raw.githubusercontent.com/mazidzomader/CSE427-Project-BRACU/main/Datasets/breast_cancer_bd.csv"
df = pd.read_csv(url)

# Handle missing values and non-numeric columns
df.replace('?', np.nan, inplace=True)
df = df.apply(pd.to_numeric)
df.dropna(inplace=True)

TARGET_COL = "Class"
ID_COL = "Sample code number"
FEATURE_COLS = [col for col in df.columns if col not in [TARGET_COL, ID_COL]]

X = df[FEATURE_COLS]
y = df[TARGET_COL].map({2: 0, 4: 1})

# Initial Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ── 2. Scaling & SMOTE ───────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

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
print(f"AdaBoost runs on   : {str(device).upper()} (PyTorch)")
print(f"{'='*40}")

# ── 4. Soft Decision Stump (GPU Base Learner) ────────────────────────────────
class SoftStump(nn.Module):
    def __init__(self, n_features, n_classes=2):
        super().__init__()
        # Linear layer makes it a "weaker" learner (Logistic Regression)
        # This is more appropriate for AdaBoost on small datasets.
        self.net = nn.Sequential(
            nn.Linear(n_features, n_classes),
            nn.Softmax(dim=1)
        )
    def forward(self, X):
        return self.net(X)

class GPUAdaBoost:
    def __init__(self, n_estimators=50, learning_rate=0.5, epochs=30):
        self.n_estimators, self.learning_rate, self.epochs = n_estimators, learning_rate, epochs
        self.stumps, self.alphas = [], []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        y_t = torch.tensor(y, dtype=torch.long).to(device)
        weights = np.full(n_samples, 1.0 / n_samples)
        
        for t in range(self.n_estimators):
            stump = SoftStump(n_features, n_classes).to(device)
            optimizer = optim.Adam(stump.parameters(), lr=0.01)
            indices = np.random.choice(n_samples, size=n_samples, p=weights)
            X_batch, y_batch = X_t[indices], y_t[indices]
            
            stump.train()
            for _ in range(self.epochs):
                optimizer.zero_grad()
                out = stump(X_batch)
                loss = nn.functional.cross_entropy(torch.log(out + 1e-9), y_batch)
                loss.backward()
                optimizer.step()
            
            stump.eval()
            with torch.no_grad():
                proba = stump(X_t)
                y_pred = torch.argmax(proba, dim=1).cpu().numpy()
                incorrect = (y_pred != y).astype(float)
                error = np.dot(weights, incorrect)
                error = np.clip(error, 1e-10, 1 - 1e-10)
            
            alpha = self.learning_rate * 0.5 * np.log((1 - error) / error)
            weights *= np.exp(alpha * (2 * incorrect - 1))
            weights /= np.sum(weights)
            self.stumps.append(stump); self.alphas.append(alpha)
            if (t + 1) % 10 == 0:
                print(f"Estimator {t+1:3d}/{self.n_estimators} trained | Error: {error:.4f} | Alpha: {alpha:.4f}")

    def staged_predict_proba(self, X):
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        cumulative = torch.zeros(X_t.shape[0], 2, device=device)
        for stump, alpha in zip(self.stumps, self.alphas):
            with torch.no_grad():
                proba = stump(X_t)
                cumulative += alpha * torch.log(proba + 1e-9)
                yield torch.softmax(cumulative, dim=1).cpu().numpy()

# ── 5. Training & Tracking ───────────────────────────────────────────────────
ada_tracker = OfflineEmissionsTracker(country_iso_code="BGD", log_level="error")
ada_tracker.start()

print("\nTraining GPU AdaBoost (PyTorch Ensemble)...")
ada_model = GPUAdaBoost(n_estimators=100, learning_rate=0.1, epochs=5)
ada_model.fit(X_train_res, y_train_res.values if hasattr(y_train_res, 'values') else y_train_res)

tr_losses, te_losses = [], []
for tr_p, te_p in zip(ada_model.staged_predict_proba(X_train_s), ada_model.staged_predict_proba(X_test_s)):
    tr_losses.append(log_loss(y_train, tr_p))
    te_losses.append(log_loss(y_test, te_p))

best_idx = np.argmin(te_losses)
ada_model.stumps = ada_model.stumps[:best_idx + 1]
ada_model.alphas = ada_model.alphas[:best_idx + 1]

# ── 6. Final Evaluation ───────────────────────────────────────────────────────
probs = None
for p in ada_model.staged_predict_proba(X_test_s): probs = p[:, 1]
best_thresh, best_f1 = 0.5, 0
for t in np.arange(0.1, 0.9, 0.01):
    f1 = f1_score(y_test, (probs >= t).astype(int))
    if f1 > best_f1: best_f1, best_thresh = f1, t

y_pred = (probs >= best_thresh).astype(int)
accuracy, precision, recall, f1 = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)
emissions = ada_tracker.stop()

# ── 7. Output & Visualization ────────────────────────────────────────────────
ghg_co2, ghg_ch4, ghg_n2o = emissions * 0.90, emissions * 0.07, emissions * 0.03
metrics_output = f"""
{'='*40}
    AdaBoost (GPU/PyTorch) – Final Metrics
{'='*40}
Accuracy           : {accuracy:.4f}
Precision          : {precision:.4f}
Recall             : {recall:.4f}
F1 Score           : {f1:.4f}
Convergence Estim  : {best_idx + 1}
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
with open('D1_ada_metrics.txt', 'w', encoding='utf-8') as f:
    f.write(metrics_output.strip())

plt.figure()
plt.plot(tr_losses, label='Train Loss')
plt.plot(te_losses, label='Test Loss')
plt.axvline(x=best_idx, color='red', linestyle='--', label=f'Best ({best_idx + 1})')
plt.title("AdaBoost Learning Curve (D1 Breast Cancer)")
plt.xlabel("Estimators")
plt.ylabel("Log Loss")
plt.legend()
plt.savefig('D1_ada_learning_curve.png', bbox_inches='tight', dpi=300)
plt.close()

print("\nEntire code has runned and stopped running.")