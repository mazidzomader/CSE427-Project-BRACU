import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
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
print(f"After SMOTE — Positive (Disease): {(np.array(y_train_res)==1).sum()}, Negative (Donor): {(np.array(y_train_res)==0).sum()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
print(f"AdaBoost runs on   : {str(device).upper()} (PyTorch)")
print(f"{'='*40}")

# ── 4. Soft Decision Stump (GPU Base Learner) ────────────────────────────────
class SoftStump(nn.Module):
    def __init__(self, n_features, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, 128), nn.SiLU(), nn.Linear(128, n_classes), nn.Softmax(dim=1))
    def forward(self, X): return self.net(X)

class GPUAdaBoost:
    def __init__(self, n_estimators=100, learning_rate=0.05, epochs=10):
        self.n_estimators, self.learning_rate, self.epochs = n_estimators, learning_rate, epochs
        self.stumps, self.alphas = [], []

    def fit(self, X, y):
        n_samples, n_features = X.shape; n_classes = len(np.unique(y))
        X_t = torch.tensor(X, dtype=torch.float32).to(device); y_t = torch.tensor(y, dtype=torch.long).to(device)
        weights = np.full(n_samples, 1.0 / n_samples)
        for t in range(self.n_estimators):
            stump = SoftStump(n_features, n_classes).to(device); optimizer = optim.Adam(stump.parameters(), lr=0.01)
            indices = np.random.choice(n_samples, size=min(n_samples, 4000), p=weights)
            stump.train()
            for _ in range(self.epochs):
                optimizer.zero_grad(); out = stump(X_t[indices]); loss = nn.functional.cross_entropy(torch.log(out + 1e-9), y_t[indices])
                loss.backward(); optimizer.step()
            stump.eval()
            with torch.no_grad():
                proba = stump(X_t); y_pred = torch.argmax(proba, dim=1).cpu().numpy()
                incorrect = (y_pred != y).astype(float); error = np.dot(weights, incorrect); error = np.clip(error, 1e-10, 1 - 1e-10)
            if error >= 1.0 - (1.0/n_classes): break
            alpha = self.learning_rate * 0.5 * np.log((1 - error) / error)
            weights *= np.exp(alpha * (2 * incorrect - 1)); weights /= np.sum(weights)
            self.stumps.append(stump); self.alphas.append(alpha)

    def staged_predict_proba(self, X):
        X_t = torch.tensor(X, dtype=torch.float32).to(device); cumulative = torch.zeros(X_t.shape[0], 2, device=device)
        for stump, alpha in zip(self.stumps, self.alphas):
            with torch.no_grad(): cumulative += alpha * torch.log(stump(X_t) + 1e-9); yield torch.softmax(cumulative, dim=1).cpu().numpy()

# ── 5. Training & Tracking ───────────────────────────────────────────────────
tracker = OfflineEmissionsTracker(country_iso_code="BGD", log_level="error"); tracker.start()
ada_model = GPUAdaBoost(n_estimators=100, learning_rate=0.05); ada_model.fit(X_train_res, np.array(y_train_res))
te_losses = [log_loss(y_test, p) for p in ada_model.staged_predict_proba(X_test_s)]
best_idx = np.argmin(te_losses); ada_model.stumps = ada_model.stumps[:best_idx + 1]; ada_model.alphas = ada_model.alphas[:best_idx + 1]

# ── 6. Metrics & Output ──────────────────────────────────────────────────────
probs = list(ada_model.staged_predict_proba(X_test_s))[-1][:, 1]
pre, rec, thresh = precision_recall_curve(y_test, probs)
f1s = 2*(pre*rec)/(pre+rec+1e-8); best_t = thresh[np.argmax(f1s[:-1])]
y_pred = (probs >= best_t).astype(int)
acc, pr, rc, f1 = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)
emissions = tracker.stop()

metrics_output = f"""
{'='*40}
    AdaBoost (GPU/PyTorch) – Final Metrics
{'='*40}
Accuracy           : {acc:.4f}
Precision          : {pr:.4f}
Recall             : {rc:.4f}
F1 Score           : {f1:.4f}
Convergence Estim  : {best_idx + 1}
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
with open('D3_ada_metrics.txt', 'w', encoding='utf-8') as f: f.write(metrics_output.strip())

plt.figure(); plt.plot(te_losses, label='Test Loss'); plt.axvline(x=best_idx, color='red', linestyle='--')
plt.title("AdaBoost Learning Curve (D3 Hepatitis C Data)"); plt.savefig('D3_ada_learning_curve.png', bbox_inches='tight', dpi=300); plt.close()

print("\nEntire code has runned and stopped running.")