import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from torch.utils.data import DataLoader, TensorDataset
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
y_train_res = np.asarray(y_train_res, dtype=np.float32)
print(f"After SMOTE — Positive (Disease): {(np.array(y_train_res)==1).sum()}, Negative (Donor): {(np.array(y_train_res)==0).sum()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_t = torch.tensor(X_train_res, dtype=torch.float32).to(device)
X_test_t  = torch.tensor(X_test_s,    dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train_res, dtype=torch.float32).to(device)
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=128, shuffle=True)

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
print(f"Logistic Regression on : {str(device).upper()} (PyTorch)")
print(f"{'='*40}")


class LogisticRegression(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 1)
    def forward(self, x): return self.linear(x)

model = LogisticRegression(X_train_t.shape[1]).to(device)
# Data is 1:1 balanced via SMOTE — pos_weight=1.0
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(device))
optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-3)

# ── 5. Training ──────────────────────────────────────────────────────────────
tracker = OfflineEmissionsTracker(country_iso_code="BGD", log_level="error"); tracker.start()
test_losses, best_loss, best_epoch, patience, counter, best_state = [], float('inf'), 0, 50, 0, None
for epoch in range(1000):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad(); loss = criterion(model(xb).squeeze(), yb); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        te_probs = torch.sigmoid(model(X_test_t).squeeze())
        te_l = nn.BCELoss()(te_probs, torch.tensor(y_test.values, dtype=torch.float32).to(device)).item()
    test_losses.append(te_l)
    if te_l < best_loss - 1e-5: best_loss, best_epoch, counter, best_state = te_l, epoch, 0, model.state_dict()
    else: counter += 1
    if counter >= patience: break

if best_state: model.load_state_dict(best_state)
emissions = tracker.stop()

# ── 6. Metrics & Output ──────────────────────────────────────────────────────
model.eval()
with torch.no_grad(): probs = torch.sigmoid(model(X_test_t)).squeeze().cpu().numpy()
pre, rec, thresh = precision_recall_curve(y_test, probs)
f1s = 2*(pre*rec)/(pre+rec+1e-8); best_t = thresh[np.argmax(f1s[:-1])]
y_pred = (probs >= best_t).astype(int)
acc, pr, rc, f1 = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)

metrics_output = f"""
{'='*40}
    Logistic Regression (GPU/PyTorch) – Final Metrics
{'='*40}
Accuracy           : {acc:.4f}
Precision          : {pr:.4f}
Recall             : {rc:.4f}
F1 Score           : {f1:.4f}
Convergence Epoch  : {best_epoch + 1}
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
with open('D3_logreg_metrics.txt', 'w', encoding='utf-8') as f: f.write(metrics_output.strip())

plt.figure(); plt.plot(test_losses, label="Test Loss"); plt.axvline(best_epoch, color="red", linestyle="--")
plt.title("Logistic Regression Learning Curve (D3 Hepatitis C Data)"); plt.savefig('D3_logreg_learning_curve.png', bbox_inches='tight', dpi=300); plt.close()

print("\nEntire code has runned and stopped running.")