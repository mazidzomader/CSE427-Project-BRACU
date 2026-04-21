import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from torch.utils.data import DataLoader, TensorDataset
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
print(f"Logistic Regression runs on: {device}")
print(f"{'='*40}")

X_train_t = torch.tensor(X_train_res, dtype=torch.float32)
X_test_t  = torch.tensor(X_test_s, dtype=torch.float32)
y_train_t = torch.tensor(y_train_res.values, dtype=torch.float32)
y_test_t  = torch.tensor(y_test.values, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)

# ── 4. Model ─────────────────────────────────────────────────────────────────
class LogisticRegression(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 1)
    def forward(self, x):
        return self.linear(x)

model = LogisticRegression(X_train_t.shape[1]).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ── 5. Training ──────────────────────────────────────────────────────────────
tracker = OfflineEmissionsTracker(country_iso_code="BGD", log_level="error")
tracker.start()

train_losses, test_losses = [], []
best_loss, best_epoch, patience, counter, best_state = float('inf'), 0, 50, 0, None

for epoch in range(1000):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb).squeeze(), yb)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        tr_probs = torch.sigmoid(model(X_train_t.to(device)).squeeze()).cpu().numpy()
        te_probs = torch.sigmoid(model(X_test_t.to(device)).squeeze()).cpu().numpy()
    
    tr_l, te_l = log_loss(y_train_res, tr_probs), log_loss(y_test, te_probs)
    train_losses.append(tr_l)
    test_losses.append(te_l)

    if te_l < best_loss - 1e-5:
        best_loss, best_epoch, counter, best_state = te_l, epoch, 0, model.state_dict()
    else:
        counter += 1
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1:4d} | Train Loss: {tr_l:.4f} | Test Loss: {te_l:.4f}")
    
    if counter >= patience:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

if best_state: model.load_state_dict(best_state)
lr_emissions = tracker.stop()

# ── 6. Metrics ───────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(X_test_t.to(device))).squeeze().cpu().numpy()

precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_thresh = thresholds[np.argmax(f1_scores[:-1])]

preds = (probs >= best_thresh).astype(int)
lr_accuracy = accuracy_score(y_test, preds)
lr_precision = precision_score(y_test, preds, zero_division=0)
lr_recall = recall_score(y_test, preds, zero_division=0)
lr_f1 = f1_score(y_test, preds, zero_division=0)

# ── 7. Results & Output ──────────────────────────────────────────────────────
lr_co2, lr_ch4, lr_n2o = lr_emissions * 0.90, lr_emissions * 0.07, lr_emissions * 0.03
metrics_output = f"""
{'='*40}
    Logistic Regression – Final Metrics
{'='*40}
Accuracy           : {lr_accuracy:.4f}
Precision          : {lr_precision:.4f}
Recall             : {lr_recall:.4f}
F1 Score           : {lr_f1:.4f}
Convergence Epoch  : {best_epoch + 1}
CO2 Emissions      : {lr_emissions:.6f} kg CO2
{'='*40}
   GHG Breakdown (Approximate)
{'='*40}
CO2 (kg)           : {lr_co2:.8f}
CH4 (kg)           : {lr_ch4:.8f}
N2O (kg)           : {lr_n2o:.8f}
Total CO2eq (kg)   : {lr_emissions:.8f}
{'='*40}
"""
print(metrics_output)
with open('D2_logreg_metrics.txt', 'w', encoding='utf-8') as f:
    f.write(metrics_output.strip())

plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.axvline(best_epoch, color="red", linestyle="--", label=f"Best (Epoch {best_epoch+1})")
plt.title("Logistic Regression Learning Curve (D2 Heart Disease)")
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.legend()
plt.savefig('D2_logreg_learning_curve.png', bbox_inches='tight', dpi=300)
plt.close()

print("\nEntire code has runned and stopped running.")