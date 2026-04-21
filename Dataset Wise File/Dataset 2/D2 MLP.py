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

# Split
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
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
print(f"MLP runs on        : {str(device).upper()}")
print(f"{'='*40}")

# Tensors
X_train_t = torch.tensor(X_train_res, dtype=torch.float32).to(device)
X_val_t   = torch.tensor(X_val_s, dtype=torch.float32).to(device)
X_test_t  = torch.tensor(X_test_s, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train_res.values, dtype=torch.float32).to(device)
y_val_t   = torch.tensor(y_val.values, dtype=torch.float32).to(device)
y_test_t  = torch.tensor(y_test.values, dtype=torch.float32).to(device)

# ── 4. Model ─────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

model = MLP(X_train_t.shape[1]).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# ── 5. Training ──────────────────────────────────────────────────────────────
tracker = OfflineEmissionsTracker(country_iso_code="BGD", log_level="error")
tracker.start()

train_losses, val_losses = [], []
best_loss, best_epoch, patience, counter, best_state = float('inf'), 0, 50, 0, None

for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train_t).squeeze()
    loss = criterion(logits, y_train_t)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        v_logits = model(X_val_t).squeeze()
        tr_p = torch.sigmoid(logits).cpu().numpy()
        vl_p = torch.sigmoid(v_logits).cpu().numpy()
        v_l = log_loss(y_val.values, vl_p)
    
    train_losses.append(log_loss(y_train_res, tr_p))
    val_losses.append(v_l)

    if v_l < best_loss - 1e-6:
        best_loss, best_epoch, counter, best_state = v_l, epoch, 0, model.state_dict()
    else:
        counter += 1
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1:4d} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {v_l:.4f}")
    
    if counter >= patience:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

if best_state: model.load_state_dict(best_state)
emissions = tracker.stop()

# ── 6. Metrics & Output ──────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(X_test_t)).squeeze().cpu().numpy()

y_pred = (probs >= 0.5).astype(int)
acc, pr, rc, f1 = accuracy_score(y_test.values, y_pred), precision_score(y_test.values, y_pred), recall_score(y_test.values, y_pred), f1_score(y_test.values, y_pred)

ghg_co2, ghg_ch4, ghg_n2o = emissions * 0.90, emissions * 0.07, emissions * 0.03
metrics_output = f"""
{'='*40}
    MLP (GPU/PyTorch) – Final Metrics
{'='*40}
Accuracy           : {acc:.4f}
Precision          : {pr:.4f}
Recall             : {rc:.4f}
F1 Score           : {f1:.4f}
Best Epoch         : {best_epoch + 1}
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
with open('D2_mlp_metrics.txt', 'w', encoding='utf-8') as f:
    f.write(metrics_output.strip())

plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.axvline(best_epoch, color="red", linestyle="--", label=f"Best (Epoch {best_epoch+1})")
plt.title("MLP Learning Curve (D2 Heart Disease)")
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.legend()
plt.savefig('D2_mlp_learning_curve.png', bbox_inches='tight', dpi=300)
plt.close()

print("\nEntire code has runned and stopped running.")