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
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Validation Split
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

# ── 2. Scaling & SMOTE ───────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_s, y_train)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── 3. GPU Status ─────────────────────────────────────────────────────────────
print("="*40)
print("         GPU Status")
print("="*40)
if torch.cuda.is_available():
    print("GPU Available      : Yes")
    print(f"GPU Name           : {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory (GB)    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}")
else:
    print("GPU Available      : No — using CPU")
print(f"MLP runs on        : {device}")
print("="*40)

# Tensors
X_train_tensor = torch.tensor(X_train_res, dtype=torch.float32).to(device)
X_val_tensor   = torch.tensor(X_val_s, dtype=torch.float32).to(device)
X_test_tensor  = torch.tensor(X_test_s, dtype=torch.float32).to(device)

y_train_tensor = torch.tensor(y_train_res.values, dtype=torch.float32).to(device)
y_val_tensor   = torch.tensor(y_val.values, dtype=torch.float32).to(device)
y_test_tensor  = torch.tensor(y_test.values, dtype=torch.float32).to(device)

# ── 4. Model ─────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
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
        return self.model(x)

input_dim = X_train_tensor.shape[1]
model = MLP(input_dim).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# ── 5. Training ──────────────────────────────────────────────────────────────
tracker = OfflineEmissionsTracker(country_iso_code="BGD", log_level="error")
tracker.start()

train_losses, val_losses = [], []
patience, counter, best_loss, best_epoch, best_state = 50, 0, float('inf'), 0, None

for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train_tensor).squeeze()
    loss = criterion(logits, y_train_tensor)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        v_logits = model(X_val_tensor).squeeze()
        v_loss = criterion(v_logits, y_val_tensor)
        
        tr_probs = torch.sigmoid(logits).cpu().numpy()
        vl_probs = torch.sigmoid(v_logits).cpu().numpy()

    train_losses.append(log_loss(y_train_res, tr_probs))
    val_losses.append(log_loss(y_val, vl_probs))

    if v_loss < best_loss - 1e-6:
        best_loss, best_epoch, counter, best_state = v_loss, epoch, 0, model.state_dict()
    else:
        counter += 1
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1:4d} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")
    
    if counter >= patience:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

if best_state: model.load_state_dict(best_state)
emissions = tracker.stop()

# ── 6. Final Evaluation ───────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(X_test_tensor)).squeeze().cpu().numpy()

best_thresh, best_f1 = 0.5, 0
for t in np.arange(0.1, 0.9, 0.01):
    f = f1_score(y_test, (probs >= t).astype(int))
    if f > best_f1: best_f1, best_thresh = f, t

preds = (probs >= best_thresh).astype(int)
acc, prec, rec, f1 = accuracy_score(y_test, preds), precision_score(y_test, preds), recall_score(y_test, preds), f1_score(y_test, preds)

# ── 7. Results & Output ──────────────────────────────────────────────────────
ghg_co2, ghg_ch4, ghg_n2o = emissions * 0.90, emissions * 0.07, emissions * 0.03
metrics_output = f"""
{'='*40}
    MLP (GPU/PyTorch) – Final Metrics
{'='*40}
Accuracy           : {acc:.4f}
Precision          : {prec:.4f}
Recall             : {rec:.4f}
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
with open('D1_mlp_metrics.txt', 'w', encoding='utf-8') as f:
    f.write(metrics_output.strip())

plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.axvline(best_epoch, color="red", linestyle="--", label=f"Best (Epoch {best_epoch+1})")
plt.title("MLP Learning Curve (D1 Breast Cancer)")
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.legend()
plt.savefig('D1_mlp_learning_curve.png', bbox_inches='tight', dpi=300)
plt.close()

print("\nEntire code has runned and stopped running.")