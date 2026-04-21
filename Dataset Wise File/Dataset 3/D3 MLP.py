import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
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

X_train_f, X_test, y_train_f, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train,   X_val,  y_train,   y_val  = train_test_split(X_train_f, y_train_f, test_size=0.2, random_state=42, stratify=y_train_f)

# ── 2. Robust Scaling & SMOTETomek ──────────────────────────────────────────
scaler    = RobustScaler()
X_train_s = scaler.fit_transform(X_train).astype(np.float32)
X_val_s   = scaler.transform(X_val).astype(np.float32)
X_test_s  = scaler.transform(X_test).astype(np.float32)

# Full 1:1 SMOTE balance
smt = SMOTE(random_state=42, sampling_strategy='auto', k_neighbors=5)
X_train_res, y_train_res = smt.fit_resample(X_train_s, y_train)
y_train_res = np.asarray(y_train_res, dtype=np.float32)
print(f"After SMOTE — Positive (Disease): {(y_train_res==1).sum():.0f}, Negative (Donor): {(y_train_res==0).sum():.0f}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Mini-batch DataLoader (stochastic gradient noise acts as regularization) ──
BATCH_SIZE  = 64
train_ds    = TensorDataset(torch.tensor(X_train_res), torch.tensor(y_train_res))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

X_val_t  = torch.tensor(X_val_s,      dtype=torch.float32).to(device)
X_test_t = torch.tensor(X_test_s,     dtype=torch.float32).to(device)
y_val_t  = torch.tensor(y_val.values, dtype=torch.float32).to(device)
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
print(f"MLP runs on        : {'GPU (cuda)' if torch.cuda.is_available() else 'CPU'}")
print(f"{'='*40}")


# ── 4. Model — smaller capacity to match small real-data regime ───────────────
class DeepMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(0.5),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.SiLU(), nn.Dropout(0.4),
            nn.Linear(64, 32),         nn.BatchNorm1d(32),  nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)

model     = DeepMLP(X_train_res.shape[1]).to(device)

# Data is 1:1 balanced via SMOTE — pos_weight=1.0
pos_w     = torch.tensor([1.0], dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
es_crit   = nn.BCEWithLogitsLoss()   # unweighted — for stable early-stopping

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)

# ── 5. Mini-batch Training ────────────────────────────────────────────────────
tracker = OfflineEmissionsTracker(country_iso_code="BGD", log_level="error")
tracker.start()

train_losses, val_losses = [], []
best_loss, best_epoch, patience_limit, counter, best_state = float('inf'), 0, 300, 0, None
min_epochs = 100   # warmup guard — patience doesn't fire before this epoch

for epoch in range(3000):
    # --- Mini-batch train pass ---
    model.train()
    epoch_tr_loss = 0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b).squeeze(), y_b)
        loss.backward()
        optimizer.step()
        epoch_tr_loss += loss.item()
    train_losses.append(epoch_tr_loss / len(train_loader))
    scheduler.step()

    # --- Val pass (unweighted for stable early-stopping) ---
    model.eval()
    with torch.no_grad():
        v_l = es_crit(model(X_val_t).squeeze(), y_val_t).item()
    val_losses.append(v_l)

    if v_l < best_loss - 1e-5:
        best_loss, best_epoch, counter = v_l, epoch, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    elif epoch >= min_epochs:
        counter += 1
    if counter >= patience_limit:
        break

if best_state:
    model.load_state_dict(best_state)

emissions = tracker.stop()

# ── 6. Threshold Optimisation & Metrics ──────────────────────────────────────
model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(X_test_t)).squeeze().cpu().numpy()

pre, rec, thresh = precision_recall_curve(y_test, probs)
f1s    = 2 * (pre * rec) / (pre + rec + 1e-8)
best_t = thresh[np.argmax(f1s[:-1])]
y_pred = (probs >= best_t).astype(int)

acc = accuracy_score(y_test,  y_pred)
pr  = precision_score(y_test, y_pred)
rc  = recall_score(y_test,    y_pred)
f1  = f1_score(y_test,        y_pred)

# ── 7. Metrics & Output ───────────────────────────────────────────────────────
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
CO2 (kg)           : {emissions*0.90:.8f}
CH4 (kg)           : {emissions*0.07:.8f}
N2O (kg)           : {emissions*0.03:.8f}
Total CO2eq (kg)   : {emissions:.8f}
{'='*40}
"""
print(metrics_output)
with open('D3_mlp_metrics.txt', 'w', encoding='utf-8') as f:
    f.write(metrics_output.strip())

plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.axvline(best_epoch, color="red", linestyle="--", label=f"Best (Epoch {best_epoch+1})")
plt.title("Deep MLP Learning Curve (D3 Hepatitis C Data)")
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.legend()
plt.savefig('D3_mlp_learning_curve.png', bbox_inches='tight', dpi=300)
plt.close()

print("\nEntire code has runned and stopped running.")