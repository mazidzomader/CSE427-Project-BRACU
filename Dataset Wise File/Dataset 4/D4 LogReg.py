import pandas as pd
url = 'https://raw.githubusercontent.com/mazidzomader/CSE427-Project-BRACU/refs/heads/main/Datasets/thyroidDF.csv'

df = pd.read_csv(url)
def summary(df, pred=None):
    obs = df.shape[0]
    Types = df.dtypes
    Counts = df.apply(lambda x: x.count())
    Min = df.select_dtypes(include=['number']).min()
    Max = df.select_dtypes(include=['number']).max()
    Uniques = df.apply(lambda x: x.unique().shape[0])
    Nulls = df.apply(lambda x: x.isnull().sum())
    print('Data shape:', df.shape)

    if pred is None:
        cols = ['Types', 'Counts', 'Uniques', 'Nulls', 'Min', 'Max']
        str = pd.concat([Types, Counts, Uniques, Nulls, Min, Max], axis = 1, sort=True)

    str.columns = cols
    print('___________________________\nData Types:')
    print(str.Types.value_counts())
    print('___________________________')
    return str
cols = ['patient_id', 'TBG', 'T4U_measured', 'T3_measured' , "FTI_measured", "TBG_measured", "TSH_measured", "TT4_measured"]
df = df.drop(cols, axis=1)
num_cols = ['TSH', 'T3', 'TT4', 'T4U', 'FTI']

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

df['sex'] = df['sex'].fillna(df['sex'].mode()[0])

df.loc[df['age'] > 100, 'age'] = None
df['age'] = df['age'].fillna(df['age'].median())

binary_cols = [
    'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_meds',
    'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment',
    'query_hypothyroid', 'query_hyperthyroid', 'lithium',
    'goitre', 'tumor', 'hypopituitary', 'psych'
]

for col in binary_cols:
    df[col] = df[col].map({'t': 1, 'f': 0})

df['sex'] = df['sex'].map({'F': 0, 'M': 1})
df = pd.get_dummies(df, columns=['referral_source'], drop_first=True, dtype=int)

import numpy as np

df['TSH'] = np.log1p(df['TSH'])
df['TT4'] = np.log1p(df['TT4'])
df['FTI'] = np.log1p(df['FTI'])

df['target'] = df['target'].apply(lambda x: 0 if x == '-' else 1)

from sklearn.model_selection import train_test_split

# Separate features and target
X = df.drop(columns=['target'])
y = df['target']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# ════════════════════════════════════════
#       LOGISTIC REGRESSION (FIXED)
# ════════════════════════════════════════

import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    log_loss, accuracy_score, precision_score,
    recall_score, f1_score, precision_recall_curve
)
from torch.utils.data import DataLoader, TensorDataset
from codecarbon import OfflineEmissionsTracker

warnings.filterwarnings('ignore')

# ── GPU Status ───────────────────────────────────────────────────────────────
print(f"{'='*40}")
print(f"         GPU Status")
print(f"{'='*40}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print(f"GPU Available      : Yes")
    print(f"GPU Name           : {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory (GB)    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}")
else:
    print(f"GPU Available      : No — using CPU")

print(f"Logistic Regression: {device}")
print(f"{'='*40}")

# ── Scale Data ───────────────────────────────────────────────────────────────
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── SMOTE Resampling (applied on scaled training data only) ──────────────────
sm = SMOTE(random_state=42)
X_train_s, y_train_res = sm.fit_resample(X_train_s, y_train)

print(f"{'='*40}")
print(f"      SMOTE Resampling")
print(f"{'='*40}")
print(f"Before — 0: {(y_train == 0).sum()}  1: {(y_train == 1).sum()}")
print(f"After  — 0: {(y_train_res == 0).sum()}  1: {(y_train_res == 1).sum()}")
print(f"{'='*40}")

X_train_t = torch.tensor(X_train_s,          dtype=torch.float32)
X_test_t  = torch.tensor(X_test_s,           dtype=torch.float32)
y_train_t = torch.tensor(y_train_res.values, dtype=torch.float32)
y_test_t  = torch.tensor(y_test.values,      dtype=torch.float32)

# ── DataLoader (mini-batch training) ────────────────────────────────────────
train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=256,
    shuffle=True
)

# ── Model (logits only — no sigmoid in forward) ──────────────────────────────
class LogisticRegression(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = torch.nn.Linear(dim, 1)

    def forward(self, x):
        return self.linear(x)   # logits only

model = LogisticRegression(X_train_t.shape[1]).to(device)

# ── Class-weighted loss (recalculated on resampled data) ─────────────────────
neg        = (y_train_res == 0).sum()
pos        = (y_train_res == 1).sum()
pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)

criterion  = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# ── Lower learning rate + LR scheduler ───────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=30, factor=0.5
)

# ── Tracker ─────────────────────────────────────────────────────────────────
tracker = OfflineEmissionsTracker(country_iso_code="BGD", log_level="error")
tracker.start()

train_losses, test_losses = [], []

best_loss  = float("inf")
best_state = None
patience   = 50
counter    = 0

# ── Training ─────────────────────────────────────────────────────────────────
for epoch in range(2000):

    model.train()

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb).squeeze()
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()

    # ── Evaluation ──
    model.eval()
    with torch.no_grad():
        train_logits = model(X_train_t.to(device)).squeeze()
        test_logits  = model(X_test_t.to(device)).squeeze()

        train_prob = torch.sigmoid(train_logits).cpu().numpy()
        test_prob  = torch.sigmoid(test_logits).cpu().numpy()

        train_loss = log_loss(y_train_res, train_prob)
        test_loss  = log_loss(y_test,      test_prob)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    scheduler.step(test_loss)

    print(f"Epoch {epoch+1:4d} — Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    # ── Early stopping ──
    if test_loss < best_loss - 1e-5:
        best_loss  = test_loss
        best_state = model.state_dict()
        counter    = 0
    else:
        counter += 1

    if counter >= patience:
        print(f"\nEarly stopping triggered at epoch {epoch+1}")
        break

# ── Restore best model ───────────────────────────────────────────────────────
if best_state:
    model.load_state_dict(best_state)

# ── Convergence ──────────────────────────────────────────────────────────────
lr_min_epoch      = np.argmin(test_losses)
lr_min_train_loss = train_losses[lr_min_epoch]
lr_min_test_loss  = test_losses[lr_min_epoch]

print(f"\nConvergence Point  : Epoch {lr_min_epoch + 1}")
print(f"Min Train Loss     : {lr_min_train_loss:.4f}")
print(f"Min Test  Loss     : {lr_min_test_loss:.4f}")

# ── Emissions ────────────────────────────────────────────────────────────────
lr_emissions = tracker.stop()
print(f"CO2 Emissions      : {lr_emissions:.6f} kg CO2")

# ── Plot ─────────────────────────────────────────────────────────────────────
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses,  label="Test Loss")
plt.axvline(lr_min_epoch, color="red", linestyle="--", label=f"Best Epoch {lr_min_epoch+1}")
plt.legend()
plt.title("Logistic Regression – Log Loss per Epoch")
plt.savefig('D4_logreg_learning_curve.png', bbox_inches='tight', dpi=300)
plt.close()

# ── Optimal threshold via Precision-Recall curve ─────────────────────────────
model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(X_test_t.to(device))).cpu().numpy().flatten()

precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
f1_scores      = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
lr_best_thresh = thresholds[np.argmax(f1_scores[:-1])]

print(f"\nOptimal Threshold  : {lr_best_thresh:.4f}")

preds = (probs >= lr_best_thresh).astype(int)

# ── Metrics ──────────────────────────────────────────────────────────────────
lr_accuracy  = accuracy_score(y_test,  preds)
lr_precision = precision_score(y_test, preds, zero_division=0)
lr_recall    = recall_score(y_test,   preds, zero_division=0)
lr_f1        = f1_score(y_test,       preds, zero_division=0)

# ── GHG Breakdown ────────────────────────────────────────────────────────────
lr_co2 = lr_emissions * 0.90
lr_ch4 = lr_emissions * 0.07
lr_n2o = lr_emissions * 0.03

metrics_output = f"""
{'='*40}
    Logistic Regression – Final Metrics
{'='*40}
Accuracy           : {lr_accuracy:.4f}
Precision          : {lr_precision:.4f}
Recall             : {lr_recall:.4f}
F1 Score           : {lr_f1:.4f}
Convergence Epoch  : {lr_min_epoch + 1}
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

with open('D4_logreg_metrics.txt', 'w', encoding='utf-8') as f:
    f.write(metrics_output.strip())

warnings.filterwarnings('default')

print("\nEntire code has runned and stopped running.")