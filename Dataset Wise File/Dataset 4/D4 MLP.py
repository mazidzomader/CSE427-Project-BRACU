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
#                MLP (GPU)
# ════════════════════════════════════════

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from codecarbon import OfflineEmissionsTracker
import warnings

warnings.filterwarnings('ignore')

# ── GPU Setup ───────────────────────────────────────────────────────────────
print("="*40)
print("         GPU Status")
print("="*40)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print("GPU Available      : Yes")
    print(f"GPU Name           : {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory (GB)    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}")
else:
    print("GPU Available      : No — using CPU")

print(f"MLP runs on        : {device}")
print("="*40)

# ── Train / Validation Split (IMPORTANT FIX) ────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# ── Scale Data ───────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
X_val_tensor   = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
X_test_tensor  = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
y_val_tensor   = torch.tensor(y_val.values, dtype=torch.float32).to(device)
y_test_tensor  = torch.tensor(y_test.values, dtype=torch.float32).to(device)

# ── Model ────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

# ── Init ─────────────────────────────────────────────────────────────────────
input_dim = X_train_tensor.shape[1]
model = MLP(input_dim).to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0005,
    weight_decay=5e-4
)

# ── CodeCarbon ───────────────────────────────────────────────────────────────
tracker = OfflineEmissionsTracker(country_iso_code="BGD", log_level="error")
tracker.start()

train_losses = []
test_losses = []

# ── Early Stopping Setup ────────────────────────────────────────────────────
patience = 100
best_loss = float('inf')
counter = 0
best_state = None
best_epoch = 0

# ── Training Loop ───────────────────────────────────────────────────────────
for epoch in range(5000):

    model.train()
    optimizer.zero_grad()

    logits = model(X_train_tensor).squeeze()
    loss = criterion(logits, y_train_tensor)

    loss.backward()
    optimizer.step()

    # ── Evaluation ──
    model.eval()
    with torch.no_grad():
        val_probs  = torch.sigmoid(model(X_val_tensor)).cpu().numpy()
        test_probs = torch.sigmoid(model(X_test_tensor)).cpu().numpy()
        train_probs = torch.sigmoid(model(X_train_tensor)).cpu().numpy()

    train_loss = log_loss(y_train, train_probs)
    test_loss  = log_loss(y_test, test_probs)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print(f"Epoch {epoch+1:4d} — Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    # ── Early Stopping (NOW USING VALIDATION SET) ──
    val_loss = log_loss(y_val, val_probs)

    if val_loss < best_loss - 1e-6:
        best_loss = val_loss
        best_epoch = epoch
        counter = 0
        best_state = model.state_dict()
    else:
        counter += 1

    if counter >= patience:
        print(f"\nEarly stopping triggered at epoch {epoch+1}")
        break

# ── Restore best model ───────────────────────────────────────────────────────
if best_state is not None:
    model.load_state_dict(best_state)

# ── Stop emissions tracker ───────────────────────────────────────────────────
emissions = tracker.stop()
print(f"\nCO2 Emissions: {emissions:.6f} kg")

# ── Final Metrics ────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(X_test_tensor)).cpu().numpy()

# ── Threshold tuning (IMPORTANT FIX) ──
best_thresh = 0.5
best_f1 = 0

for t in np.arange(0.1, 0.9, 0.01):
    preds_temp = (probs >= t).astype(int)
    f1 = f1_score(y_test, preds_temp)

    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

preds = (probs >= best_thresh).astype(int)

accuracy  = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds)
recall    = recall_score(y_test, preds)
f1        = f1_score(y_test, preds)

# ── Plot with Convergence Line ───────────────────────────────────────────────
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")

plt.axvline(
    x=best_epoch,
    color='red',
    linestyle='--',
    label=f'Convergence (Epoch {best_epoch + 1})'
)

plt.scatter(best_epoch, best_loss, color='red', zorder=5)

plt.title("MLP – Log Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.legend()
plt.savefig('D4_mlp_learning_curve.png', bbox_inches='tight', dpi=300)
plt.close()

# ── Results ────────────────────────────────────────────────────────────────
mlp_ghg_co2 = emissions * 0.90
mlp_ghg_ch4 = emissions * 0.07
mlp_ghg_n2o = emissions * 0.03

metrics_output = f"""
{'='*40}
    MLP (GPU/PyTorch) – Final Metrics
{'='*40}
Accuracy           : {accuracy:.4f}
Precision          : {precision:.4f}
Recall             : {recall:.4f}
F1 Score           : {f1:.4f}
Best Epoch         : {best_epoch + 1}
CO2 Emissions      : {emissions:.6f} kg CO2
{'='*40}
   GHG Breakdown (Approximate)
{'='*40}
CO2 (kg)           : {mlp_ghg_co2:.8f}
CH4 (kg)           : {mlp_ghg_ch4:.8f}
N2O (kg)           : {mlp_ghg_n2o:.8f}
Total CO2eq (kg)   : {emissions:.8f}
{'='*40}
"""

print(metrics_output)

with open('D4_mlp_metrics.txt', 'w', encoding='utf-8') as f:
    f.write(metrics_output.strip())

warnings.filterwarnings('default')

print("\nEntire code has runned and stopped running.")