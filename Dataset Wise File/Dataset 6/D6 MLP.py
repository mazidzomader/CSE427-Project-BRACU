import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
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
print(f"MLP runs on        : {str(device).upper()} (Cleaned)")
print(f"{'='*40}")


X_train_t = torch.tensor(X_train_res).to(device)
X_val_t   = torch.tensor(X_val_s).to(device)
X_test_t  = torch.tensor(X_test_s).to(device)
y_train_t = torch.tensor(y_train_res, dtype=torch.long).to(device)
y_val_t   = torch.tensor(y_val, dtype=torch.long).to(device)

# ── 4. Model ─────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_classes))
    def forward(self, x): return self.net(x)

model = MLP(X_train_t.shape[1], n_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ── 5. Training ──────────────────────────────────────────────────────────────
tracker = OfflineEmissionsTracker(country_iso_code="BGD", log_level="error")
tracker.start()

val_losses = []
best_loss, best_epoch, patience, counter, best_state = float('inf'), 0, 30, 0, None

for epoch in range(500):
    model.train(); optimizer.zero_grad()
    loss = criterion(model(X_train_t), y_train_t)
    loss.backward(); optimizer.step()

    model.eval()
    with torch.no_grad():
        v_loss = criterion(model(X_val_t), y_val_t).item()
    val_losses.append(v_loss)

    if v_loss < best_loss - 1e-4:
        best_loss, best_epoch, counter, best_state = v_loss, epoch, 0, model.state_dict()
    else:
        counter += 1
    if counter >= patience: break

if best_state: model.load_state_dict(best_state)
emissions = tracker.stop()

# ── 6. Metrics & Output ──────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    probs = torch.softmax(model(X_test_t), dim=1).cpu().numpy()
y_pred = np.argmax(probs, axis=1)
acc, pr, rc, f1 = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='weighted'), recall_score(y_test, y_pred, average='weighted'), f1_score(y_test, y_pred, average='weighted')

metrics_output = f"""
{'='*40}
    MLP (Cleaned D6) – Final Metrics
{'='*40}
Accuracy           : {acc:.4f}
Precision (wt)     : {pr:.4f}
Recall (wt)        : {rc:.4f}
F1 Score (wt)      : {f1:.4f}
Best Epoch         : {best_epoch + 1}
CO2 Emissions      : {emissions:.6f} kg CO2
{'='*40}
"""
print(metrics_output)
with open('D6_mlp_metrics.txt', 'w', encoding='utf-8') as f: f.write(metrics_output.strip())

plt.figure(); plt.plot(val_losses, label="Val Loss"); plt.axvline(best_epoch, color="red", linestyle="--")
plt.title("MLP Learning Curve (Cleaned D6)"); plt.savefig('D6_mlp_learning_curve.png'); plt.close()

print("\nEntire code has runned and stopped running.")
