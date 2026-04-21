import pandas as pd
import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
print(f"XGBoost runs on    : {str(device).upper()} (Cleaned)")
print(f"{'='*40}")


# ── 3. Training ──────────────────────────────────────────────────────────────
tracker = OfflineEmissionsTracker(country_iso_code="BGD", log_level="error")
tracker.start()

if len(le.classes_) > 2:
    xgb_params = {'objective': 'multi:softprob', 'num_class': len(le.classes_), 'eval_metric': 'mlogloss'}
else:
    xgb_params = {'objective': 'binary:logistic', 'eval_metric': 'logloss'}

xgb_model = XGBClassifier(
    n_estimators=100,
    random_state=42,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbosity=0,
    **xgb_params
)

xgb_model.fit(X_train_res, y_train_res, eval_set=[(X_train_res, y_train_res), (X_test_s, y_test)], verbose=False)

xgb_results = xgb_model.evals_result()
xgb_test_losses = xgb_results['validation_1'][xgb_params['eval_metric']]
xgb_min_iter = np.argmin(xgb_test_losses)
emissions = tracker.stop()

# ── 4. Metrics & Output ──────────────────────────────────────────────────────
y_pred = xgb_model.predict(X_test_s)
acc, pr, rc, f1 = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='weighted'), recall_score(y_test, y_pred, average='weighted'), f1_score(y_test, y_pred, average='weighted')

metrics_output = f"""
{'='*40}
    XGBoost (Cleaned D6) – Final Metrics
{'='*40}
Accuracy           : {acc:.4f}
Precision (wt)     : {pr:.4f}
Recall (wt)        : {rc:.4f}
F1 Score (wt)      : {f1:.4f}
Convergence Iter   : {xgb_min_iter + 1}
CO2 Emissions      : {emissions:.6f} kg CO2
{'='*40}
"""
print(metrics_output)
with open('D6_xgb_metrics.txt', 'w', encoding='utf-8') as f: f.write(metrics_output.strip())

plt.figure(); plt.plot(xgb_test_losses, label='Test Loss'); plt.axvline(x=xgb_min_iter, color='red', linestyle='--')
plt.title("XGBoost Learning Curve (Cleaned D6)"); plt.savefig('D6_xgb_learning_curve.png'); plt.close()

print("\nEntire code has runned and stopped running.")
