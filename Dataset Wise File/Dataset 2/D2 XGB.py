import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
print(f"XGBoost runs on    : {'GPU (cuda)' if torch.cuda.is_available() else 'CPU'}")
print(f"{'='*40}")

# ── 4. Training ──────────────────────────────────────────────────────────────
xgb_tracker = OfflineEmissionsTracker(country_iso_code="BGD", log_level="error")
xgb_tracker.start()

xgb_model = XGBClassifier(
    n_estimators=100,
    random_state=42,
    eval_metric='logloss',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbosity=0
)

# Train on SMOTE data
xgb_model.fit(
    X_train_res, y_train_res,
    eval_set=[(X_train_res, y_train_res), (X_test_s, y_test)],
    verbose=False
)

xgb_results = xgb_model.evals_result()
xgb_train_losses = xgb_results['validation_0']['logloss']
xgb_test_losses  = xgb_results['validation_1']['logloss']

xgb_min_iter      = np.argmin(xgb_test_losses)
xgb_emissions = xgb_tracker.stop()

# ── 5. Metrics ───────────────────────────────────────────────────────────────
xgb_y_pred = xgb_model.predict(X_test_s)
xgb_accuracy  = accuracy_score(y_test,  xgb_y_pred)
xgb_precision = precision_score(y_test, xgb_y_pred)
xgb_recall    = recall_score(y_test,    xgb_y_pred)
xgb_f1        = f1_score(y_test,        xgb_y_pred)

# ── 6. Results & Output ──────────────────────────────────────────────────────
xgb_ghg_co2, xgb_ghg_ch4, xgb_ghg_n2o = xgb_emissions * 0.90, xgb_emissions * 0.07, xgb_emissions * 0.03
metrics_output = f"""
{'='*40}
    XGBoost (GPU/cuda) – Final Metrics
{'='*40}
Accuracy           : {xgb_accuracy:.4f}
Precision          : {xgb_precision:.4f}
Recall             : {xgb_recall:.4f}
F1 Score           : {xgb_f1:.4f}
Convergence Iter   : {xgb_min_iter + 1}
CO2 Emissions      : {xgb_emissions:.6f} kg CO2
{'='*40}
   GHG Breakdown (Approximate)
{'='*40}
CO2 (kg)           : {xgb_ghg_co2:.8f}
CH4 (kg)           : {xgb_ghg_ch4:.8f}
N2O (kg)           : {xgb_ghg_n2o:.8f}
Total CO2eq (kg)   : {xgb_emissions:.8f}
{'='*40}
"""
print(metrics_output)
with open('D2_xgb_metrics.txt', 'w', encoding='utf-8') as f:
    f.write(metrics_output.strip())

plt.figure()
plt.plot(xgb_train_losses, label='Train Loss')
plt.plot(xgb_test_losses,  label='Test Loss')
plt.axvline(x=xgb_min_iter, color='red', linestyle='--', label=f'Best (Iter {xgb_min_iter+1})')
plt.title("XGBoost Learning Curve (D2 Heart Disease)")
plt.xlabel("Iteration")
plt.ylabel("Log Loss")
plt.legend()
plt.savefig('D2_xgb_learning_curve.png', bbox_inches='tight', dpi=300)
plt.close()

print("\nEntire code has runned and stopped running.")
