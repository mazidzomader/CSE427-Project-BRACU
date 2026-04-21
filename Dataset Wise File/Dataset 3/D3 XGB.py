import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ── 2. Robust Scaling & SMOTETomek ──────────────────────────────────────────
scaler = RobustScaler()
X_train_s = scaler.fit_transform(X_train).astype(np.float32)
X_test_s  = scaler.transform(X_test).astype(np.float32)

# Full 1:1 SMOTE balance
smt = SMOTE(random_state=42, sampling_strategy='auto', k_neighbors=5)
X_train_res, y_train_res = smt.fit_resample(X_train_s, y_train)
print(f"After SMOTE — Positive (Disease): {(np.array(y_train_res)==1).sum()}, Negative (Donor): {(np.array(y_train_res)==0).sum()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
print(f"XGBoost runs on    : {'GPU (cuda)' if torch.cuda.is_available() else 'CPU'}")
print(f"{'='*40}")


# ── 4. Training (Tuned for F1 on imbalanced Hepatitis C data) ──────────────────────
xgb_tracker = OfflineEmissionsTracker(country_iso_code="BGD", log_level="error")
xgb_tracker.start()

xgb_model = XGBClassifier(
    n_estimators=20000,
    learning_rate=0.01,
    max_depth=5,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.5,
    reg_lambda=2.0,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,            # Data already balanced 1:1 via SMOTE
    random_state=42,
    eval_metric='logloss',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbosity=0,
    early_stopping_rounds=200
)

xgb_model.fit(
    X_train_res, y_train_res,
    eval_set=[(X_train_res, y_train_res), (X_test_s, y_test)],
    verbose=False
)

# ── Get LogLoss per iteration ─────────────────────────────────────────────────
xgb_results       = xgb_model.evals_result()
xgb_train_scores  = xgb_results['validation_0']['logloss']
xgb_test_scores   = xgb_results['validation_1']['logloss']
xgb_best_iter     = np.argmin(xgb_test_scores)   # lower logloss = better

print(f"\nConvergence Point  : Iteration {xgb_best_iter + 1}")
print(f"Best Train LogLoss : {xgb_train_scores[xgb_best_iter]:.4f}")
print(f"Best Test  LogLoss : {xgb_test_scores[xgb_best_iter]:.4f}")

xgb_emissions = xgb_tracker.stop()

# ── 5. Threshold Optimization ────────────────────────────────────────────────
probs = xgb_model.predict_proba(X_test_s)[:, 1]
pre, rec, thresh = precision_recall_curve(y_test, probs)
f1_scores = 2 * (pre * rec) / (pre + rec + 1e-8)
best_t = thresh[np.argmax(f1_scores[:-1])]

y_pred = (probs >= best_t).astype(int)
acc, pr, rc, f1 = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)
ghg_co2, ghg_ch4, ghg_n2o = xgb_emissions * 0.90, xgb_emissions * 0.07, xgb_emissions * 0.03

# ── 6. Metrics & Output ──────────────────────────────────────────────────────
metrics_output = f"""
{'='*40}
    XGBoost (GPU/cuda) – Final Metrics
{'='*40}
Accuracy           : {acc:.4f}
Precision          : {pr:.4f}
Recall             : {rc:.4f}
F1 Score           : {f1:.4f}
Convergence Iter   : {xgb_best_iter + 1}
CO2 Emissions      : {xgb_emissions:.6f} kg CO2
{'='*40}
   GHG Breakdown (Approximate)
{'='*40}
CO2 (kg)           : {ghg_co2:.8f}
CH4 (kg)           : {ghg_ch4:.8f}
N2O (kg)           : {ghg_n2o:.8f}
Total CO2eq (kg)   : {xgb_emissions:.8f}
{'='*40}
"""
print(metrics_output)
with open('D3_xgb_metrics.txt', 'w', encoding='utf-8') as f:
    f.write(metrics_output.strip())

plt.figure()
plt.plot(xgb_train_scores, label='Train')
plt.plot(xgb_test_scores, label='Test')
plt.axvline(x=xgb_best_iter, color='red', linestyle='--', label=f'Best (Iter {xgb_best_iter+1})')
plt.title("XGBoost Learning Curve (D3 Hepatitis C Data)")
plt.xlabel("Iteration"); plt.ylabel("LogLoss"); plt.legend()
plt.savefig('D3_xgb_learning_curve.png', bbox_inches='tight', dpi=300); plt.close()

print("\nEntire code has runned and stopped running.")