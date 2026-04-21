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


from xgboost import XGBClassifier
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from codecarbon import OfflineEmissionsTracker
import matplotlib.pyplot as plt
import numpy as np
import warnings
import torch

warnings.filterwarnings('ignore')

# ── Check GPU Status ─────────────────────────────────────────────────────────
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

# ── 1. Initialise CodeCarbon Emissions Tracker ───────────────────────────────
xgb_tracker = OfflineEmissionsTracker(country_iso_code="BGD", log_level="error")
xgb_tracker.start()

xgb_train_losses = []
xgb_test_losses  = []

# ── 2. Train XGBoost on GPU; record loss per iteration ───────────────────────
xgb_model = XGBClassifier(
    n_estimators=100,
    random_state=42,
    eval_metric='logloss',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbosity=0
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

# ── Get loss per iteration from eval results ──────────────────────────────────
xgb_results      = xgb_model.evals_result()
xgb_train_losses = xgb_results['validation_0']['logloss']
xgb_test_losses  = xgb_results['validation_1']['logloss']

# ── 3. Identify minimum-loss iteration (convergence point) ───────────────────
xgb_min_iter      = np.argmin(xgb_test_losses)
xgb_min_test_loss = xgb_test_losses[xgb_min_iter]
xgb_min_train_loss= xgb_train_losses[xgb_min_iter]

print(f"\nConvergence Point  : Iteration {xgb_min_iter + 1}")
print(f"Min Train Loss     : {xgb_min_train_loss:.4f}")
print(f"Min Test  Loss     : {xgb_min_test_loss:.4f}")

# ── 4. Stop tracker; record CO2 emissions ────────────────────────────────────
xgb_emissions = xgb_tracker.stop()
print(f"CO2 Emissions      : {xgb_emissions:.6f} kg CO2")

# ── Plot ─────────────────────────────────────────────────────────────────────
plt.figure()
plt.plot(xgb_train_losses, label='Train Loss')
plt.plot(xgb_test_losses,  label='Test Loss')
plt.axvline(x=xgb_min_iter, color='red', linestyle='--', label=f'Convergence (Iter {xgb_min_iter+1})')
plt.scatter(xgb_min_iter, xgb_min_test_loss, color='red', zorder=5)
plt.title("XGBoost (GPU) – Log Loss per Iteration")
plt.xlabel("Iteration")
plt.ylabel("Log Loss")
plt.legend()
plt.savefig('D4_xgb_learning_curve.png', bbox_inches='tight', dpi=300)
plt.close()

# ── 5. Metrics ───────────────────────────────────────────────────────────────
xgb_y_pred       = xgb_model.predict(X_test)
xgb_y_pred_proba = xgb_model.predict_proba(X_test)

xgb_accuracy  = accuracy_score(y_test,  xgb_y_pred)
xgb_precision = precision_score(y_test, xgb_y_pred)
xgb_recall    = recall_score(y_test,    xgb_y_pred)
xgb_f1        = f1_score(y_test,        xgb_y_pred)

# ── GHG Breakdown ────────────────────────────────────────────────────────────
xgb_ghg_co2 = xgb_emissions * 0.90
xgb_ghg_ch4 = xgb_emissions * 0.07
xgb_ghg_n2o = xgb_emissions * 0.03

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

with open('D4_xgb_metrics.txt', 'w', encoding='utf-8') as f:
    f.write(metrics_output.strip())

warnings.filterwarnings('default')

print("\nEntire code has runned and stopped running.")