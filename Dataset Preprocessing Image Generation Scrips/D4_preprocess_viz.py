import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from imblearn.over_sampling import SMOTE

# Path configuration
SAVE_PATH = r"C:\Users\Mazid\Documents\GitHub\CSE427-Project-BRACU\Dataset Preprocessing Image Generation Scrips"
os.makedirs(SAVE_PATH, exist_ok=True)

# IEEE Style
plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'axes.labelsize': 12, 'axes.titlesize': 12})

# 1. Load Data
url = "https://raw.githubusercontent.com/mazidzomader/CSE427-Project-BRACU/main/Datasets/thyroidDF.csv"
df = pd.read_csv(url)

# 2. Preprocessing
cols_to_drop = ['patient_id', 'TBG', 'T4U_measured', 'T3_measured' , "FTI_measured", "TBG_measured", "TSH_measured", "TT4_measured", "referral_source"]
df = df.drop(cols_to_drop, axis=1)
num_cols = ['TSH', 'T3', 'TT4', 'T4U', 'FTI', 'age']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())
df['target'] = df['target'].apply(lambda x: 0 if x == '-' else 1)

# Convert categorical clinical flags
for col in df.select_dtypes(include=['object']).columns:
    if col != 'target':
        df[col] = df[col].map({'t': 1, 'f': 0, 'M': 1, 'F': 0}).fillna(0)

# Feature Transformation
df['TSH'] = np.log1p(df['TSH'])

X = df.drop('target', axis=1)
y = df['target']

# 3. Apply SMOTE (Processing)
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
df_res = pd.DataFrame(X_res, columns=X.columns)
df_res['target'] = y_res

# --- FIGURE 1: Correlation Graph ---
plt.figure(figsize=(14, 12))
sns.heatmap(df_res.corr(), annot=True, cmap='RdBu_r', fmt=".2f", linewidths=0.1)
plt.title("D4: Correlation Matrix (Post-Preprocessing)")
plt.savefig(os.path.join(SAVE_PATH, "D4_correlation.png"), dpi=300, bbox_inches='tight')
plt.close()

# --- FIGURE 2: Balance Graph ---
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df_res, palette='viridis')
plt.title("D4: Dataset Balance (Post-SMOTE)")
plt.xticks([0, 1], ['Normal (0)', 'Disease (1)'])
plt.savefig(os.path.join(SAVE_PATH, "D4_balance.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"D4 figures saved to {SAVE_PATH}")
