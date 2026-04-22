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
url = "https://raw.githubusercontent.com/mazidzomader/CSE427-Project-BRACU/main/Datasets/HepatitisCdata.csv"
df = pd.read_csv(url)
df = df.drop(df.columns[0], axis=1)

# 2. Preprocessing
df['Category'] = df['Category'].apply(lambda x: 0 if str(x).startswith('0') else 1)
df['Sex'] = df['Sex'].map({'m': 1, 'f': 0}).fillna(0)
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].median())

# Clinical Ratios
df['AST_ALT_ratio'] = df['AST'] / (df['ALT'] + 1e-8)
df['ALB_PROT_ratio'] = df['ALB'] / (df['PROT'] + 1e-8)

X = df.drop('Category', axis=1)
y = df['Category']

# 3. Apply SMOTE (Processing)
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
df_res = pd.DataFrame(X_res, columns=X.columns)
df_res['Category'] = y_res

# --- FIGURE 1: Correlation Graph ---
plt.figure(figsize=(10, 8))
sns.heatmap(df_res.corr(), annot=True, cmap='RdBu_r', fmt=".2f", linewidths=0.5)
plt.title("D3: Correlation Matrix (Post-Preprocessing)")
plt.savefig(os.path.join(SAVE_PATH, "D3_correlation.png"), dpi=300, bbox_inches='tight')
plt.close()

# --- FIGURE 2: Balance Graph ---
plt.figure(figsize=(6, 4))
sns.countplot(x='Category', data=df_res, palette='viridis')
plt.title("D3: Dataset Balance (Post-SMOTE)")
plt.xticks([0, 1], ['Donor (0)', 'Disease (1)'])
plt.savefig(os.path.join(SAVE_PATH, "D3_balance.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"D3 figures saved to {SAVE_PATH}")
