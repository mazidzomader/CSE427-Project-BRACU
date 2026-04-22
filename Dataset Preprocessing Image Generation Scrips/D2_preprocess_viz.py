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
url = "https://raw.githubusercontent.com/mazidzomader/CSE427-Project-BRACU/main/Datasets/heart.csv"
df = pd.read_csv(url)

# 2. Preprocessing
df.replace('?', np.nan, inplace=True)
df = pd.get_dummies(df, drop_first=True)
df = df.apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# 3. Apply SMOTE (Processing)
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
df_res = pd.DataFrame(X_res, columns=X.columns)
df_res['HeartDisease'] = y_res

# --- FIGURE 1: Correlation Graph ---
plt.figure(figsize=(12, 10))
sns.heatmap(df_res.corr(), annot=True, cmap='RdBu_r', fmt=".2f", linewidths=0.1)
plt.title("D2: Correlation Matrix (Post-Preprocessing)")
plt.savefig(os.path.join(SAVE_PATH, "D2_correlation.png"), dpi=300, bbox_inches='tight')
plt.close()

# --- FIGURE 2: Balance Graph ---
plt.figure(figsize=(6, 4))
sns.countplot(x='HeartDisease', data=df_res, palette='viridis')
plt.title("D2: Dataset Balance (Post-SMOTE)")
plt.xticks([0, 1], ['Normal (0)', 'Disease (1)'])
plt.savefig(os.path.join(SAVE_PATH, "D2_balance.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"D2 figures saved to {SAVE_PATH}")
