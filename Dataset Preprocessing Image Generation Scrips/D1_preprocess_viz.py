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
url = "https://raw.githubusercontent.com/mazidzomader/CSE427-Project-BRACU/main/Datasets/breast_cancer_bd.csv"
df = pd.read_csv(url)

# 2. Preprocessing
df.replace('?', np.nan, inplace=True)
df = df.apply(pd.to_numeric)
df.dropna(inplace=True)
df['Class'] = df['Class'].map({2: 0, 4: 1})
df = df.drop("Sample code number", axis=1)

X = df.drop("Class", axis=1)
y = df["Class"]

# 3. Apply SMOTE (Processing)
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
df_res = pd.DataFrame(X_res, columns=X.columns)
df_res['Class'] = y_res

# --- FIGURE 1: Correlation Graph ---
plt.figure(figsize=(10, 8))
sns.heatmap(df_res.corr(), annot=True, cmap='RdBu_r', fmt=".2f", linewidths=0.5)
plt.title("D1: Correlation Matrix (Post-Preprocessing)")
plt.savefig(os.path.join(SAVE_PATH, "D1_correlation.png"), dpi=300, bbox_inches='tight')
plt.close()

# --- FIGURE 2: Balance Graph ---
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df_res, palette='viridis')
plt.title("D1: Dataset Balance (Post-SMOTE)")
plt.xticks([0, 1], ['Benign (0)', 'Malignant (1)'])
plt.savefig(os.path.join(SAVE_PATH, "D1_balance.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"D1 figures saved to {SAVE_PATH}")
