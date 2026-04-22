import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Path configuration
SAVE_PATH = r"C:\Users\Mazid\Documents\GitHub\CSE427-Project-BRACU\Dataset Preprocessing Image Generation Scrips"
os.makedirs(SAVE_PATH, exist_ok=True)

# IEEE Style
plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'axes.labelsize': 12, 'axes.titlesize': 12})

# 1. Load Data
url = "https://raw.githubusercontent.com/mazidzomader/CSE427-Project-BRACU/main/Datasets/Asthma.csv"
df = pd.read_csv(url)

# 2. Preprocessing
df = df.drop(columns=['PatientID', 'DoctorInCharge'], errors='ignore')
le = LabelEncoder()
df['Diagnosis'] = le.fit_transform(df['Diagnosis'])

# Fill missing numerical
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].median())

X = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]

# 3. Apply SMOTE (Processing)
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
df_res = pd.DataFrame(X_res, columns=X.columns)
df_res['Diagnosis'] = y_res

# --- FIGURE 1: Correlation Graph ---
plt.figure(figsize=(14, 12))
sns.heatmap(df_res.corr(), annot=True, cmap='RdBu_r', fmt=".2f", linewidths=0.1)
plt.title("D6: Correlation Matrix (Post-Preprocessing)")
plt.savefig(os.path.join(SAVE_PATH, "D6_correlation.png"), dpi=300, bbox_inches='tight')
plt.close()

# --- FIGURE 2: Balance Graph ---
plt.figure(figsize=(6, 4))
sns.countplot(x='Diagnosis', data=df_res, palette='viridis')
plt.title("D6: Dataset Balance (Post-SMOTE)")
plt.xticks([0, 1], ['Negative (0)', 'Positive (1)'])
plt.savefig(os.path.join(SAVE_PATH, "D6_balance.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"D6 figures saved to {SAVE_PATH}")
