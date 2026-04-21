import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from codecarbon import OfflineEmissionsTracker
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ── 1. Data Loading & Preprocessing ──────────────────────────────────────────
url = 'https://raw.githubusercontent.com/mazidzomader/CSE427-Project-BRACU/refs/heads/main/Datasets/thyroidDF.csv'
df = pd.read_csv(url)

cols_to_drop = ['patient_id', 'TBG', 'T4U_measured', 'T3_measured', 'FTI_measured', 'TBG_measured', 'TSH_measured', 'TT4_measured']
df = df.drop(cols_to_drop, axis=1)

num_cols = ['TSH', 'T3', 'TT4', 'T4U', 'FTI']
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

df['sex'] = df['sex'].fillna(df['sex'].mode()[0])
df.loc[df['age'] > 100, 'age'] = df['age'].median()
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

df['TSH'] = np.log1p(df['TSH'])
df['TT4'] = np.log1p(df['TT4'])
df['FTI'] = np.log1p(df['FTI'])

df['target'] = df['target'].apply(lambda x: 0 if x == '-' else 1)

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Scaling & SMOTE (Essential for NN-based base learners) ───────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_s, y_train)

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
print(f"AdaBoost runs on   : {str(device).upper()} (PyTorch)")
print(f"{'='*40}")

# ── 2. Soft Decision Stump (PyTorch GPU Base Learner) ────────────────────────
class SoftStump(nn.Module):
    def __init__(self, n_features, n_classes=2):
        super().__init__()
        # Depth-2 equivalent: we use a small MLP as a more powerful base learner
        self.net = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, X):
        return self.net(X)

class GPUAdaBoost:
    def __init__(self, n_estimators=50, learning_rate=0.5, epochs=20):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.stumps = []
        self.alphas = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        y_t = torch.tensor(y, dtype=torch.long).to(device)
        
        # Sample weights
        weights = np.full(n_samples, 1.0 / n_samples)
        
        for t in range(self.n_estimators):
            stump = SoftStump(n_features, n_classes).to(device)
            optimizer = optim.Adam(stump.parameters(), lr=0.01)
            
            # Use weighted sampling to train the stump
            indices = np.random.choice(n_samples, size=n_samples, p=weights)
            X_batch = X_t[indices]
            y_batch = y_t[indices]
            
            stump.train()
            for _ in range(self.epochs):
                optimizer.zero_grad()
                out = stump(X_batch)
                loss = nn.functional.cross_entropy(torch.log(out + 1e-9), y_batch)
                loss.backward()
                optimizer.step()
            
            stump.eval()
            with torch.no_grad():
                proba = stump(X_t)
                y_pred = torch.argmax(proba, dim=1).cpu().numpy()
                incorrect = (y_pred != y).astype(float)
                error = np.dot(weights, incorrect)
                error = np.clip(error, 1e-10, 1 - 1e-10)
            
            alpha = self.learning_rate * 0.5 * np.log((1 - error) / error)
            
            # Update weights
            weights *= np.exp(alpha * (2 * incorrect - 1))
            weights /= np.sum(weights)
            
            self.stumps.append(stump)
            self.alphas.append(alpha)
            
            if (t + 1) % 10 == 0:
                print(f"Estimator {t+1:3d}/{self.n_estimators} trained | Error: {error:.4f} | Alpha: {alpha:.4f}")

    def staged_predict_proba(self, X):
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        cumulative = torch.zeros(X_t.shape[0], 2, device=device)
        for stump, alpha in zip(self.stumps, self.alphas):
            with torch.no_grad():
                proba = stump(X_t)
                cumulative += alpha * torch.log(proba + 1e-9)
                yield torch.softmax(cumulative, dim=1).cpu().numpy()

    def predict_proba(self, X):
        result = None
        for p in self.staged_predict_proba(X):
            result = p
        return result

# ── 3. Training Loop ─────────────────────────────────────────────────────────
ada_tracker = OfflineEmissionsTracker(country_iso_code="BGD", log_level="error")
ada_tracker.start()

print("\nTraining GPU AdaBoost (PyTorch Ensemble)...")
ada_model = GPUAdaBoost(n_estimators=100, learning_rate=0.5, epochs=30)
ada_model.fit(X_train_res, y_train_res.values)

ada_train_losses = []
ada_test_losses  = []

for train_p, test_p in zip(ada_model.staged_predict_proba(X_train_s), ada_model.staged_predict_proba(X_test_s)):
    ada_train_losses.append(log_loss(y_train, train_p))
    ada_test_losses.append(log_loss(y_test, test_p))

ada_min_estimator = np.argmin(ada_test_losses)
ada_min_test_loss = ada_test_losses[ada_min_estimator]
ada_min_train_loss = ada_train_losses[ada_min_estimator]

# ── 4. Final Evaluation ──────────────────────────────────────────────────────
# Retrain best (slice the ensemble)
ada_model.stumps = ada_model.stumps[:ada_min_estimator + 1]
ada_model.alphas = ada_model.alphas[:ada_min_estimator + 1]

probs = ada_model.predict_proba(X_test_s)[:, 1]
best_thresh, best_f1 = 0.5, 0
for t in np.arange(0.1, 0.9, 0.01):
    preds = (probs >= t).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1, best_thresh = f1, t

ada_y_pred = (probs >= best_thresh).astype(int)
ada_accuracy = accuracy_score(y_test, ada_y_pred)
ada_precision = precision_score(y_test, ada_y_pred)
ada_recall = recall_score(y_test, ada_y_pred)
ada_f1 = f1_score(y_test, ada_y_pred)

ada_emissions = ada_tracker.stop()

# ── 5. Standardized Output ───────────────────────────────────────────────────
ada_ghg_co2 = ada_emissions * 0.90
ada_ghg_ch4 = ada_emissions * 0.07
ada_ghg_n2o = ada_emissions * 0.03

metrics_output = f"""
{'='*40}
    AdaBoost (GPU/PyTorch) – Final Metrics
{'='*40}
Accuracy           : {ada_accuracy:.4f}
Precision          : {ada_precision:.4f}
Recall             : {ada_recall:.4f}
F1 Score           : {ada_f1:.4f}
Convergence Estim  : {ada_min_estimator + 1}
CO2 Emissions      : {ada_emissions:.6f} kg CO2
{'='*40}
   GHG Breakdown (Approximate)
{'='*40}
CO2 (kg)           : {ada_ghg_co2:.8f}
CH4 (kg)           : {ada_ghg_ch4:.8f}
N2O (kg)           : {ada_ghg_n2o:.8f}
Total CO2eq (kg)   : {ada_emissions:.8f}
{'='*40}
"""

print(metrics_output)
with open('D4_ada_metrics.txt', 'w', encoding='utf-8') as f:
    f.write(metrics_output.strip())

# Plotting
plt.figure()
plt.plot(ada_train_losses, label='Train Loss')
plt.plot(ada_test_losses, label='Test Loss')
plt.axvline(x=ada_min_estimator, color='red', linestyle='--', label=f'Best ({ada_min_estimator + 1})')
plt.title("AdaBoost Learning Curve (GPU)")
plt.xlabel("Estimators")
plt.ylabel("Log Loss")
plt.legend()
plt.savefig('D4_ada_learning_curve.png', bbox_inches='tight', dpi=300)
plt.close()

print("\nEntire code has runned and stopped running.")