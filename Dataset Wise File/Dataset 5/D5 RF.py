import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from codecarbon import OfflineEmissionsTracker
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ── 1. Data Loading & Preprocessing ──────────────────────────────────────────
url = 'https://raw.githubusercontent.com/mazidzomader/CSE427-Project-BRACU/refs/heads/main/Datasets/diabetes.csv'
df = pd.read_csv(url)

# In this dataset, 0 is often a missing value for these columns
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zeros:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

X = df.drop(columns=['Outcome'])
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 2. Scaling & SMOTE ───────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

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
print(f"Random Forest runs on : {str(device).upper()} (PyTorch)")
print(f"{'='*40}")

X_train_np = X_train_res.astype(np.float32)
X_test_np  = X_test_s.astype(np.float32)
y_train_np = y_train_res.values if hasattr(y_train_res, 'values') else np.array(y_train_res)
y_test_np  = y_test.values if hasattr(y_test, 'values') else np.array(y_test)

# ── 4. Soft Decision Tree (GPU-compatible via PyTorch) ────────────────────────
class SoftDecisionTree(nn.Module):
    def __init__(self, n_features, max_depth, n_classes):
        super().__init__()
        self.max_depth, self.n_leaves = max_depth, 2 ** max_depth
        self.n_nodes = 2 ** max_depth - 1
        self.feature_weights = nn.Parameter(torch.randn(self.n_nodes, n_features))
        self.thresholds = nn.Parameter(torch.randn(self.n_nodes))
        self.leaf_logits = nn.Parameter(torch.randn(self.n_leaves, n_classes))

    def forward(self, X):
        batch = X.shape[0]
        feat_attn = torch.softmax(self.feature_weights, dim=1)
        node_vals = X @ feat_attn.T
        split_prob = torch.sigmoid(node_vals - self.thresholds)
        path_probs = torch.ones(batch, self.n_leaves, device=X.device)
        for depth in range(self.max_depth):
            n_nodes_at_depth = 2 ** depth
            start_node = n_nodes_at_depth - 1
            for i in range(n_nodes_at_depth):
                node_idx = start_node + i
                half = self.n_leaves // (2 * n_nodes_at_depth)
                left_idx  = slice(i * (self.n_leaves // n_nodes_at_depth), i * (self.n_leaves // n_nodes_at_depth) + half)
                right_idx = slice(i * (self.n_leaves // n_nodes_at_depth) + half, (i + 1) * (self.n_leaves // n_nodes_at_depth))
                go_right = split_prob[:, node_idx].unsqueeze(1)
                path_probs[:, left_idx] *= (1 - go_right)
                path_probs[:, right_idx] *= go_right
        return path_probs @ torch.softmax(self.leaf_logits, dim=1)

class GPURandomForest:
    def __init__(self, n_estimators=100, max_depth=4, n_classes=2, lr=0.01, epochs=50):
        self.n_estimators, self.max_depth, self.n_classes, self.lr, self.epochs = n_estimators, max_depth, n_classes, lr, epochs
        self.trees, self.feature_subsets = [], []

    def fit(self, X, y):
        X_t, y_t = torch.tensor(X, dtype=torch.float32).to(device), torch.tensor(y, dtype=torch.long).to(device)
        n_features = X.shape[1]
        for t in range(self.n_estimators):
            idx = torch.randint(0, X.shape[0], (int(X.shape[0] * 0.8),), device=device)
            feat_idx = torch.randperm(n_features, device=device)[:int(n_features * 0.8)]
            self.feature_subsets.append(feat_idx)
            tree = SoftDecisionTree(len(feat_idx), self.max_depth, self.n_classes).to(device)
            optimizer = torch.optim.Adam(tree.parameters(), lr=self.lr)
            tree.train()
            for _ in range(self.epochs):
                optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(tree(X_t[idx][:, feat_idx]), y_t[idx])
                loss.backward()
                optimizer.step()
            self.trees.append(tree.eval())
            if (t + 1) % 20 == 0: print(f"  Trained tree {t+1:3d}/{self.n_estimators}")

    def predict_proba(self, X):
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        all_probs = []
        with torch.no_grad():
            for tree, feat_idx in zip(self.trees, self.feature_subsets):
                all_probs.append(tree(X_t[:, feat_idx]).cpu().numpy())
        return np.mean(all_probs, axis=0)

# ── 5. Training & Tracking ───────────────────────────────────────────────────
rf_tracker = OfflineEmissionsTracker(country_iso_code="BGD", log_level="error")
rf_tracker.start()

print("\nTraining GPU Random Forest (PyTorch Soft Decision Trees)...")
rf_model = GPURandomForest(n_estimators=100, max_depth=4, n_classes=len(np.unique(y_train_np)))
rf_model.fit(X_train_np, y_train_np)

rf_train_losses, rf_test_losses = [], []
tree_counts = list(range(10, rf_model.n_estimators + 1, 10))
X_tr_t = torch.tensor(X_train_np, dtype=torch.float32).to(device)
X_te_t = torch.tensor(X_test_np, dtype=torch.float32).to(device)

for n in tree_counts:
    all_tr, all_te = [], []
    with torch.no_grad():
        for tree, f_idx in zip(rf_model.trees[:n], rf_model.feature_subsets[:n]):
            all_tr.append(tree(X_tr_t[:, f_idx]).cpu().numpy())
            all_te.append(tree(X_te_t[:, f_idx]).cpu().numpy())
    tr_l, te_l = log_loss(y_train_np, np.mean(all_tr, axis=0)), log_loss(y_test_np, np.mean(all_te, axis=0))
    rf_train_losses.append(tr_l); rf_test_losses.append(te_l)
    print(f"Trees {n:3d} | Train Loss: {tr_l:.4f} | Test Loss: {te_l:.4f}")

rf_min_idx = np.argmin(rf_test_losses)
rf_min_tree = tree_counts[rf_min_idx]
rf_emissions = rf_tracker.stop()

# ── 6. Metrics & Output ──────────────────────────────────────────────────────
probs = rf_model.predict_proba(X_test_np)[:, 1]
rf_y_pred = (probs >= 0.5).astype(int)
rf_accuracy = accuracy_score(y_test_np, rf_y_pred)
rf_precision = precision_score(y_test_np, rf_y_pred, zero_division=0)
rf_recall = recall_score(y_test_np, rf_y_pred, zero_division=0)
rf_f1 = f1_score(y_test_np, rf_y_pred, zero_division=0)

rf_ghg_co2, rf_ghg_ch4, rf_ghg_n2o = rf_emissions * 0.90, rf_emissions * 0.07, rf_emissions * 0.03
metrics_output = f"""
{'='*40}
    Random Forest (GPU/PyTorch) – Final Metrics
{'='*40}
Accuracy           : {rf_accuracy:.4f}
Precision          : {rf_precision:.4f}
Recall             : {rf_recall:.4f}
F1 Score           : {rf_f1:.4f}
Convergence Tree   : {rf_min_tree}
CO2 Emissions      : {rf_emissions:.6f} kg CO2
{'='*40}
   GHG Breakdown (Approximate)
{'='*40}
CO2 (kg)           : {rf_ghg_co2:.8f}
CH4 (kg)           : {rf_ghg_ch4:.8f}
N2O (kg)           : {rf_ghg_n2o:.8f}
Total CO2eq (kg)   : {rf_emissions:.8f}
{'='*40}
"""
print(metrics_output)
with open('D5_rf_metrics.txt', 'w', encoding='utf-8') as f:
    f.write(metrics_output.strip())

plt.figure()
plt.plot(tree_counts, rf_train_losses, label='Train Loss')
plt.plot(tree_counts, rf_test_losses, label='Test Loss')
plt.axvline(x=rf_min_tree, color='red', linestyle='--', label=f'Best (Tree {rf_min_tree})')
plt.title("Random Forest Learning Curve (D5 Diabetes)")
plt.xlabel("Number of Trees")
plt.ylabel("Log Loss")
plt.legend()
plt.savefig('D5_rf_learning_curve.png', bbox_inches='tight', dpi=300)
plt.close()

print("\nEntire code has runned and stopped running.")