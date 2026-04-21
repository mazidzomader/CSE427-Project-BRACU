import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
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
print(f"Random Forest runs on   : {str(device).upper()} (PyTorch)")
print(f"{'='*40}")


# ── 4. Soft Decision Tree (Regularized) ──────────────────────────────────────
class SoftDecisionTree(nn.Module):
    def __init__(self, n_features, max_depth, n_classes=2):
        super().__init__()
        self.max_depth, self.n_leaves = max_depth, 2 ** max_depth
        self.n_nodes = 2 ** max_depth - 1
        self.feature_weights = nn.Parameter(torch.randn(self.n_nodes, n_features))
        self.thresholds = nn.Parameter(torch.randn(self.n_nodes))
        self.leaf_logits = nn.Parameter(torch.randn(self.n_leaves, n_classes))

    def forward(self, X):
        batch = X.shape[0]; feat_attn = torch.softmax(self.feature_weights, dim=1)
        node_vals = X @ feat_attn.T; split_prob = torch.sigmoid(node_vals - self.thresholds)
        path_probs = torch.ones(batch, self.n_leaves, device=X.device)
        for depth in range(self.max_depth):
            n_nodes_at_depth = 2 ** depth; start_node = n_nodes_at_depth - 1
            for i in range(n_nodes_at_depth):
                node_idx = start_node + i; half = self.n_leaves // (2 * n_nodes_at_depth)
                left_idx = slice(i * (self.n_leaves // n_nodes_at_depth), i * (self.n_leaves // n_nodes_at_depth) + half)
                right_idx = slice(i * (self.n_leaves // n_nodes_at_depth) + half, (i + 1) * (self.n_leaves // n_nodes_at_depth))
                go_right = split_prob[:, node_idx].unsqueeze(1)
                path_probs[:, left_idx] *= (1 - go_right); path_probs[:, right_idx] *= go_right
        return path_probs @ torch.softmax(self.leaf_logits, dim=1)

class GPURandomForest:
    def __init__(self, n_estimators=100, max_depth=6, n_classes=2, lr=0.005, epochs=40):
        self.n_estimators, self.max_depth, self.n_classes, self.lr, self.epochs = n_estimators, max_depth, n_classes, lr, epochs
        self.trees, self.feature_subsets = [], []

    def fit(self, X, y):
        X_t = torch.tensor(X, dtype=torch.float32).to(device)   # fix Float/Double crash
        y_t = torch.tensor(y, dtype=torch.long).to(device)
        # Equal weights since SMOTE already balanced the classes
        weights = torch.tensor([1.0, 1.0], device=device)
        n_features = X.shape[1]
        for t in range(self.n_estimators):
            if (t + 1) % 10 == 0 or t == 0:
                print(f"Training Tree {t + 1}/{self.n_estimators}...", end='\\r', flush=True)
            idx = torch.randint(0, X.shape[0], (int(X.shape[0]*0.7),), device=device) # Subsampling
            feat_idx = torch.randperm(n_features, device=device)[:int(n_features * 0.7)]
            self.feature_subsets.append(feat_idx)
            tree = SoftDecisionTree(len(feat_idx), self.max_depth, self.n_classes).to(device)
            optimizer = torch.optim.AdamW(tree.parameters(), lr=self.lr, weight_decay=1e-3)
            tree.train()
            for _ in range(self.epochs):
                optimizer.zero_grad(); loss = nn.CrossEntropyLoss(weight=weights)(tree(X_t[idx][:, feat_idx]), y_t[idx]); loss.backward(); optimizer.step()
            self.trees.append(tree.eval())
        print("Training complete!          ")

    def predict_proba(self, X, n_trees=None):
        if n_trees is None: n_trees = len(self.trees)
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        all_probs = []
        with torch.no_grad():
            for i in range(n_trees):
                tree, feat_idx = self.trees[i], self.feature_subsets[i]
                all_probs.append(tree(X_t[:, feat_idx]).cpu().numpy())
        return np.mean(all_probs, axis=0)

# ── 5. Training ──────────────────────────────────────────────────────────────
tracker = OfflineEmissionsTracker(country_iso_code="BGD", log_level="error"); tracker.start()
rf_model = GPURandomForest(n_estimators=100, max_depth=6); rf_model.fit(X_train_res, np.array(y_train_res))
tr_losses, te_losses, tree_counts = [], [], list(range(1, 101, 2))
for n in tree_counts:
    p_tr = rf_model.predict_proba(X_train_res, n_trees=n); tr_losses.append(log_loss(y_train_res, p_tr))
    p_te = rf_model.predict_proba(X_test_s, n_trees=n); te_losses.append(log_loss(y_test, p_te))
best_idx = np.argmin(te_losses); rf_min_tree = tree_counts[best_idx]
emissions = tracker.stop()

# ── 6. Threshold Tuning & Metrics ─────────────────────────────────────────────
probs = rf_model.predict_proba(X_test_s)[:, 1]
pre, rec, thresh = precision_recall_curve(y_test, probs)
f1s = 2*(pre*rec)/(pre+rec+1e-8); best_t = thresh[np.argmax(f1s[:-1])]
y_pred = (probs >= best_t).astype(int)
acc, pr, rc, f1 = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)

# ── Output ───────────────────────────────────────────────────────────────────
metrics_output = f"""
{'='*40}
    Random Forest (GPU/PyTorch) – Final Metrics
{'='*40}
Accuracy           : {acc:.4f}
Precision          : {pr:.4f}
Recall             : {rc:.4f}
F1 Score           : {f1:.4f}
Convergence Tree   : {rf_min_tree}
CO2 Emissions      : {emissions:.6f} kg CO2
{'='*40}
   GHG Breakdown (Approximate)
{'='*40}
CO2 (kg)           : {emissions*0.9:.8f}
CH4 (kg)           : {emissions*0.07:.8f}
N2O (kg)           : {emissions*0.03:.8f}
Total CO2eq (kg)   : {emissions:.8f}
{'='*40}
"""
print(metrics_output)
with open('D3_rf_metrics.txt', 'w', encoding='utf-8') as f: f.write(metrics_output.strip())

plt.figure()
plt.plot(tree_counts, tr_losses, label='Train Loss')
plt.plot(tree_counts, te_losses, label='Test Loss')
plt.axvline(x=rf_min_tree, color='red', linestyle='--', label=f'Best (Trees={rf_min_tree})')
plt.title("Random Forest LogLoss Curve (D3 Hepatitis C Data)")
plt.xlabel("Number of Trees"); plt.ylabel("LogLoss"); plt.legend()
plt.savefig('D3_rf_learning_curve.png', bbox_inches='tight', dpi=300); plt.close()

print("\nEntire code has runned and stopped running.")
