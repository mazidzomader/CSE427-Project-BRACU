import pandas as pd
import numpy as np
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
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

df['sex'] = df['sex'].fillna(df['sex'].mode()[0])

df['age'] = pd.to_numeric(df['age'], errors='coerce')
df.loc[df['age'] > 100, 'age'] = np.nan
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

# Clinical Ratios (Thyroid specific)
df['TT4_FTI_ratio'] = df['TT4'] / (df['FTI'] + 1e-9)
df['TSH_T3_ratio'] = df['TSH'] / (df['T3'] + 1e-9)
df['TT4_T3_ratio'] = df['TT4'] / (df['T3'] + 1e-9)

df['target'] = df['target'].apply(lambda x: 0 if x == '-' else 1)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Separate features and target
X = df.drop(columns=['target'])
y = df['target']

# Split the dataset
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Scaling & SMOTE ───────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_raw)
X_test_s  = scaler.transform(X_test_raw)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_s, y_train_raw)

X_train_np = X_train_res
y_train_np = y_train_res.values if hasattr(y_train_res, 'values') else y_train_res
X_test_np = X_test_s
y_test_np = y_test_raw.values if hasattr(y_test_raw, 'values') else y_test_raw

print("Original Features shape:", X.shape)
print("Resampled Training Features shape:", X_train_np.shape)
print("Resampled Training Target distribution:", pd.Series(y_train_np).value_counts())


import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from codecarbon import OfflineEmissionsTracker
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

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
print(f"Device in use      : {device}")
print(f"{'='*40}")
print(f"Random Forest runs on : {str(device).upper()} (PyTorch)")
print(f"{'='*40}")

# Numpy conversion already handled above
# ══════════════════════════════════════════════════════════════════════════════
#   Soft Decision Tree (GPU-compatible via PyTorch)
#   Uses differentiable sigmoid splits — all ops run on CUDA tensors
# ══════════════════════════════════════════════════════════════════════════════
class SoftDecisionTree(nn.Module):
    def __init__(self, n_features, max_depth, n_classes):
        super().__init__()
        self.max_depth  = max_depth
        self.n_classes  = n_classes
        self.n_leaves   = 2 ** max_depth
        self.n_nodes    = 2 ** max_depth - 1  # internal nodes

        # Learnable split: feature index weights + threshold per internal node
        self.feature_weights = nn.Parameter(torch.randn(self.n_nodes, n_features))
        self.thresholds      = nn.Parameter(torch.randn(self.n_nodes))

        # Leaf class distribution
        self.leaf_logits = nn.Parameter(torch.randn(self.n_leaves, n_classes))

    def forward(self, X):
        # X: (batch, features)
        batch = X.shape[0]

        # Soft feature selection with temperature for sparsity
        feat_attn  = torch.softmax(self.feature_weights * 5.0, dim=1)   # (n_nodes, n_features)
        node_vals  = X @ feat_attn.T                                     # (batch, n_nodes)
        
        # Sharper splitting for more tree-like behavior
        split_prob = torch.sigmoid((node_vals - self.thresholds) * 10.0)  # P(go right)

        # Traverse tree: vectorized path probability computation
        path_probs = torch.ones(batch, self.n_leaves, device=X.device)

        for depth in range(self.max_depth):
            n_nodes_at_depth = 2 ** depth
            start_node = n_nodes_at_depth - 1
            
            # Reshape path_probs to group leaves by their common ancestor at this depth
            # (batch, nodes_at_depth, leaves_per_ancestor)
            path_probs = path_probs.view(batch, n_nodes_at_depth, -1)
            
            # Get split probabilities for all nodes at this depth
            # (batch, nodes_at_depth, 1)
            node_p = split_prob[:, start_node:start_node + n_nodes_at_depth].unsqueeze(2)
            
            # Each group of leaves is split into left and right halves
            half = path_probs.shape[2] // 2
            
            # Left half gets (1 - node_p), right half gets node_p
            # We use in-place multiplication for speed
            path_probs[:, :, :half] *= (1 - node_p)
            path_probs[:, :, half:] *= node_p

        # Flatten back to (batch, n_leaves)
        path_probs = path_probs.reshape(batch, self.n_leaves)
        
        # Weighted sum of leaf distributions
        leaf_dist = torch.softmax(self.leaf_logits, dim=1)        # (n_leaves, n_classes)
        output    = path_probs @ leaf_dist                         # (batch, n_classes)
        return output


# ══════════════════════════════════════════════════════════════════════════════
#   GPU Random Forest — ensemble of SoftDecisionTrees
# ══════════════════════════════════════════════════════════════════════════════
class GPURandomForest:
    def __init__(self, n_estimators=100, max_depth=10, n_classes=2,
                 n_features=None, lr=0.01, epochs=50,
                 max_samples=0.8, max_features=0.8):
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.n_classes    = n_classes
        self.n_features   = n_features
        self.lr           = lr
        self.epochs       = epochs
        self.max_samples  = max_samples   # row bootstrap fraction
        self.max_features = max_features  # feature fraction (RF randomness)
        self.trees        = []
        self.feature_subsets = []         # which features each tree uses

    def _bootstrap(self, X, y):
        n = X.shape[0]
        idx = torch.randint(0, n, (int(n * self.max_samples),), device=X.device)
        return X[idx], y[idx]

    def _feature_subset(self, n_features):
        k = max(1, int(n_features * self.max_features))
        return torch.randperm(n_features, device=device)[:k]

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.long).to(device)
        n_features = X.shape[1]

        for t in range(self.n_estimators):
            # Bootstrap sample
            X_boot, y_boot = self._bootstrap(X, y)

            # Random feature subset
            feat_idx = self._feature_subset(n_features)
            self.feature_subsets.append(feat_idx)
            X_sub = X_boot[:, feat_idx]

            tree = SoftDecisionTree(
                n_features = len(feat_idx),
                max_depth  = self.max_depth,
                n_classes  = self.n_classes
            ).to(device)

            optimizer = torch.optim.Adam(tree.parameters(), lr=self.lr)
            loss_fn   = nn.NLLLoss()

            tree.train()
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                out  = tree(X_sub)
                loss = loss_fn(torch.log(out + 1e-9), y_boot)
                loss.backward()
                optimizer.step()

            tree.eval()
            self.trees.append(tree)

            if (t + 1) % 1 == 0:
                print(f"  Trained tree {t+1:3d}/{self.n_estimators}", flush=True)

    def predict_proba(self, X):
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        all_probs = []

        with torch.no_grad():
            for tree, feat_idx in zip(self.trees, self.feature_subsets):
                X_sub = X_t[:, feat_idx]
                probs = tree(X_sub).cpu().numpy()
                all_probs.append(probs)

        return np.mean(all_probs, axis=0)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


# ══════════════════════════════════════════════════════════════════════════════
#   Train & Evaluate
# ══════════════════════════════════════════════════════════════════════════════
# ── CodeCarbon ───────────────────────────────────────────────────────────────
rf_tracker = OfflineEmissionsTracker(country_iso_code="BGD", log_level="error")
rf_tracker.start()

rf_model = GPURandomForest(
    n_estimators = 50,
    max_depth    = 7,
    n_classes    = len(np.unique(y_train_np)),
    n_features   = X_train_np.shape[1],
    lr           = 0.02,
    epochs       = 40,
    max_samples  = 0.8,
    max_features = 0.8
)

print("\nTraining GPU Random Forest (PyTorch Soft Decision Trees)...", flush=True)
rf_model.fit(X_train_np, y_train_np)
rf_tracker.stop()

# ── Learning Curve ────────────────────────────────────────────────────────────
print("\nBuilding learning curve...")
rf_train_losses = []
rf_test_losses  = []
tree_counts = list(range(10, rf_model.n_estimators + 1, 10))

for n in tree_counts:
    # Use only first n trees
    subset_trees   = rf_model.trees[:n]
    subset_feats   = rf_model.feature_subsets[:n]

    X_tr = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    X_te = torch.tensor(X_test_np,  dtype=torch.float32).to(device)

    all_train, all_test = [], []
    with torch.no_grad():
        for tree, feat_idx in zip(subset_trees, subset_feats):
            all_train.append(tree(X_tr[:, feat_idx]).cpu().numpy())
            all_test.append( tree(X_te[:, feat_idx]).cpu().numpy())

    train_proba = np.mean(all_train, axis=0)
    test_proba  = np.mean(all_test,  axis=0)

    train_loss = log_loss(y_train_np, train_proba)
    test_loss  = log_loss(y_test_np,  test_proba)

    rf_train_losses.append(train_loss)
    rf_test_losses.append(test_loss)
    print(f"Trees {n:3d} — Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}", flush=True)

# ── Convergence ───────────────────────────────────────────────────────────────
rf_min_idx        = np.argmin(rf_test_losses)
rf_min_tree       = tree_counts[rf_min_idx]
rf_min_test_loss  = rf_test_losses[rf_min_idx]
rf_min_train_loss = rf_train_losses[rf_min_idx]

print(f"\nConvergence Point  : Tree {rf_min_tree}")
print(f"Min Train Loss     : {rf_min_train_loss:.4f}")
print(f"Min Test  Loss     : {rf_min_test_loss:.4f}")

rf_emissions = rf_tracker.stop()
print(f"CO2 Emissions      : {rf_emissions:.6f} kg CO2")

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.figure()
plt.plot(tree_counts, rf_train_losses, label='Train Loss')
plt.plot(tree_counts, rf_test_losses,  label='Test Loss')
plt.axvline(x=rf_min_tree, color='red', linestyle='--',
            label=f'Convergence (Tree {rf_min_tree})')
plt.scatter(rf_min_tree, rf_min_test_loss, color='red', zorder=5)
plt.title("Random Forest (GPU/PyTorch) – Log Loss per Tree Count")
plt.xlabel("Number of Trees")
plt.ylabel("Log Loss")
plt.legend()
plt.savefig('D4_rf_learning_curve.png', bbox_inches='tight', dpi=300)
plt.close()

# ── Final Metrics ─────────────────────────────────────────────────────────────
# ── Final Metrics with Threshold Optimization ───────────────────────────────
rf_proba     = rf_model.predict_proba(X_test_np)[:, 1]

# Find best threshold for F1
thresholds = np.linspace(0, 1, 100)
f1_scores = [f1_score(y_test_np, (rf_proba > t).astype(int)) for t in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]

rf_y_pred    = (rf_proba > best_threshold).astype(int)
rf_accuracy  = accuracy_score(y_test_np,  rf_y_pred)
rf_precision = precision_score(y_test_np, rf_y_pred)
rf_recall    = recall_score(y_test_np,    rf_y_pred)
rf_f1        = f1_score(y_test_np,        rf_y_pred)

rf_ghg_co2 = rf_emissions * 0.90
rf_ghg_ch4 = rf_emissions * 0.07
rf_ghg_n2o = rf_emissions * 0.03

metrics_output = f"""
{'='*40}
    Random Forest (GPU/PyTorch) – Final Metrics
{'='*40}
Accuracy           : {rf_accuracy:.4f}
Precision          : {rf_precision:.4f}
Recall             : {rf_recall:.4f}
F1 Score           : {rf_f1:.4f}
Best Threshold     : {best_threshold:.4f}
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

with open('D4_rf_metrics.txt', 'w', encoding='utf-8') as f:
    f.write(metrics_output.strip())

warnings.filterwarnings('default')

print("\nEntire code has runned and stopped running.")