import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from codecarbon import OfflineEmissionsTracker
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ── 1. Advanced Data Loading & Cleaning ──────────────────────────────────────
url = 'https://raw.githubusercontent.com/mazidzomader/CSE427-Project-BRACU/refs/heads/main/Datasets/Asthma.csv'
df = pd.read_csv(url)

# Preprocessing
df = df.drop(columns=['PatientID', 'DoctorInCharge'], errors='ignore')
X = df.drop(columns=['Diagnosis'])
y = df['Diagnosis']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
n_classes = len(np.unique(y_encoded))

# Split
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

# ── 2. Scaling & SMOTE ───────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train).astype(np.float32)
X_val_s   = scaler.transform(X_val).astype(np.float32)
X_test_s  = scaler.transform(X_test).astype(np.float32)

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
print(f"Random Forest runs on: {str(device).upper()} (Cleaned)")
print(f"{'='*40}")


# ── 4. Soft Decision Tree (GPU-compatible) ────────────────────────────────────
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
    def __init__(self, n_estimators=50, max_depth=4, n_classes=4, lr=0.01, epochs=20):
        self.n_estimators, self.max_depth, self.n_classes, self.lr, self.epochs = n_estimators, max_depth, n_classes, lr, epochs
        self.trees, self.feature_subsets = [], []

    def fit(self, X, y):
        X_t, y_t = torch.tensor(X).to(device), torch.tensor(y, dtype=torch.long).to(device)
        n_features = X.shape[1]
        for t in range(self.n_estimators):
            idx = torch.randint(0, X.shape[0], (X.shape[0],), device=device)
            feat_idx = torch.randperm(n_features, device=device)[:int(n_features * 0.8)]
            self.feature_subsets.append(feat_idx)
            tree = SoftDecisionTree(len(feat_idx), self.max_depth, self.n_classes).to(device)
            optimizer = torch.optim.Adam(tree.parameters(), lr=self.lr)
            tree.train()
            for _ in range(self.epochs):
                optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(tree(X_t[idx][:, feat_idx]), y_t[idx])
                loss.backward(); optimizer.step()
            self.trees.append(tree.eval())

    def predict_proba(self, X):
        X_t = torch.tensor(X).to(device)
        all_probs = []
        with torch.no_grad():
            for tree, feat_idx in zip(self.trees, self.feature_subsets):
                all_probs.append(tree(X_t[:, feat_idx]).cpu().numpy())
        return np.mean(all_probs, axis=0)

# ── 5. Training ──────────────────────────────────────────────────────────────
tracker = OfflineEmissionsTracker(country_iso_code="BGD", log_level="error")
tracker.start()

print("\nTraining Cleaned GPU Random Forest...")
rf_model = GPURandomForest(n_estimators=50, max_depth=4, n_classes=n_classes)
rf_model.fit(X_train_res, y_train_res)

te_losses = []
tree_counts = list(range(10, rf_model.n_estimators + 1, 10))
for n in tree_counts:
    probs = []
    with torch.no_grad():
        for tree, f_idx in zip(rf_model.trees[:n], rf_model.feature_subsets[:n]):
            probs.append(tree(torch.tensor(X_test_s).to(device)[:, f_idx]).cpu().numpy())
    te_losses.append(log_loss(y_test, np.mean(probs, axis=0)))

rf_min_tree = tree_counts[np.argmin(te_losses)]
emissions = tracker.stop()

# ── 6. Metrics & Output ──────────────────────────────────────────────────────
probs = rf_model.predict_proba(X_test_s)
y_pred = np.argmax(probs, axis=1)
acc, pr, rc, f1 = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='weighted'), recall_score(y_test, y_pred, average='weighted'), f1_score(y_test, y_pred, average='weighted')

metrics_output = f"""
{'='*40}
    Random Forest (Cleaned D6) – Final Metrics
{'='*40}
Accuracy           : {acc:.4f}
Precision (wt)     : {pr:.4f}
Recall (wt)        : {rc:.4f}
F1 Score (wt)      : {f1:.4f}
Convergence Tree   : {rf_min_tree}
CO2 Emissions      : {emissions:.6f} kg CO2
{'='*40}
"""
print(metrics_output)
with open('D6_rf_metrics.txt', 'w', encoding='utf-8') as f: f.write(metrics_output.strip())

plt.figure(); plt.plot(tree_counts, te_losses, label='Test Loss'); plt.axvline(x=rf_min_tree, color='red', linestyle='--')
plt.title("Random Forest Learning Curve (Cleaned D6)"); plt.savefig('D6_rf_learning_curve.png'); plt.close()

print("\nEntire code has runned and stopped running.")
