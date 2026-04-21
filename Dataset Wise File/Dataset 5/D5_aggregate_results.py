import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os

# ── Configuration (Dataset 5) ───────────────────────────────────────────────
files = {
    "MLP": "D5_mlp_metrics.txt",
    "Random Forest": "D5_rf_metrics.txt",
    "XGBoost": "D5_xgb_metrics.txt",
    "AdaBoost": "D5_ada_metrics.txt",
    "Logistic Regression": "D5_logreg_metrics.txt",
    "KNN": "D5_knn_metrics.txt"
}

def parse_metrics(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    data = {}
    
    # regex to find labels and values
    patterns = {
        "Accuracy": r"Accuracy\s+:\s+([\d.]+)",
        "Precision": r"Precision\s+:\s+([\d.]+)",
        "Recall": r"Recall\s+:\s+([\d.]+)",
        "F1 Score": r"F1 Score\s+:\s+([\d.]+)",
        "Convergence": r"(?:Convergence|Best)\s+(?:Estim|Iter|Tree|Epoch|k)\s+:\s+([\d.]+)",
        "CO2": r"CO2 Emissions\s+:\s+([\d.e-]+)"
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            data[key] = float(match.group(1))
        else:
            data[key] = None
            
    return data

# ── Collect Data ─────────────────────────────────────────────────────────────
summary_data = []

for model_name, file_path in files.items():
    metrics = parse_metrics(file_path)
    if metrics:
        summary_data.append({
            "Model": model_name,
            "Accuracy": f"{metrics['Accuracy']*100:.3f}%" if metrics['Accuracy'] is not None else "N/A",
            "Precision": f"{metrics['Precision']*100:.3f}%" if metrics['Precision'] is not None else "N/A",
            "Recall": f"{metrics['Recall']*100:.3f}%" if metrics['Recall'] is not None else "N/A",
            "F1 Score": f"{metrics['F1 Score']*100:.3f}%" if metrics['F1 Score'] is not None else "N/A",
            "F1_raw": metrics['F1 Score'],
            "Convergence": metrics['Convergence'],
            "CO2 (kg)": metrics['CO2']
        })

if not summary_data:
    print("No metric files found. Please run the model scripts first.")
    exit()

df_summary = pd.DataFrame(summary_data)

# ── CES Calculation ─────────────────────────────────────────────────────────
def compute_ces(score, co2):
    if co2 is None or co2 == 0 or np.isnan(co2):
        return 0.0
    return score * np.log1p(1 / (co2 + 1e-12))

ces_data = []
for item in summary_data:
    f1 = item['F1_raw']
    co2 = item['CO2 (kg)']
    ces = compute_ces(f1, co2) if f1 is not None else 0.0
    
    ces_data.append({
        "Model": item['Model'],
        "Metric Used": "F1 Score",
        "Score": f1,
        "CO2 (kg)": co2,
        "CES": ces
    })

df_ces = pd.DataFrame(ces_data)
df_ces_sorted = df_ces.sort_values(by="CES", ascending=False)

# ── Display Results ──────────────────────────────────────────────────────────
print("\n" + "="*80)
print("               MODEL COMPARISON TABLE (DATASET 5)")
print("="*80)
print(df_summary.drop(columns=["F1_raw"]))

print("\n" + "="*80)
print("           CONVERGENCE EFFICIENCY SCORE (CES)")
print("="*80)
print(df_ces_sorted)

# ── Save Files ───────────────────────────────────────────────────────────────
df_summary.drop(columns=["F1_raw"]).to_csv("D5_model_comparison.csv", index=False)
df_ces_sorted.to_csv("D5_ces_scores.csv", index=False)

with open("D5_final_summary.txt", "w", encoding="utf-8") as f:
    f.write("="*80 + "\n")
    f.write("               MODEL COMPARISON TABLE (DATASET 5)\n")
    f.write("="*80 + "\n")
    f.write(df_summary.drop(columns=["F1_raw"]).to_string(index=False) + "\n\n")
    f.write("="*80 + "\n")
    f.write("           CONVERGENCE EFFICIENCY SCORE (CES)\n")
    f.write("="*80 + "\n")
    f.write(df_ces_sorted.to_string(index=False) + "\n")

# ── Visualization ────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
colors = plt.cm.plasma(np.linspace(0, 1, len(df_ces_sorted)))
plt.bar(df_ces_sorted["Model"], df_ces_sorted["CES"], color=colors)
plt.xticks(rotation=45, ha='right')
plt.title("D5 Convergence Efficiency Score (CES) - Higher is Better")
plt.ylabel("Efficiency Score")
plt.xlabel("Model")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('D5_ces_comparison_bar_chart.png', dpi=300)

print("\nAggregation process completed. Results saved to D5_final_summary.txt and D5_ces_comparison_bar_chart.png")
