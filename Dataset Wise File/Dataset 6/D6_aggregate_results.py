import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os

# ── Configuration (Dataset 6) ───────────────────────────────────────────────
files = {
    "MLP": "D6_mlp_metrics.txt",
    "Random Forest": "D6_rf_metrics.txt",
    "XGBoost": "D6_xgb_metrics.txt",
    "AdaBoost": "D6_ada_metrics.txt",
    "Logistic Regression": "D6_logreg_metrics.txt",
    "KNN": "D6_knn_metrics.txt"
}

def parse_metrics(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    data = {}
    patterns = {
        "Accuracy": r"Accuracy\s+:\s+([\d.]+)",
        "Precision": r"Precision.*:\s+([\d.]+)",
        "Recall": r"Recall.*:\s+([\d.]+)",
        "F1 Score": r"F1 Score.*:\s+([\d.]+)",
        "Convergence": r"(?:Convergence|Best)\s+(?:Estim|Iter|Tree|Epoch|k)\s+:\s+([\d.]+)",
        "CO2": r"CO2 Emissions\s+:\s+([\d.e-]+)"
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match: data[key] = float(match.group(1))
        else: data[key] = None
    return data

# ── Collect Data ─────────────────────────────────────────────────────────────
summary_data = []
for model_name, file_path in files.items():
    metrics = parse_metrics(file_path)
    if metrics:
        summary_data.append({
            "Model": model_name,
            "Accuracy": f"{metrics['Accuracy']*100:.2f}%" if metrics['Accuracy'] is not None else "N/A",
            "Precision": f"{metrics['Precision']*100:.2f}%" if metrics['Precision'] is not None else "N/A",
            "Recall": f"{metrics['Recall']*100:.2f}%" if metrics['Recall'] is not None else "N/A",
            "F1 Score": f"{metrics['F1 Score']*100:.2f}%" if metrics['F1 Score'] is not None else "N/A",
            "F1_raw": metrics['F1 Score'],
            "Convergence": metrics['Convergence'],
            "CO2 (kg)": metrics['CO2']
        })

if not summary_data:
    print("No metric files found. Please run the model scripts first.")
    exit()

df_summary = pd.DataFrame(summary_data)

# ── CES Calculation ─────────────────────────────────────────────────────────
ces_data = []
for item in summary_data:
    f1, co2 = item['F1_raw'], item['CO2 (kg)']
    ces = f1 * np.log1p(1 / (co2 + 1e-12)) if f1 is not None and co2 is not None else 0.0
    ces_data.append({"Model": item['Model'], "Metric": "F1 Score", "Score": f1, "CO2": co2, "CES": ces})

df_ces = pd.DataFrame(ces_data).sort_values(by="CES", ascending=False)

# ── Display & Save ───────────────────────────────────────────────────────────
print("\n" + "="*80 + "\nMODEL COMPARISON TABLE (DATASET 6)\n" + "="*80)
print(df_summary.drop(columns=["F1_raw"]))
print("\n" + "="*80 + "\nCONVERGENCE EFFICIENCY SCORE (CES)\n" + "="*80)
print(df_ces)

df_summary.drop(columns=["F1_raw"]).to_csv("D6_model_comparison.csv", index=False)
df_ces.to_csv("D6_ces_scores.csv", index=False)

with open("D6_final_summary.txt", "w", encoding="utf-8") as f:
    f.write(df_summary.drop(columns=["F1_raw"]).to_string(index=False) + "\n\n" + df_ces.to_string(index=False))

plt.figure(figsize=(10, 6))
plt.bar(df_ces["Model"], df_ces["CES"], color=plt.cm.viridis(np.linspace(0, 1, len(df_ces))))
plt.xticks(rotation=45, ha='right'); plt.title("D6 CES - Higher is Better"); plt.ylabel("Score")
plt.tight_layout(); plt.savefig('D6_ces_comparison_bar_chart.png', dpi=300); plt.close()

print("\nAggregation completed. Results saved to D6_final_summary.txt")
