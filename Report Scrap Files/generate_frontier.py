import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data from Table 1
data = [
    ("Ada", 0.9600, 1.27e-5), ("XGB", 0.9600, 3.79e-6), ("RF", 0.9495, 2.32e-4), ("MLP", 0.9600, 1.23e-5), ("LR", 0.9600, 3.66e-5), ("KNN", 0.9600, 3.01e-6),
    ("Ada", 0.8966, 6.48e-6), ("XGB", 0.8911, 4.32e-6), ("RF", 0.9055, 2.29e-4), ("MLP", 0.9029, 5.08e-6), ("LR", 0.9208, 1.78e-5), ("KNN", 0.9360, 3.02e-6),
    ("Ada", 0.9032, 9.39e-6), ("XGB", 0.9091, 1.37e-5), ("RF", 0.8667, 8.35e-4), ("MLP", 0.8966, 6.73e-5), ("LR", 0.8966, 2.59e-5), ("KNN", 0.8966, 2.94e-6),
    ("Ada", 0.7763, 2.05e-5), ("XGB", 0.9030, 4.25e-6), ("RF", 0.8004, 5.91e-5), ("MLP", 0.8185, 4.47e-5), ("LR", 0.4930, 9.74e-5), ("KNN", 0.7009, 3.38e-6),
    ("Ada", 0.6977, 2.28e-5), ("XGB", 0.6875, 4.13e-6), ("RF", 0.6667, 2.82e-4), ("MLP", 0.7050, 9.66e-6), ("LR", 0.6809, 3.26e-5), ("KNN", 0.6880, 3.22e-6),
    ("Ada", 0.9011, 1.85e-5), ("XGB", 0.9214, 4.65e-6), ("RF", 0.5053, 6.21e-5), ("MLP", 0.9106, 4.50e-6), ("LR", 0.7256, 1.76e-5), ("KNN", 0.9224, 3.39e-6)
]

df = pd.DataFrame(data, columns=["Model", "F1", "CO2"])

colors = {"Ada": "purple", "XGB": "green", "RF": "red", "MLP": "blue", "LR": "orange", "KNN": "cyan"}

plt.figure(figsize=(4, 2.5)) # Compact size to save space

for model in colors.keys():
    subset = df[df["Model"] == model]
    plt.scatter(subset["CO2"], subset["F1"], label=model, color=colors[model], s=20, alpha=0.7)

sorted_df = df.sort_values(by=["CO2", "F1"], ascending=[True, False]).reset_index(drop=True)
pareto_front_x = []
pareto_front_y = []

max_y_so_far = -1
for index, row in sorted_df.iterrows():
    if row["F1"] > max_y_so_far:
        pareto_front_x.append(row["CO2"])
        pareto_front_y.append(row["F1"])
        max_y_so_far = row["F1"]

plt.plot(pareto_front_x, pareto_front_y, 'k--', label="Efficient Frontier", linewidth=1.0)
plt.xscale('log')
plt.xlabel("CO2 Emissions (kg, log)", fontsize=8)
plt.ylabel("F1-score", fontsize=8)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.legend(fontsize=6, loc='lower right', framealpha=0.8)
plt.grid(True, which="both", ls="--", alpha=0.4, linewidth=0.5)
plt.tight_layout(pad=0.2)
plt.savefig("efficient_frontier_plot.png", dpi=300)
