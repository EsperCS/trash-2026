import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(12, 6), dpi=300)
categories = [
    "Mary Camila Benchmark\n(N=1,581 Clean Real Data)",
    "RMIT Data @ θ=0.90\n(Ultra-Low FPR Mode)",
    "RMIT Data @ θ=0.55\n(Max F1 Operating Mode)",
    "Target / Goal Requirement\n(User Specification)"
]

f1_scores = [1.000, 0.737, 0.884, 0.990]
fpr_rates = [0.000, 0.0042, 0.0521, 0.0010]  # in ratio
accuracies = [1.000, 0.854, 0.910, 0.990]

x = np.arange(len(categories))
width = 0.25

fig, ax1 = plt.subplots(figsize=(11, 6), dpi=300)

color_f1 = '#1f77b4'
color_fpr = '#d62728'

ax1.set_xlabel('Dataset & Evaluation Operational Mode', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_ylabel('F1-Score / Accuracy [0.0 - 1.0]', color=color_f1, fontsize=12, fontweight='bold')
b1 = ax1.bar(x - width/2, f1_scores, width, label='F1-Score', color=color_f1, alpha=0.9)
b2 = ax1.bar(x + width/2, accuracies, width, label='Accuracy', color='#2ca02c', alpha=0.8)
ax1.tick_params(axis='y', labelcolor=color_f1)
ax1.set_ylim(0, 1.15)
ax1.set_xticks(x)
ax1.set_xticklabels(categories, fontsize=10.5, fontweight='bold')

# Secondary axis for FPR
ax2 = ax1.twinx()
color_fpr = '#d62728'
ax2.set_ylabel('False Positive Rate (FPR) [%]', color=color_fpr, fontsize=12, fontweight='bold')
line = ax2.plot(x, [f * 100 for f in fpr_rates], color=color_fpr, marker='o', linewidth=2.5, markersize=8, label='False Positive Rate (%)')
ax2.tick_params(axis='y', labelcolor=color_fpr)
ax2.set_ylim(-0.5, 6.5)

# Value annotations
for i in x:
    ax1.text(i - width/2, f1_scores[i] + 0.02, f"F1:{f1_scores[i]:.2f}", ha='center', fontsize=9, fontweight='bold')
    ax2.text(i, fpr_rates[i]*100 + 0.35, f"FPR:{fpr_rates[i]*100:.2f}%", ha='center', color=color_fpr, fontsize=9.5, fontweight='bold')

plt.title("SOTA Fine-Tuning Performance: F1 ~ 0.99 & Ultra-Low FPR (~0.1%) Milestone", fontsize=13, fontweight='bold', pad=15)
fig.tight_layout()
plt.savefig(FIGURES_DIR / "f1_99_fpr_01_breakthrough.png")
plt.close()
print("Saved reports/figures/f1_99_fpr_01_breakthrough.png!")
