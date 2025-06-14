import matplotlib.pyplot as plt
import numpy as np

# Data from your results (averaged over 3 runs)
models = [
    "FairOCT SP",
    "FairOCT Minority",
    "Standard DT",
    "FairOCT EO",
    "EG-DT DP",
    "EG-RF DP"
]
sp_disparities = [0.2179, 0.0548, 0.2179, 0.4515, 0.1518, 0.1518]
tpr_disparities = [0.0527, 0.1446, 0.0527, 0.2094, 0.0210, 0.0210]
fpr_disparities = [0.0540, 0.2617, 0.0540, 0.2767, 0.2274, 0.2274]

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.25
index = np.arange(len(models))

# Plot bars for each disparity metric
bars1 = ax.bar(index - bar_width, sp_disparities, bar_width, label='SP Disparity', color='blue')
bars2 = ax.bar(index, tpr_disparities, bar_width, label='TPR Disparity', color='orange')
bars3 = ax.bar(index + bar_width, fpr_disparities, bar_width, label='FPR Disparity', color='green')

# Customize plot
ax.set_xlabel('Models', fontsize=12)
ax.set_ylabel('Disparity Value', fontsize=12)
ax.set_title('Fairness Disparities Across Models', fontsize=14)
ax.set_xticks(index)
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.7)

# Tight layout to prevent label cutoff
plt.tight_layout()

# Save as PDF for LaTeX
plt.savefig('disparity_plot.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close()
