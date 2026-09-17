"""
22_conformal_visualization.py
Visualize conformal prediction results: coverage validation and uncertainty distribution.

Input:  results/conformal_results.npz
Output: results/Fig_Conformal.tif
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 16,
    'font.weight': 'bold',
    'axes.linewidth': 2.0,
    'axes.labelweight': 'bold',
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
})

data = np.load(os.path.join(config.RESULTS_DIR, 'conformal_results.npz'))
p_values = data['p_values']
test_probs = data['test_probs']
test_labels = data['test_labels']
q_hat = data['q_hat']

navy = '#08306b'
pale_blue = '#c6dbef'

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Panel a: coverage validation
alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
coverages = []
for a in alphas:
    pred_pos = p_values > a
    pred_neg = ~pred_pos
    cov = np.mean((pred_pos & (test_labels == 1)) | (pred_neg & (test_labels == 0)))
    coverages.append(cov)

axes[0].plot(alphas, coverages, 'o-', color=navy, linewidth=2, markersize=8)
axes[0].plot([min(alphas), max(alphas)], [1 - min(alphas), 1 - max(alphas)],
             '--', color='gray', linewidth=1.5, label='Theoretical')
axes[0].set_xlabel('Significance level (α)')
axes[0].set_ylabel('Empirical coverage')
axes[0].set_title('Coverage validation', fontsize=20, fontweight='bold')
axes[0].legend(fontsize=14, frameon=False)
axes[0].set_ylim(0.6, 1.0)
axes[0].set_xlim(0.03, 0.32)
for spine in ['top', 'right']:
    axes[0].spines[spine].set_visible(False)

# Panel b: uncertainty distribution
uncertainty_score = np.abs(p_values - 0.1)
sorted_idx = np.argsort(uncertainty_score)
top_uncertain = sorted_idx[:50]

axes[1].hist(test_probs, bins=50, color=pale_blue, edgecolor='#6baed6', alpha=0.7,
             label='All compounds')
axes[1].hist(test_probs[top_uncertain], bins=20, color=navy, alpha=0.8,
             label='High-uncertainty compounds')
axes[1].axvline(x=0.5, color='black', linestyle='--', linewidth=1, label='Decision boundary')
axes[1].set_xlabel('Predicted toxicity probability')
axes[1].set_ylabel('Number of compounds')
axes[1].set_title('Uncertainty distribution', fontsize=20, fontweight='bold')
axes[1].legend(fontsize=14, frameon=False)
for spine in ['top', 'right']:
    axes[1].spines[spine].set_visible(False)

plt.tight_layout()
save_path = os.path.join(config.RESULTS_DIR, 'Fig_Conformal.tif')
plt.savefig(save_path, dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
plt.close()
print(f"Saved: {save_path}")