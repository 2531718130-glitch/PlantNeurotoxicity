"""
17_shap_barplot.py
Top 20 SHAP feature importance bar plot with gradient coloring.

Input:  results/shap_feature_importance.csv
Output: results/shap_top20_gradient.tif
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

plt.rcParams.update({
    'font.family': 'Arial',
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'font.size': 14,
    'axes.linewidth': 2.0,
    'xtick.major.width': 2.0,
    'ytick.major.width': 2.0,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
})

imp_df = pd.read_csv(os.path.join(config.RESULTS_DIR, 'shap_feature_importance.csv'))
top20 = imp_df.sort_values('importance', ascending=False).head(20)
top20 = top20.sort_values('importance', ascending=True)  # for barh, lowest at bottom

# Gradient from light blue to dark blue (more important = darker)
n = len(top20)
colors = []
for i in range(n):
    t = i / (n - 1) if n > 1 else 0
    r = int(135 + (30 - 135) * t)
    g = int(206 + (58 - 206) * t)
    b = int(235 + (95 - 235) * t)
    colors.append(f'#{r:02x}{g:02x}{b:02x}')

fig, ax = plt.subplots(figsize=(8, 10))
bars = ax.barh(top20['feature'], top20['importance'], color=colors,
               edgecolor='white', linewidth=0.6)

for bar, val in zip(bars, top20['importance']):
    ax.text(val + 0.0003, bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', ha='left', fontsize=16,
            fontweight='bold', color='#000000')

ax.set_xlabel('Mean |SHAP value|', fontsize=16, fontweight='bold')
ax.set_title('Top 20 Feature Importance (SHAP)', fontsize=18, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', length=0)
ax.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(config.RESULTS_DIR, 'shap_top20_gradient.tif'), dpi=600,
            bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
plt.close()
print(f"Saved: {os.path.join(config.RESULTS_DIR, 'shap_top20_gradient.tif')}")