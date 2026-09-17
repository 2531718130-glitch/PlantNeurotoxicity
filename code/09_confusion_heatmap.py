"""
09_confusion_heatmap.py
Combined confusion matrix heatmap: 6 models x 4 outcomes (TP, TN, FP, FN).

Replaces the original 6-panel confusion matrix figure to comply with
the journal requirement of no more than 4 panels per figure.

Input:  OOF probability files from scripts 01-07
Output: results/Fig_Confusion_Heatmap.tif
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 18,
    'font.weight': 'bold',
    'axes.linewidth': 2.5,
    'axes.labelweight': 'bold',
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'figure.dpi': 600,
})


def try_load(paths):
    for p in paths:
        full = os.path.join(config.RESULTS_DIR, p)
        if os.path.exists(full):
            return np.load(full)
    return None


model_files = [
    ('KNN',           ['stacking_prob_knn_oof.npy', 'prob_knn_oof.npy']),
    ('Random Forest', ['stacking_prob_random_forest_oof.npy', 'prob_random_forest_oof.npy']),
    ('XGBoost',       ['stacking_prob_xgboost_oof.npy', 'prob_xgboost_oof.npy']),
    ('SVM',           ['stacking_prob_svm_oof.npy', 'prob_svm_oof.npy']),
    ('Stacking',      ['stacking_prob_stacking_oof.npy']),
    ('GNN',           ['prob_gnn_calibrated.npy']),
]

all_probs = {}
for name, paths in model_files:
    data = try_load(paths)
    if data is not None:
        all_probs[name] = data

y_true = try_load([
    'label_gnn_calibrated.npy', 'stacking_label_oof.npy', 'label_oof.npy'
])
all_models = list(all_probs.keys())

rows = ['TP', 'TN', 'FP', 'FN']
matrix = np.zeros((len(rows), len(all_models)))

for j, name in enumerate(all_models):
    y_pred = (all_probs[name] >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    tn, fp, fn, tp = cm_norm.ravel()
    matrix[0, j] = tp
    matrix[1, j] = tn
    matrix[2, j] = fp
    matrix[3, j] = fn

fig, ax = plt.subplots(figsize=(12, 6.5))
blues_cmap = LinearSegmentedColormap.from_list('custom_blues', ['#f7fbff', '#08306b'], N=256)
im = ax.imshow(matrix, cmap=blues_cmap, vmin=0, vmax=1, aspect='auto')

for i in range(len(rows)):
    for j in range(len(all_models)):
        val = matrix[i, j]
        text_color = 'white' if val > 0.6 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=24, fontweight='bold', color=text_color)

ax.set_xticks(np.arange(len(all_models)))
ax.set_yticks(np.arange(len(rows)))
ax.set_xticklabels(all_models, fontsize=20, fontweight='bold', rotation=30, ha='right')
ax.set_yticklabels(rows, fontsize=24, fontweight='bold')
ax.tick_params(axis='both', length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_title('Confusion Matrix Comparison across Models (5-CV OOF, Row-normalized)',
             fontsize=22, fontweight='bold', pad=15)
cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
cbar.set_label('Proportion', rotation=270, labelpad=30, fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=16)

plt.tight_layout()
plt.savefig(os.path.join(config.RESULTS_DIR, 'Fig_Confusion_Heatmap.tif'), dpi=600,
            bbox_inches='tight', pad_inches=0.06, pil_kwargs={"compression": "tiff_lzw"})
plt.close()
print(f"Saved: {os.path.join(config.RESULTS_DIR, 'Fig_Confusion_Heatmap.tif')}")