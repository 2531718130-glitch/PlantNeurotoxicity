"""
08_model_comparison_plots.py
Generate ROC/PR/radar/calibration and metric comparison plots from OOF probabilities.

Reads OOF probability files produced by scripts 01-07.
Outputs:
  Fig_ROC_PR.tif
  Fig_Radar.tif
  Fig_Bar.tif
  Fig_Calibration.tif
  Fig_Disagreement.tif
  Fig_MultiMetric.tif
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, matthews_corrcoef
)
from sklearn.calibration import calibration_curve

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 18,
    'font.weight': 'bold',
    'axes.linewidth': 2.5,
    'axes.labelweight': 'bold',
    'axes.labelsize': 28,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.width': 2.5,
    'ytick.major.width': 2.5,
    'xtick.major.size': 8,
    'ytick.major.size': 8,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 20,
    'figure.dpi': 600,
})


def try_load(paths):
    for p in paths:
        full = os.path.join(config.RESULTS_DIR, p)
        if os.path.exists(full):
            return np.load(full)
    return None


model_files = [
    ('Naive Bayes',   ['prob_nb_oof.npy']),
    ('KNN',           ['stacking_prob_knn_oof.npy', 'prob_knn_oof.npy']),
    ('Random Forest', ['stacking_prob_random_forest_oof.npy', 'prob_random_forest_oof.npy']),
    ('XGBoost',       ['stacking_prob_xgboost_oof.npy', 'prob_xgboost_oof.npy']),
    ('SVM',           ['stacking_prob_svm_oof.npy', 'prob_svm_oof.npy']),
    ('Voting',        ['voting_prob_voting_oof.npy']),
    ('Stacking',      ['stacking_prob_stacking_oof.npy']),
    ('GNN',           ['prob_gnn_calibrated.npy']),
]

all_probs = {}
for name, paths in model_files:
    data = try_load(paths)
    if data is not None:
        all_probs[name] = data

label_data = try_load([
    'label_gnn_calibrated.npy', 'stacking_label_oof.npy',
    'label_oof.npy', 'voting_label_oof.npy'
])
y_true = label_data
all_models = list(all_probs.keys())

colors = {
    'GNN': '#3A86FF', 'Naive Bayes': '#FB5607', 'KNN': '#FFBE0B',
    'Random Forest': '#38B000', 'XGBoost': '#FF006E', 'SVM': '#06D6A0',
    'Voting': '#8338EC', 'Stacking': '#E71D36',
}

# ==================== ROC + PR ====================
fig, axes = plt.subplots(1, 2, figsize=(30, 15))
ax = axes[0]
ax.plot([0, 1], [0, 1], 'k--', lw=2.5, alpha=0.35)
for name in all_models:
    fpr, tpr, _ = roc_curve(y_true, all_probs[name])
    lw = 4.5 if name in ['Stacking', 'GNN'] else 3.5
    ax.plot(fpr, tpr, color=colors[name], lw=lw, label=f"{name} ({auc(fpr, tpr):.3f})")
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve (5-CV OOF)')
ax.legend(loc='lower right', frameon=False)
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])

ax = axes[1]
baseline = y_true.sum() / len(y_true)
ax.axhline(baseline, color='gray', lw=2.5, ls='--', alpha=0.4)
for name in all_models:
    prec, rec, _ = precision_recall_curve(y_true, all_probs[name])
    lw = 4.5 if name in ['Stacking', 'GNN'] else 3.5
    ax.plot(rec, prec, color=colors[name], lw=lw,
            label=f"{name} ({average_precision_score(y_true, all_probs[name]):.3f})")
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Precision-Recall (5-CV OOF)')
ax.legend(loc='lower left', frameon=False)
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])

plt.tight_layout()
plt.savefig(os.path.join(config.RESULTS_DIR, 'Fig_ROC_PR.tif'), dpi=600,
            bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
plt.close()

# ==================== Radar ====================
metrics_radar = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1']
N = len(metrics_radar)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
for name in all_models:
    y_pred = (all_probs[name] >= 0.5).astype(int)
    values = [
        roc_auc_score(y_true, all_probs[name]),
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        f1_score(y_true, y_pred, zero_division=0),
    ]
    values += values[:1]
    lw = 4.5 if name in ['Stacking', 'GNN'] else 3.5
    alpha = 0.25 if name in ['Stacking', 'GNN'] else 0.1
    ax.plot(angles, values, color=colors[name], lw=lw, label=name)
    ax.fill(angles, values, color=colors[name], alpha=alpha)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics_radar, fontsize=18, fontweight='bold')
ax.set_ylim(0, 1.0)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(config.RESULTS_DIR, 'Fig_Radar.tif'), dpi=600,
            bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
plt.close()

# ==================== Calibration ====================
fig, ax = plt.subplots(figsize=(9, 8))
ax.plot([0, 1], [0, 1], 'k--', lw=2.5, alpha=0.35, label='Perfectly calibrated')
for name in all_models:
    prob_true, prob_pred = calibration_curve(y_true, all_probs[name], n_bins=10, strategy='uniform')
    lw = 4.5 if name in ['Stacking', 'GNN'] else 3.5
    ax.plot(prob_pred, prob_true, color=colors[name], lw=lw, marker='o', markersize=5, label=name)
ax.set_xlabel('Mean Predicted Probability'); ax.set_ylabel('Fraction of Positives')
ax.set_title('Calibration Curve (5-CV OOF)')
ax.legend(loc='upper left', frameon=False)
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
plt.tight_layout()
plt.savefig(os.path.join(config.RESULTS_DIR, 'Fig_Calibration.tif'), dpi=600,
            bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
plt.close()

# ==================== Disagreement heatmap ====================
disagree = np.zeros((len(all_models), len(all_models)))
for i, mi in enumerate(all_models):
    for j, mj in enumerate(all_models):
        if i != j:
            pred_i = (all_probs[mi] >= 0.5).astype(int)
            pred_j = (all_probs[mj] >= 0.5).astype(int)
            disagree[i, j] = np.mean(pred_i != pred_j)

fig, ax = plt.subplots(figsize=(10, 9))
im = ax.imshow(disagree, cmap='YlOrRd', vmin=0)
ax.set_xticks(np.arange(len(all_models)))
ax.set_yticks(np.arange(len(all_models)))
ax.set_xticklabels(all_models, rotation=45, ha='right', fontweight='bold')
ax.set_yticklabels(all_models, fontweight='bold')
for i in range(len(all_models)):
    for j in range(len(all_models)):
        text_color = "white" if disagree[i, j] > 0.08 else "black"
        ax.text(j, i, f'{disagree[i, j]:.3f}', ha="center", va="center",
                color=text_color, fontweight='bold')
ax.set_title('Prediction Disagreement Rate (5-CV OOF)')
cbar = fig.colorbar(im, ax=ax, shrink=0.75, label='Disagreement Rate')
plt.tight_layout()
plt.savefig(os.path.join(config.RESULTS_DIR, 'Fig_Disagreement.tif'), dpi=600,
            bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
plt.close()

print("All comparison figures generated.")