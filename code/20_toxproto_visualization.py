"""
20_toxproto_visualization.py
ToxProto visualization: t-SNE, toxicity rates, and substructure enrichment forest plots.

Input:  results/dtrl/toxproto_results.npz
Output: results/dtrl/Fig_ToxProto.tif
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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

toxproto_path = os.path.join(config.RESULTS_DIR, 'dtrl', 'toxproto_results.npz')
data = np.load(toxproto_path)
h_tox = data['h_tox']
cluster_labels = data['cluster_labels']
labels = data['labels']

print(f"h_tox: {h_tox.shape}")

# t-SNE (subsample to 5000 for speed)
n_max = 5000
if len(h_tox) > n_max:
    np.random.seed(config.RANDOM_STATE)
    sample_idx = np.random.choice(len(h_tox), n_max, replace=False)
    h_tox_sample = h_tox[sample_idx]
    cluster_sample = cluster_labels[sample_idx]
else:
    h_tox_sample = h_tox
    cluster_sample = cluster_labels

print("Computing t-SNE...")
tsne = TSNE(n_components=2, random_state=config.RANDOM_STATE, perplexity=30, max_iter=1000)
h_tsne = tsne.fit_transform(h_tox_sample)

blues = ['#08306b', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef']


def add_panel_label(ax, label, x=-0.10, y=1.10, fontsize=22):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight='bold', va='bottom', ha='right', color='#000000', zorder=10)


fig, axes = plt.subplots(2, 2, figsize=(18, 15))

# Panel a: t-SNE
ax = axes[0, 0]
for k in range(6):
    mask = cluster_sample == k
    ax.scatter(h_tsne[mask, 0], h_tsne[mask, 1],
               c=blues[k], label=f'Prototype {k}', s=15, alpha=0.6, edgecolors='none')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_title('Toxicity prototype space', fontsize=20, fontweight='bold')
handles, legend_labels = ax.get_legend_handles_labels()
leg_left = ax.legend(handles[:3], legend_labels[:3], loc='upper left',
                     fontsize=13, markerscale=2.5, frameon=False)
leg_right = ax.legend(handles[3:], legend_labels[3:], loc='upper right',
                      fontsize=13, markerscale=2.5, frameon=False)
ax.add_artist(leg_left)

# Panel b: toxicity rates
ax = axes[0, 1]
proto_tox_rates, proto_n_mols = [], []
for k in range(6):
    mask = cluster_labels == k
    n_total = mask.sum()
    n_toxic = labels[mask].sum()
    proto_tox_rates.append(n_toxic / n_total * 100)
    proto_n_mols.append(n_total)

bars = ax.bar(range(6), proto_tox_rates, color=blues, edgecolor='#08306b', linewidth=1.5)
for i, (bar, rate, n) in enumerate(zip(bars, proto_tox_rates, proto_n_mols)):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.5,
            f'{rate:.1f}%\n(n={n})', ha='center', va='bottom', fontsize=13)
ax.set_xlabel('Prototype')
ax.set_ylabel('Neurotoxicity rate (%)')
ax.set_title('Prototype toxicity rates', fontsize=20, fontweight='bold')
ax.set_ylim(0, 80)
ax.axhline(y=labels.mean() * 100, color='#d7301f', linestyle='--', linewidth=2.0,
           label=f'Dataset average ({labels.mean() * 100:.0f}%)')
ax.legend(fontsize=13, frameon=False)

# Panels c, d: forest plots
substructure_data = {
    'Prototype 0': {
        'pyrido[2,3-d]\npyrimidine': (2.17, 6.36e-12, 0.42),
        'cyano': (3.44, 1.86e-63, 0.38),
        'sulfonamide': (1.35, 2.51e-06, 0.18),
        'sulfonyl': (1.38, 2.17e-08, 0.19),
    },
    'Prototype 5': {
        'indole': (2.42, 3.54e-18, 0.35),
        'sulfonamide': (2.06, 9.00e-31, 0.28),
        'sulfonyl': (2.03, 1.55e-35, 0.27),
        'quinoline': (1.69, 1.50e-08, 0.31),
    }
}


def plot_forest(ax, data, title):
    names = list(data.keys())
    ORs = [data[n][0] for n in names]
    pvals = [data[n][1] for n in names]
    ci_errs = [data[n][2] for n in names]
    y_pos = np.arange(len(names))[::-1]

    ax.errorbar(x=ORs, y=y_pos, xerr=ci_errs, fmt='o', color='#08306b',
                ecolor='#6baed6', elinewidth=2.0, capsize=5, markersize=10)
    ax.axvline(x=1, color='black', linestyle='--', linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=15)
    ax.set_xlabel('Odds Ratio')
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_xlim(0, 6.5)
    for i, (or_val, p_val) in enumerate(zip(ORs, pvals)):
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        ax.text(or_val + ci_errs[i] + 0.2, y_pos[i], sig, va='center',
                fontsize=16, fontweight='bold', color='#08306b')
    ax.text(0.98, 0.02, '* p<0.05, ** p<0.01, *** p<0.001\n(Fisher\'s exact test)',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.9))


plot_forest(axes[1, 0], substructure_data['Prototype 0'],
            'Prototype 0: high-neurotoxicity substructures')
plot_forest(axes[1, 1], substructure_data['Prototype 5'],
            'Prototype 5: heterocycle-enriched substructures')

for ax, lab in zip(axes.flatten(), ['a', 'b', 'c', 'd']):
    add_panel_label(ax, lab)

plt.tight_layout()
save_path = os.path.join(config.RESULTS_DIR, 'dtrl', 'Fig_ToxProto.tif')
plt.savefig(save_path, dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
plt.close()
print(f"Saved: {save_path}")