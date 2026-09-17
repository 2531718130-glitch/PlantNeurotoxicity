"""
01_descriptor_distribution.py
Molecular descriptor distribution plots: main figure (4 descriptors)
and supplementary figure (HBD and HBA).

Input:  data/step3_final_v4.csv
Output: results/Fig_Descriptor_Main.tif
        results/Supplementary_Figure_S1_Descriptors.tif
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
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, mannwhitneyu
from rdkit import Chem
from rdkit.Chem import Descriptors

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'font.weight': 'bold',
    'axes.linewidth': 1.25,
    'axes.labelweight': 'bold',
    'axes.labelsize': 12,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
})

C_NONTOXIC_FACE = '#F4D0A4'
C_NONTOXIC_EDGE = '#D4A373'
C_TOXIC_FACE = '#A8D0E6'
C_TOXIC_EDGE = '#5B9BD5'
C_TEXT = '#333333'
C_SPINE = '#999999'
C_GRID = '#E5E5E5'

# Load and compute descriptors
df = pd.read_csv(config.CSV_PATH)


def calc_desc(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return {
            'MolWt': Descriptors.MolWt(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'LogP': Descriptors.MolLogP(mol),
            'MolMR': Descriptors.MolMR(mol),
        }
    except Exception:
        return None


records = []
for _, row in df.iterrows():
    d = calc_desc(row['smiles'])
    if d:
        d['label'] = int(row['label'])
        records.append(d)

df_d = pd.DataFrame(records)
df_neg = df_d[df_d['label'] == 0]
df_pos = df_d[df_d['label'] == 1]


def add_panel_label(ax, label, x=-0.10, y=1.10, fontsize=18):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight='bold', va='bottom', ha='right', color='#000000', zorder=10)


def plot_descriptor(ax, col, title, unit, xlim, bins, is_discrete, df_neg, df_pos):
    d_neg = df_neg[col].dropna().values
    d_pos = df_pos[col].dropna().values

    ax.hist(d_neg, bins=bins, range=xlim, density=True,
            color=C_NONTOXIC_FACE, edgecolor=C_NONTOXIC_EDGE,
            alpha=0.55, linewidth=0.75, zorder=2)
    ax.hist(d_pos, bins=bins, range=xlim, density=True,
            color=C_TOXIC_FACE, edgecolor=C_TOXIC_EDGE,
            alpha=0.55, linewidth=0.75, zorder=3)

    if not is_discrete:
        for data, color, z in [(d_neg, C_NONTOXIC_EDGE, 4), (d_pos, C_TOXIC_EDGE, 5)]:
            if len(data) > 1:
                bw = data.std() * (len(data) ** (-1 / 5)) * 1.06
                if bw <= 0:
                    bw = 0.1
                kde = gaussian_kde(data, bw_method=bw / data.std() if data.std() > 0 else 0.1)
                x_min = max(xlim[0], data.min() - bw)
                x_max = min(xlim[1], data.max() + bw)
                x_grid = np.linspace(x_min, x_max, 500)
                ax.plot(x_grid, kde(x_grid), color=color, lw=1.75, zorder=z)

    xlabel = f"{title} ({unit})" if unit else title
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.set_xlim(xlim)
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, color=C_GRID)
    ax.set_axisbelow(True)

    if len(d_neg) > 0 and len(d_pos) > 0:
        _, p = mannwhitneyu(d_neg, d_pos, alternative='two-sided')
        if p == 0 or p < 1e-10:
            p_text = r'$p < 1.0 \times 10^{-10}$'
        elif p < 0.001:
            mantissa, exp = f'{p:.2e}'.split('e')
            p_text = r'$p = ' + mantissa + r' \times 10^{' + str(int(exp)) + '}$'
        else:
            p_text = f'$p = {p:.3f}$'
        ax.text(0.97, 0.97, p_text, transform=ax.transAxes,
                fontsize=12, ha='right', va='top', color=C_TEXT, zorder=10)


legend_elements = [
    Line2D([0], [0], color=C_NONTOXIC_EDGE, lw=1.75, marker='s',
           markerfacecolor=C_NONTOXIC_FACE, markersize=6, label='Non-toxic'),
    Line2D([0], [0], color=C_TOXIC_EDGE, lw=1.75, marker='s',
           markerfacecolor=C_TOXIC_FACE, markersize=6, label='Toxic'),
]

# Main figure: 4 descriptors (MW, TPSA, LogP, MolMR)
cfg_main = [
    ('MolWt', 'Molecular Weight', 'g/mol', (0, 1000), 80, False),
    ('TPSA', 'Topological Polar Surface Area', 'Å²', (0, 250), 80, False),
    ('LogP', 'LogP', '', (-5, 12.5), 80, False),
    ('MolMR', 'Molar Refractivity', 'Å³', (0, 300), 80, False),
]

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes = axes.flatten()
for idx, (col, title, unit, xlim, bins, is_discrete) in enumerate(cfg_main):
    plot_descriptor(axes[idx], col, title, unit, xlim, bins,
                    is_discrete, df_neg, df_pos)

for ax, lab in zip(axes, ['a', 'b', 'c', 'd']):
    add_panel_label(ax, lab)

fig.legend(handles=legend_elements, loc='lower center',
           ncol=2, frameon=False, fontsize=13,
           bbox_to_anchor=(0.5, -0.01), handlelength=2.5)

fig.suptitle('Key Molecular Descriptor Distributions by Toxicity Class',
             fontsize=15, fontweight='bold', y=0.97, color=C_TEXT)

plt.tight_layout(rect=[0, 0.05, 1, 0.93])
save_main = os.path.join(config.RESULTS_DIR, 'Fig_Descriptor_Main.tif')
plt.savefig(save_main, dpi=600, bbox_inches='tight', pad_inches=0.04,
            pil_kwargs={"compression": "tiff_lzw"})
plt.close()
print(f"Saved main descriptor figure: {save_main}")


# ==================== Supplementary figure: HBD and HBA ====================
cfg_supp = [
    ('NumHDonors', 'Number of H-Bond Donors', '', (-0.5, 6.5), 7, True),
    ('NumHAcceptors', 'Number of H-Bond Acceptors', '', (-0.5, 18), 19, True),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes = axes.flatten()
for idx, (col, title, unit, xlim, bins, is_discrete) in enumerate(cfg_supp):
    plot_descriptor(axes[idx], col, title, unit, xlim, bins,
                    is_discrete, df_neg, df_pos)

for ax, lab in zip(axes, ['a', 'b']):
    add_panel_label(ax, lab, fontsize=14)

fig.legend(handles=legend_elements, loc='lower center',
           ncol=2, frameon=False, fontsize=11,
           bbox_to_anchor=(0.5, -0.02), handlelength=2.0)

fig.suptitle('Supplementary: H-Bond Donor and Acceptor Distributions',
             fontsize=13, fontweight='bold', y=0.98, color=C_TEXT)

plt.tight_layout(rect=[0, 0.06, 1, 0.92])
save_supp = os.path.join(config.RESULTS_DIR, 'Supplementary_Figure_S1_Descriptors.tif')
plt.savefig(save_supp, dpi=600, bbox_inches='tight', pad_inches=0.04,
            pil_kwargs={"compression": "tiff_lzw"})
plt.close()
print(f"Saved supplementary descriptor figure: {save_supp}")

print("\nDescriptor distribution figures complete.")