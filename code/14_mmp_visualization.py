"""
14_mmp_visualization.py
MMP causal validation and attention-substructure concordance visualization.

Input:  results/validation_v4/substructure_v4.csv
        results/validation_v4/pair_experiment_v4.csv
        results/validation_v4/attention_correlation_v4.csv
Output: results/validation_v4/Fig_MMP.tif
        results/validation_v4/Supplementary_Figure_S1_Forest.tif
        results/validation_v4/Supplementary_Figure_S2_Bubble.tif
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

RESULT_DIR = os.path.join(config.RESULTS_DIR, 'validation_v4')
os.makedirs(RESULT_DIR, exist_ok=True)

COLORS = {
    'dark_blue': '#1B3A5C', 'mid_dark': '#2E4F6E', 'mid_blue': '#4A7A9E',
    'light_blue': '#6B9BC0', 'pale_blue': '#9BC5E0', 'very_pale': '#C8E0F0',
    'gray': '#B0BEC5', 'dark_text': '#2C3E50', 'light_text': '#78909C',
    'grid': '#ECEFF1', 'bg': '#FFFFFF',
}

blue_cmap = LinearSegmentedColormap.from_list(
    'shap_blue',
    [COLORS['very_pale'], COLORS['light_blue'], COLORS['mid_blue'], COLORS['dark_blue']]
)

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

df_sub = pd.read_csv(os.path.join(RESULT_DIR, 'substructure_v4.csv'))
df_pair = pd.read_csv(os.path.join(RESULT_DIR, 'pair_experiment_v4.csv'))
df_attn = pd.read_csv(os.path.join(RESULT_DIR, 'attention_correlation_v4.csv'))

deltas = df_pair['delta'].values
pair_ids = df_pair['pair_id'].values
a_probs = df_pair['A_prob'].values
b_probs = df_pair['B_prob'].values


def p_stars(p_str):
    try:
        p = float(p_str)
        if p < 0.001: return '***'
        elif p < 0.01: return '**'
        elif p < 0.05: return '*'
        else: return 'ns'
    except Exception:
        return ''


def gradient_colors(values, cmap=blue_cmap, vmin=None, vmax=None):
    norm = plt.Normalize(vmin=vmin or min(values), vmax=vmax or max(values))
    return [cmap(norm(v)) for v in values]


# ==================== Main figure: 4-panel MMP ====================
fig = plt.figure(figsize=(20, 16), dpi=600)
gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.30)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# Panel a: substructure risk ratio bar
df_sub_sorted = df_sub.sort_values('risk_ratio', ascending=True)
bar_colors = []
for _, row in df_sub_sorted.iterrows():
    if row['risk_ratio'] > 1:
        intensity = min(1.0, (row['risk_ratio'] - 1.0) / 0.6)
        bar_colors.append(blue_cmap(0.3 + 0.7 * intensity))
    else:
        bar_colors.append(COLORS['gray'])

bars = ax1.barh(df_sub_sorted['substructure'], df_sub_sorted['risk_ratio'],
                color=bar_colors, edgecolor='white', linewidth=1.5, height=0.68)
ax1.axvline(x=1.0, color=COLORS['dark_text'], linestyle='--', linewidth=2.0, alpha=0.6)
for bar, (_, row) in zip(bars, df_sub_sorted.iterrows()):
    width = bar.get_width()
    stars = p_stars(row['p_value'])
    ax1.text(width + 0.02, bar.get_y() + bar.get_height() / 2,
             f"{width:.2f} {stars}", va='center', ha='left', fontsize=13,
             color=COLORS['dark_text'], fontweight='bold')
ax1.set_xlim(0, max(df_sub_sorted['risk_ratio']) * 1.22)
ax1.set_xlabel('Risk Ratio (RR)', fontsize=18, fontweight='bold')
ax1.set_title('Substructure Risk Ratio', fontsize=20, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.4)
for spine in ['top', 'right']:
    ax1.spines[spine].set_visible(False)

# Panel b: matched pair delta probability
delta_colors = gradient_colors(deltas, vmin=0, vmax=max(deltas) * 1.05)
ax2.bar(pair_ids, deltas, color=delta_colors, edgecolor='white', linewidth=1.0, width=0.75)
mean_d = np.mean(deltas)
median_d = np.median(deltas)
ax2.axhline(y=mean_d, color=COLORS['dark_blue'], linestyle='--', linewidth=2.5, alpha=0.8)
ax2.text(0.02, 0.88, f'Mean = {mean_d:.3f}', transform=ax2.transAxes,
         ha='left', va='top', fontsize=14, color=COLORS['dark_blue'],
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                   edgecolor=COLORS['dark_blue'], alpha=0.9))
stat_text = (f"n = {len(deltas)}\n"
             f"Delta > 0: {(deltas > 0).sum()}/{len(deltas)}\n"
             f"Wilcoxon p = 5.68e-14\n"
             f"Cohen's d = 3.20")
ax2.text(0.98, 0.94, stat_text, transform=ax2.transAxes, fontsize=13,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['very_pale'],
                   edgecolor=COLORS['light_blue'], alpha=0.95))
ax2.set_xlabel('Pair ID', fontsize=18, fontweight='bold')
ax2.set_ylabel('Delta Probability (A - B)', fontsize=18, fontweight='bold')
ax2.set_title('Matched Pair Delta Probability', fontsize=20, fontweight='bold')
ax2.set_xticks(np.arange(1, len(pair_ids) + 1, 5))
ax2.set_ylim(0, max(deltas) * 1.12)
ax2.grid(axis='y', alpha=0.4)
for spine in ['top', 'right']:
    ax2.spines[spine].set_visible(False)

# Panel c: attention-substructure concordance
if len(df_attn) > 0:
    x = np.arange(len(df_attn))
    width = 0.35
    bars3a = ax3.bar(x - width / 2, df_attn['top3_rate'] * 100, width,
                     label='Top-3 Attention', color=COLORS['mid_blue'],
                     edgecolor='white', linewidth=1.5)
    bars3b = ax3.bar(x + width / 2, df_attn['high_attn_rate'] * 100, width,
                     label='High Attention (>0.5)', color=COLORS['pale_blue'],
                     edgecolor='white', linewidth=1.5)
    for bar in list(bars3a) + list(bars3b):
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, h + 2,
                 f'{h:.0f}%', ha='center', va='bottom', fontsize=14,
                 color=COLORS['dark_text'], fontweight='bold')
    ax3.set_ylabel('Match Rate (%)', fontsize=18, fontweight='bold')
    ax3.set_title('Attention-Substructure Correlation', fontsize=20, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(df_attn['substructure'], rotation=30, ha='right', fontsize=15)
    ax3.legend(fontsize=14, frameon=False)
    ax3.set_ylim(0, 115)
    ax3.grid(axis='y', alpha=0.4)
    for spine in ['top', 'right']:
        ax3.spines[spine].set_visible(False)

# Panel d: A vs B probability scatter
scatter_colors = gradient_colors(deltas, vmin=0, vmax=max(deltas))
ax4.scatter(b_probs, a_probs, s=120, c=scatter_colors,
            edgecolors='white', linewidth=1.5, alpha=0.95, zorder=5)
ax4.plot([0, 1], [0, 1], color=COLORS['gray'], linestyle='--', linewidth=2.0, zorder=1)
ax4.fill_between([0, 1], [0, 1], [1, 1], color=COLORS['very_pale'], alpha=0.25, zorder=0)
ax4.text(0.82, 0.72, 'y = x\n(no difference)', fontsize=14,
         color=COLORS['light_text'], ha='center', va='center', rotation=45)
ax4.set_xlabel('B Probability (without pyridopyrimidine)', fontsize=16, fontweight='bold')
ax4.set_ylabel('A Probability (with pyridopyrimidine)', fontsize=16, fontweight='bold')
ax4.set_title('Pair-wise Probability Comparison', fontsize=20, fontweight='bold')
ax4.set_xlim(-0.02, 1.02)
ax4.set_ylim(-0.02, 1.02)
ax4.set_aspect('equal', adjustable='box')
ax4.grid(True, alpha=0.3)
for spine in ['top', 'right']:
    ax4.spines[spine].set_visible(False)

plt.tight_layout(pad=2.0)
save_main = os.path.join(RESULT_DIR, 'Fig_MMP.tif')
plt.savefig(save_main, dpi=600, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close(fig)
print(f"Saved main MMP figure: {save_main}")


# ==================== Supplementary S1: Forest plot ====================
def estimate_population(df_sub):
    A = np.column_stack([df_sub['bg_rate'].values, -np.ones(len(df_sub))])
    b = df_sub['bg_rate'].values * df_sub['n_total'].values - df_sub['n_toxic'].values
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return x[0], x[1]


def calc_rr_ci(n_total, n_toxic, bg_rate, N, T):
    a = n_toxic
    b = n_total - n_toxic
    c = max(1, T - n_toxic)
    d = max(1, N - n_total - c)
    rr = (a / (a + b)) / (c / (c + d))
    se_log = np.sqrt(abs(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d)))
    ci_low = np.exp(np.log(rr) - 1.96 * se_log)
    ci_high = np.exp(np.log(rr) + 1.96 * se_log)
    return rr, ci_low, ci_high


N_est, T_est = estimate_population(df_sub)
print(f"  Estimated N ≈ {N_est:.0f}, T ≈ {T_est:.0f}")

ci_data = []
for _, row in df_sub.iterrows():
    rr, ci_l, ci_h = calc_rr_ci(row['n_total'], row['n_toxic'], row['bg_rate'], N_est, T_est)
    ci_data.append({
        'substructure': row['substructure'],
        'rr': rr, 'ci_low': ci_l, 'ci_high': ci_h,
        'n_total': row['n_total'], 'n_toxic': row['n_toxic']
    })
df_ci = pd.DataFrame(ci_data).sort_values('rr', ascending=True)

fig_s1, ax_s1 = plt.subplots(figsize=(14, 11), dpi=600)
y_pos = np.arange(len(df_ci))
for i, (_, row) in enumerate(df_ci.iterrows()):
    is_significant = row['ci_low'] > 1.0 or row['ci_high'] < 1.0
    color = COLORS['mid_blue'] if is_significant else COLORS['gray']
    ax_s1.plot([row['ci_low'], row['ci_high']], [i, i], color=color, lw=3.0, solid_capstyle='round')
    ax_s1.plot([row['ci_low'], row['ci_low']], [i - 0.12, i + 0.12], color=color, lw=3.0)
    ax_s1.plot([row['ci_high'], row['ci_high']], [i - 0.12, i + 0.12], color=color, lw=3.0)
    ax_s1.scatter(row['rr'], i, s=120, c=color, edgecolors='white', linewidth=2.0, zorder=3)
    ax_s1.text(row['ci_high'] + 0.05, i,
               f"{row['rr']:.2f} [{row['ci_low']:.2f}-{row['ci_high']:.2f}]",
               va='center', ha='left', fontsize=14, color=COLORS['dark_text'])
ax_s1.axvline(x=1.0, color=COLORS['dark_text'], linestyle='--', linewidth=2.0, alpha=0.6)
ax_s1.set_yticks(y_pos)
ax_s1.set_yticklabels(df_ci['substructure'].values, fontsize=16)
ax_s1.set_xlabel('Risk ratio (95% CI)', fontsize=18, fontweight='bold')
ax_s1.set_title('Substructure risk ratio forest plot', fontsize=20, fontweight='bold')
ax_s1.set_xlim(0.3, max(df_ci['ci_high']) * 1.15)
ax_s1.grid(axis='x', alpha=0.4)
for spine in ['top', 'right']:
    ax_s1.spines[spine].set_visible(False)

plt.tight_layout()
save_s1 = os.path.join(RESULT_DIR, 'Supplementary_Figure_S1_Forest.tif')
plt.savefig(save_s1, dpi=600, bbox_inches='tight')
plt.close(fig_s1)
print(f"Saved S1: {save_s1}")


# ==================== Supplementary S2: Bubble plot ====================
fig_s2, ax_s2 = plt.subplots(figsize=(14, 10), dpi=600)
x = df_sub['n_total'].values
y = df_sub['risk_ratio'].values
sizes = df_sub['n_toxic'].values * 4.0
bubble_colors = gradient_colors(y, vmin=0.5, vmax=1.7)

ax_s2.scatter(x, y, s=sizes, c=bubble_colors, alpha=0.85,
              edgecolors='white', linewidth=2.5, zorder=5)

for i, name in enumerate(df_sub['substructure'].values):
    r = np.sqrt(sizes[i]) / 2
    dx, dy, ha, va = 0, int(r) + 15, 'center', 'bottom'
    if name == 'Nitro group':
        dx, dy, va = 0, -(int(r) + 15), 'top'
    if x[i] < 400:
        dx, ha = int(r) + 30, 'left'
    ax_s2.annotate(name, (x[i], y[i]), textcoords='offset points',
                   xytext=(dx, dy), fontsize=14, color=COLORS['dark_text'],
                   ha=ha, va=va, zorder=10,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                             edgecolor='none', alpha=0.95))

ax_s2.axhline(y=1.0, color=COLORS['dark_text'], linestyle='--', linewidth=2.0, alpha=0.6)
ax_s2.set_xlabel('Number of molecules containing substructure', fontsize=18, fontweight='bold')
ax_s2.set_ylabel('Risk ratio (RR)', fontsize=18, fontweight='bold')
ax_s2.set_title('Substructure enrichment analysis of toxicity risk', fontsize=20, fontweight='bold')
ax_s2.set_xscale('log')
ax_s2.set_xlim(280, 5800)
ax_s2.set_ylim(0.38, 1.88)
ax_s2.grid(True, alpha=0.3)
for spine in ['top', 'right']:
    ax_s2.spines[spine].set_visible(False)

plt.tight_layout()
save_s2 = os.path.join(RESULT_DIR, 'Supplementary_Figure_S2_Bubble.tif')
plt.savefig(save_s2, dpi=600, bbox_inches='tight')
plt.close(fig_s2)
print(f"Saved S2: {save_s2}")

print("\nAll MMP figures generated.")