import sys
sys.path.insert(0, '.')
from common import CONTRAST_TO_CATEGORY, MODELS
sys.path.insert(0, '..')
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from pastel import LIGHT, INK, style
style()
SRC = '.'
OUT = '.'
CAT_COLORS = {'race': INK['blue'], 'sex_gender': INK['indigo'], 'sexuality': INK['teal'], 'race_x_sex': INK['rose'], 'disability': INK['orange']}
CAT_ORDER = ['race', 'sex_gender', 'sexuality', 'race_x_sex', 'disability']

def make_category_arrows(label, title):
    df = pd.read_parquet(f'{SRC}/group_means_{label}_projected.parquet')
    df['category'] = df['contrast_id'].map(CONTRAST_TO_CATEGORY)
    df = df[df['category'].isin(CAT_ORDER)]
    piv = df.pivot_table(index=['model', 'category'], columns='chosen_group', values=['PC1', 'PC2'], aggfunc='mean')
    piv.columns = ['_'.join(c) for c in piv.columns]
    piv = piv.reset_index()
    piv = piv.dropna(subset=['PC1_majority', 'PC1_minority', 'PC2_majority', 'PC2_minority'])
    ncols = 4
    nrows = math.ceil(len(MODELS) / ncols)
    (fig, axes) = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 3.8 * nrows), sharex=True, sharey=True)
    axes = axes.flatten()
    all_x = pd.concat([piv['PC1_majority'], piv['PC1_minority']])
    all_y = pd.concat([piv['PC2_majority'], piv['PC2_minority']])
    pad_x = (all_x.max() - all_x.min()) * 0.15
    pad_y = (all_y.max() - all_y.min()) * 0.15
    xlim = (all_x.min() - pad_x, all_x.max() + pad_x)
    ylim = (all_y.min() - pad_y, all_y.max() + pad_y)
    for (i, model) in enumerate(MODELS):
        ax = axes[i]
        sub = piv[piv['model'] == model]
        for (_, row) in sub.iterrows():
            cat = row['category']
            color = CAT_COLORS.get(cat, '#888')
            ax.annotate('', xy=(row['PC1_minority'], row['PC2_minority']), xytext=(row['PC1_majority'], row['PC2_majority']), arrowprops=dict(arrowstyle='-|>', color=color, lw=2, mutation_scale=14))
            ax.scatter([row['PC1_majority']], [row['PC2_majority']], color=color, s=22, zorder=5, edgecolor='white', linewidth=0.6)
        ax.axhline(0, color='#ddd', linewidth=0.6, zorder=0)
        ax.axvline(0, color='#ddd', linewidth=0.6, zorder=0)
        ax.set_title(model, fontsize=9.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=7.5)
    for j in range(len(MODELS), len(axes)):
        axes[j].axis('off')
    handles = [plt.Line2D([0], [0], color=CAT_COLORS[c], lw=2.5, label=c.replace('_x_', ' x ').replace('_', ' / ')) for c in CAT_ORDER]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.text(0.5, -0.06, 'Axis 1 (confident and positive to the right)', ha='center', fontsize=10)
    fig.text(0.005, 0.5, 'Axis 2 (relational and experiential upward)', va='center', rotation=90, fontsize=10)
    fig.tight_layout()
    fname = '../../figures/fig_category_arrows_pooled_no_age_religion.pdf'
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(fname, 'done,', len(piv), 'category-model rows')
make_category_arrows('pooled', 'Majority→minority reasoning-style shift, one arrow per demographic category (5 of 7 — age, religion excluded), per model — GT+AMB pooled')
