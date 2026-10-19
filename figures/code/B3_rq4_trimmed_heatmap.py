import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
try:
    import seaborn as sns
    HAVE_SNS = True
except ImportError:
    HAVE_SNS = False
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, '..', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
stage_labels = ['Pretrain', 'SFT', 'DPO-chosen', 'DPO-rejected', 'RLVR']
stage_keys = ['pretrain', 'sft', 'dpo_chosen', 'dpo_rejected', 'rlvr']
DATA = {'transgender people': {'pretrain': (55.6, 763), 'sft': (59.1, 262), 'dpo_chosen': (57.5, 87), 'dpo_rejected': (10.5, 48), 'rlvr': (25.0, 8)}, 'disabled people': {'pretrain': (48.7, 557), 'sft': (46.8, 683), 'dpo_chosen': (53.7, 335), 'dpo_rejected': (31.7, 265), 'rlvr': (13.2, 38)}, 'bisexual people': {'pretrain': (39.6, 803), 'sft': None, 'dpo_chosen': None, 'dpo_rejected': None, 'rlvr': None}, 'Black people': {'pretrain': (37.3, 671), 'sft': (41.0, 644), 'dpo_chosen': (49.7, 469), 'dpo_rejected': (25.4, 390), 'rlvr': (21.8, 46)}, 'lesbian people': {'pretrain': (37.2, 703), 'sft': (44.4, 18), 'dpo_chosen': (26.7, 15), 'dpo_rejected': (0.0, 8), 'rlvr': None}, 'Sikh people': {'pretrain': (36.3, 157), 'sft': (33.6, 101), 'dpo_chosen': (33.4, 33), 'dpo_rejected': (20.0, 20), 'rlvr': None}, 'Atheist people': {'pretrain': (-20.1, 437), 'sft': (10.4, 115), 'dpo_chosen': (-15.0, 20), 'dpo_rejected': (-17.7, 17), 'rlvr': (-28.6, 7)}, 'White people': {'pretrain': (-18.0, 443), 'sft': (-10.7, 429), 'dpo_chosen': (-10.2, 108), 'dpo_rejected': (-18.6, 86), 'rlvr': (-37.5, 16)}, 'cisgender people': {'pretrain': (-17.6, 461), 'sft': None, 'dpo_chosen': None, 'dpo_rejected': None, 'rlvr': None}, 'non-disabled people': {'pretrain': (-17.1, 152), 'sft': None, 'dpo_chosen': None, 'dpo_rejected': None, 'rlvr': None}, 'heterosexual people': {'pretrain': (-6.8, 264), 'sft': None, 'dpo_chosen': None, 'dpo_rejected': None, 'rlvr': None}, 'Christian people': {'pretrain': (-4.4, 495), 'sft': (-0.4, 542), 'dpo_chosen': (6.0, 133), 'dpo_rejected': (0.0, 107), 'rlvr': (0.0, 13)}}
group_order = ['transgender people', 'disabled people', 'bisexual people', 'Black people', 'lesbian people', 'Sikh people', 'Atheist people', 'White people', 'cisgender people', 'non-disabled people', 'heterosexual people', 'Christian people']
net = pd.DataFrame(index=group_order, columns=stage_labels, dtype=float)
n_annot = pd.DataFrame(index=group_order, columns=stage_labels, dtype=object)
for g in group_order:
    for (skey, slabel) in zip(stage_keys, stage_labels):
        cell = DATA[g][skey]
        if cell is None:
            net.loc[g, slabel] = np.nan
            n_annot.loc[g, slabel] = 'n/a'
        else:
            (v, n) = cell
            net.loc[g, slabel] = v
            n_annot.loc[g, slabel] = f'{v:+.1f}\n(n={n})'
(fig, ax) = plt.subplots(figsize=(7, 6.5))
if HAVE_SNS:
    cmap = sns.diverging_palette(10, 220, s=75, l=50, as_cmap=True)
    sns.heatmap(net.astype(float), annot=n_annot, fmt='', cmap=cmap, center=0, vmin=-40, vmax=60, linewidths=1, linecolor='white', cbar_kws={'label': 'Net framing (positive% − negative%)'}, ax=ax)
else:
    cmap = plt.get_cmap('RdBu')
    norm = mcolors.TwoSlopeNorm(vmin=-40, vcenter=0, vmax=60)
    im = ax.imshow(net.astype(float).values, cmap=cmap, norm=norm, aspect='auto')
    ax.set_xticks(range(len(stage_labels)))
    ax.set_xticklabels(stage_labels)
    ax.set_yticks(range(len(group_order)))
    ax.set_yticklabels(group_order)
    for i in range(len(group_order)):
        for j in range(len(stage_labels)):
            ax.text(j, i, n_annot.values[i, j], ha='center', va='center', fontsize=7)
    fig.colorbar(im, ax=ax, label='Net framing (positive% − negative%)')
ax.axhline(6, color='black', linewidth=2.0)
ax.set_title('RQ4 trimmed heatmap: top 6 / bottom 6 groups by pretraining net framing\n(LLM stance classification, real n shown per cell)', fontsize=12)
ax.set_xlabel('')
ax.set_ylabel('')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'B3_rq4_trimmed_heatmap.png'), dpi=200)
fig.savefig(os.path.join(OUT_DIR, 'B3_rq4_trimmed_heatmap.pdf'))
print('Saved B3_rq4_trimmed_heatmap.{png,pdf}')
