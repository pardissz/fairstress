from label_map import human_label
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
from pastel import LIGHT, INK, style
style()
CONFIDENT = LIGHT['orange']
DISTRESSED = LIGHT['blue']
loadings = pd.read_csv('pca_loadings.csv')
(fig, axes) = plt.subplots(1, 2, figsize=(15, 7.2))
for (ax, col, title, poles) in zip(axes, ['PC1_loading', 'PC2_loading'], ['Axis 1: confident and positive vs. distressed and negative', 'Axis 2: relational and experiential vs. topical and procedural'], [('Confident / Positive', 'Distressed / Negative'), ('Relational / Experiential', 'Topical / Procedural')]):
    top = loadings.nlargest(15, col).iloc[::-1]
    bot = loadings.nsmallest(15, col)
    combo = pd.concat([bot, top])
    labels = [human_label(c) for c in combo['category']]
    colors = [CONFIDENT if v > 0 else DISTRESSED for v in combo[col]]
    ax.barh(range(len(combo)), combo[col], color=colors, height=0.72)
    ax.set_yticks(range(len(combo)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color='#333', linewidth=0.8)
    ax.set_title(title, fontsize=11.5, pad=10)
    ax.set_xlabel('PCA loading')
    (xmin, xmax) = ax.get_xlim()
    ax.text(xmin * 0.98, -1.6, f'← {poles[1]}', ha='left', fontsize=8.5, color=INK['blue'], style='italic')
    ax.text(xmax * 0.98, -1.6, f'{poles[0]} →', ha='right', fontsize=8.5, color=INK['orange'], style='italic')
fig.tight_layout()
fig.savefig('../../figures/fig_loadings_readable_11models.pdf', dpi=150, bbox_inches='tight')
print('done')
