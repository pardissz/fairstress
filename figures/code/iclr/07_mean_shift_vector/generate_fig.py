import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
from pastel import LIGHT, INK, style
style()
CONFIDENT = LIGHT['orange']
DISTRESSED = LIGHT['blue']
summ = pd.read_csv('summary_shift_vectors.csv')
summ_sorted = summ.sort_values('mean_dx', ascending=False).reset_index(drop=True)
(fig, ax) = plt.subplots(figsize=(7, 4.2))
y = np.arange(len(summ_sorted))
h = 0.36
ax.barh(y + h / 2, summ_sorted['mean_dx'], height=h, xerr=summ_sorted['se_dx'], color=CONFIDENT, label='axis 1 (toward confident and positive)', capsize=2, edgecolor='#666', linewidth=0.4, ecolor='#555')
ax.barh(y - h / 2, summ_sorted['mean_dy'], height=h, xerr=summ_sorted['se_dy'], color=DISTRESSED, label='axis 2 (toward relational)', capsize=2, edgecolor='#666', linewidth=0.4, ecolor='#555')
ax.set_yticks(y)
ax.set_yticklabels(summ_sorted['model'], fontsize=9.5)
ax.invert_yaxis()
ax.axvline(0, color='#333', linewidth=0.8)
ax.set_xlabel('Mean shift from majority-chosen to minority-chosen reasoning')
ax.legend(loc='lower right', fontsize=9.5, frameon=False)
fig.tight_layout()
fig.savefig('../../figures/fig_mean_shift_vector_11models.pdf', dpi=150, bbox_inches='tight')
print('done')
