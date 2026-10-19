import os
import numpy as np
import matplotlib.pyplot as plt
import csv
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, '..', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
DATA_CSV = os.path.join(HERE, '..', 'synthesis_plots', 'cross_method_separability_summary.csv')
METHOD_ORDER = ['CoT (open-weight)', 'CoT / reasoning-mode (closed)', 'Native thinking (closed)', 'NLA text (verbalization)', 'NLA activation']
COND_ORDER = ['GT (both correct)', 'AMB (null only)']
rows = {}
if os.path.exists(DATA_CSV):
    with open(DATA_CSV) as f:
        for r in csv.DictReader(f):
            rows[r['method'], r['condition']] = (float(r['mean']), float(r['std']), int(r['count']))
else:
    rows = {('CoT (open-weight)', 'AMB (null only)'): (0.5464444444444444, 0.01992758742900763, 6), ('CoT (open-weight)', 'GT (both correct)'): (0.5132666666666666, 0.005796857386166715, 6), ('CoT / reasoning-mode (closed)', 'AMB (null only)'): (0.58585, 0.018061201510420066, 6), ('CoT / reasoning-mode (closed)', 'GT (both correct)'): (0.5171666666666667, 0.00181291661878496, 6), ('NLA activation', 'AMB (null only)'): (0.6429656593406593, 0.07344403369109359, 8), ('NLA activation', 'GT (both correct)'): (0.5890865369533848, 0.07286064171520945, 8), ('NLA text (verbalization)', 'AMB (null only)'): (0.5501441254507669, 0.0739467269251549, 30), ('NLA text (verbalization)', 'GT (both correct)'): (0.5278407213940619, 0.006045971750400087, 2), ('Native thinking (closed)', 'AMB (null only)'): (0.5662, 0.00432589104501413, 4), ('Native thinking (closed)', 'GT (both correct)'): (0.5069, 0.009274696760541563, 4)}
COLOR_GT = '#8B99A6'
COLOR_AMB = '#C1554D'
(fig, ax) = plt.subplots(figsize=(11, 6.2))
x = np.arange(len(METHOD_ORDER))
width = 0.38
for (i, cond) in enumerate(COND_ORDER):
    means = [rows[m, cond][0] for m in METHOD_ORDER]
    stds = [rows[m, cond][1] for m in METHOD_ORDER]
    offset = (i - 0.5) * width
    color = COLOR_GT if cond == 'GT (both correct)' else COLOR_AMB
    ax.bar(x + offset, means, width, yerr=stds, capsize=3, color=color, label=cond, edgecolor='none', error_kw={'elinewidth': 1.3, 'ecolor': 'black'})
ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.2, zorder=0)
ax.text(len(METHOD_ORDER) - 0.55, 0.505, 'chance', color='gray', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(METHOD_ORDER, rotation=15, ha='right', fontsize=11)
ax.set_ylabel('2D held-out separability accuracy\n(minority-chosen vs. majority-chosen)', fontsize=11)
ax.set_ylim(0.45, 0.75)
ax.grid(axis='y', color='0.85', linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.legend(loc='upper left', frameon=True, fontsize=11)
ax.set_title('Every reasoning-extraction method under-reports what the internal state carries\n(2D held-out separability, minority-chosen vs. majority-chosen)', fontsize=15, fontweight='bold', pad=14)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '02_cross_method_separability.png'), dpi=200)
fig.savefig(os.path.join(OUT_DIR, '02_cross_method_separability.pdf'))
print('Saved 02_cross_method_separability.{png,pdf}')
