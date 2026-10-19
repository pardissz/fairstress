import os
import csv
import statistics
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, '..', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
ORIG_DATA_CSV = os.path.join(HERE, '..', 'synthesis_plots', 'cross_method_separability_summary.csv')
orig_rows = {}
if os.path.exists(ORIG_DATA_CSV):
    with open(ORIG_DATA_CSV) as f:
        for r in csv.DictReader(f):
            orig_rows[r['method'], r['condition']] = (float(r['mean']), float(r['std']))
else:
    orig_rows = {('CoT (open-weight)', 'AMB (null only)'): (0.5464444444444444, 0.01992758742900763), ('CoT (open-weight)', 'GT (both correct)'): (0.5132666666666666, 0.005796857386166715), ('CoT / reasoning-mode (closed)', 'AMB (null only)'): (0.58585, 0.018061201510420066), ('CoT / reasoning-mode (closed)', 'GT (both correct)'): (0.5171666666666667, 0.00181291661878496), ('NLA activation', 'AMB (null only)'): (0.6429656593406593, 0.07344403369109359), ('NLA activation', 'GT (both correct)'): (0.5890865369533848, 0.07286064171520945), ('NLA text (verbalization)', 'AMB (null only)'): (0.5501441254507669, 0.0739467269251549), ('NLA text (verbalization)', 'GT (both correct)'): (0.5278407213940619, 0.006045971750400087), ('Native thinking (closed)', 'AMB (null only)'): (0.5662, 0.00432589104501413), ('Native thinking (closed)', 'GT (both correct)'): (0.5069, 0.009274696760541563)}
TEC_CSV = os.path.join(HERE, '..', 'firmcore_nla', 'analysis', 'text_embedding_control', 'full_results.csv')
TASK_TO_COND_LABEL = {'GT_bothcorrect': 'GT (both correct)', 'AMB_chosen': 'AMB (null only)'}
COND_TO_METHOD = {'C_text_lasttoken': 'Text embedding (last-token)', 'D_text_meanpool': 'Text embedding (mean-pool)'}
COND_TO_METHOD_ALL = dict(COND_TO_METHOD)
COND_TO_METHOD_ALL['A_raw_activation'] = 'NLA activation'
tec_rows = {}
act_full_rows = {}
if os.path.exists(TEC_CSV):
    agg = defaultdict(list)
    with open(TEC_CSV) as f:
        for r in csv.DictReader(f):
            if r['condition'] not in COND_TO_METHOD_ALL:
                continue
            best = max(float(r['logreg_mean']), float(r['gb_mean']))
            key = (COND_TO_METHOD_ALL[r['condition']], TASK_TO_COND_LABEL[r['task']])
            agg[key].append(best)
    for (key, vals) in agg.items():
        mean_std = (statistics.mean(vals), statistics.pstdev(vals) if len(vals) > 1 else 0.0)
        if key[0] == 'NLA activation':
            act_full_rows[key] = mean_std
        else:
            tec_rows[key] = mean_std
    print('Recomputed real text-embedding-control values (n=8 cells each):')
    for (k, v) in sorted({**tec_rows, **act_full_rows}.items()):
        print(f'  {k}: mean={v[0]:.4f} std={v[1]:.4f}')
else:
    print('WARNING: full_results.csv not found, using hardcoded fallback (still real, not illustrative).')
    tec_rows = {('Text embedding (last-token)', 'GT (both correct)'): (0.5177, 0.0455), ('Text embedding (last-token)', 'AMB (null only)'): (0.6209, 0.06), ('Text embedding (mean-pool)', 'GT (both correct)'): (0.5704, 0.0791), ('Text embedding (mean-pool)', 'AMB (null only)'): (0.6123, 0.0645)}
    act_full_rows = {('NLA activation', 'GT (both correct)'): (0.7274, 0.0985), ('NLA activation', 'AMB (null only)'): (0.7339, 0.0487)}
METHOD_ORDER = ['CoT (open-weight)', 'CoT / reasoning-mode (closed)', 'Native thinking (closed)', 'NLA text (verbalization)', 'Text embedding (last-token)', 'Text embedding (mean-pool)', 'NLA activation']
COND_ORDER = ['GT (both correct)', 'AMB (null only)']
all_rows = dict(orig_rows)
all_rows.update(tec_rows)
all_rows.update(act_full_rows)
COLOR_GT = '#8B99A6'
COLOR_AMB = '#C1554D'
NEW_BAR_METHODS = {'Text embedding (last-token)', 'Text embedding (mean-pool)', 'NLA activation'}
(fig, ax) = plt.subplots(figsize=(13, 6.5))
x = np.arange(len(METHOD_ORDER))
width = 0.38
for (i, cond) in enumerate(COND_ORDER):
    means = [all_rows[m, cond][0] for m in METHOD_ORDER]
    stds = [all_rows[m, cond][1] for m in METHOD_ORDER]
    offset = (i - 0.5) * width
    color = COLOR_GT if cond == 'GT (both correct)' else COLOR_AMB
    bars = ax.bar(x + offset, means, width, yerr=stds, capsize=3, color=color, label=cond, edgecolor='none', error_kw={'elinewidth': 1.3, 'ecolor': 'black'})
for (i, m) in enumerate(METHOD_ORDER):
    if m in NEW_BAR_METHODS:
        for (j, cond) in enumerate(COND_ORDER):
            offset = (j - 0.5) * width
            mean = all_rows[m, cond][0]
            color = COLOR_GT if cond == 'GT (both correct)' else COLOR_AMB
            ax.bar(x[i] + offset, mean, width, color=color, edgecolor='black', linewidth=0.8, hatch='///', zorder=3)
ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.2, zorder=0)
ax.text(len(METHOD_ORDER) - 0.55, 0.505, 'chance', color='gray', fontsize=10)
DISPLAY_LABEL = {'NLA activation': 'NLA activation\n(full 3,840-d)', 'Text embedding (last-token)': 'Text embedding\n(last-token, full-d)', 'Text embedding (mean-pool)': 'Text embedding\n(mean-pool, full-d)'}
xticklabels = [DISPLAY_LABEL.get(m, m) for m in METHOD_ORDER]
ax.set_xticks(x)
ax.set_xticklabels(xticklabels, rotation=20, ha='right', fontsize=10)
ax.set_ylabel('Held-out separability accuracy\n(minority-chosen vs. majority-chosen)', fontsize=11)
ax.set_ylim(0.45, 0.85)
ax.grid(axis='y', color='0.85', linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.legend(loc='upper left', frameon=True, fontsize=11)
ax.set_title('Appendix: separability at matched full dimensionality\n(hatched bars = same text-embedding-control manifest, 2 models x 4 contrasts, all full-d; unhatched bars = broader 14-model 2D-compressed study)', fontsize=12.5, fontweight='bold', pad=14)
fig.text(0.5, -0.02, 'Note: "NLA activation" here (0.727/0.734) is the full 3,840-dim value from the same manifest as the two\ntext-embedding bars -- not the 2D-compressed value (0.589/0.643) shown for this method in the main-text\nFigure 02, and not the 0.762 single-cell (n=201, one contrast) citation used in earlier drafts, which is\nflagged as non-representative in both REPORT_2026-09-14.md (E1) and TEXT_EMBEDDING_CONTROL_REPORT.md.', ha='center', fontsize=8.5, color='#444444')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'C1_fig2_expanded_appendix.png'), dpi=200, bbox_inches='tight')
print('Saved C1_fig2_expanded_appendix.png')
