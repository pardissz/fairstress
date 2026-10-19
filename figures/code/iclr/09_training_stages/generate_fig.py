import sys
sys.path.insert(0, '..')
from pastel import LIGHT, INK, style
import pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
style()
rd = lambda p: pd.read_csv(p, keep_default_na=False, na_values=[])
st = rd('../03_stage_progression/stage_progression_full.csv')
tu = rd('../04_tulu3_vs_tulu31/tulu3_vs_31_full.csv')
sf = rd('../05_tulu_sft_vs_nosafety/tulu_safety_output_summary.csv')
conds = ['directional_pro_majority', 'null', 'directional_pro_minority']
xl = ['pro-majority', 'no pressure', 'pro-minority']
(fig, ax) = plt.subplots(1, 4, figsize=(7.0, 2.1))

def lines(a, df, key, names, labels, cols, title):
    a.axhline(0.5, color='#999', lw=0.6, ls='--')
    for (n, l, c) in zip(names, labels, cols):
        s = df[df[key] == n].set_index('pressure_condition').reindex(conds)['amb_min_chosen_rate']
        a.plot(range(3), s.values, marker='o', ms=3.5, lw=1.8, color=c, label=f'{l} (swing {s.values[2] - s.values[0]:.3f})')
    a.set_xticks(range(3))
    a.set_xticklabels(xl, fontsize=6.5)
    a.set_title(title, fontsize=8.5)
    a.set_ylim(0.28, 0.8)
    a.legend(loc='upper left', fontsize=6.2)
lines(ax[0], st, 'stage', ['SFT', 'DPO', 'final (base instruct)'], ['SFT', 'DPO', 'final'], [INK['sand'], INK['blue'], INK['teal']], '(a) OLMo-3-7B stages')
lines(ax[1], sf, 'model', ['Tulu-3-8B-SFT', 'Tulu-3-8B-SFT-no-safety-data'], ['with safety data', 'without'], [INK['rose'], INK['indigo']], '(b) Tulu-3 SFT, safety data')
lines(ax[2], tu, 'stage', ['Tulu-3-8B', 'Tulu-3.1-8B'], ['Tulu-3 (PPO)', 'Tulu-3.1 (GRPO)'], [INK['blue'], INK['orange']], '(c) Tulu-3 vs. Tulu-3.1')
ax[0].set_ylabel('minority picked on ties')
n = tu[tu.pressure_condition == 'null'].set_index('stage').reindex(['Tulu-3-8B', 'Tulu-3.1-8B'])
w = 0.36
ax[3].bar([-w / 2, 1 - w / 2], n.gt_min_acc, w, color=LIGHT['pink'], edgecolor=INK['rose'], lw=0.6, label='truth favors minority')
ax[3].bar([w / 2, 1 + w / 2], n.gt_maj_acc, w, color=LIGHT['blue'], edgecolor=INK['blue'], lw=0.6, label='truth favors majority')
ax[3].set_xticks([0, 1])
ax[3].set_xticklabels(['Tulu-3', 'Tulu-3.1'])
ax[3].set_ylim(0.8, 1.0)
ax[3].set_title('(d) Accuracy, no pressure', fontsize=8.5)
ax[3].legend(loc='upper right', fontsize=6.2)
fig.tight_layout(w_pad=0.6)
fig.savefig('../../figures/fig_training_stages.pdf', bbox_inches='tight')
