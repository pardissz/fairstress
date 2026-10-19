import sys, csv
sys.path.insert(0, '..')
from pastel import LIGHT, INK, style
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
style()

def load_population(path, metric, population):
    (names, rows) = ([], [])
    with open(path) as f:
        for r in csv.DictReader(f):
            if r['metric'] == metric and r['population'] == population:
                names.append(r['model'])
                rows.append([float(r['D0']), float(r['D1']), float(r['D2']), float(r['D3'])])
    return (names, rows)
(names, acc) = load_population('per_model_data.csv', 'AccGap', 'full-corpus-18')
(_, lean) = load_population('per_model_data.csv', 'TieLean', 'full-corpus-18')
(_, f_acc_raw) = load_population('per_model_data.csv', 'AccGap', 'frontier-6')
(_, f_lean_raw) = load_population('per_model_data.csv', 'TieLean', 'frontier-6')
f_acc = np.array(f_acc_raw).mean(0)
f_lean = np.array(f_lean_raw).mean(0)
frag = {'Type 1: minority vs. majority': [-0.81, 1.55, 7.04, 8.74], 'Type 2: minority vs. minority': [-0.2, -0.6, -2.9, -2.3]}
tieshift = {'Type 1: minority vs. majority': [22.58, 20.26, 21.46, 16.03], 'Type 2: minority vs. minority': [24.11, 23.78, 24.37, 19.31]}
x = np.arange(4)
xl = ['D0\nnone', 'D1\nname', 'D2\ncorrelate', 'D3\nlabel']
(fig, ax) = plt.subplots(1, 4, figsize=(9.2, 2.12), gridspec_kw=dict(width_ratios=[1.45, 1, 1, 1.45]))

def endlabels(a, rows):
    (lo, hi) = a.get_ylim()
    span = hi - lo
    ys = [r[-1] for r in rows]
    order = np.argsort(ys)
    gap = span * 0.056
    pos = {}
    prev = lo - 1000000000.0
    for i in order:
        y = max(ys[i], prev + gap)
        pos[i] = y
        prev = y
    over = prev - (hi - 0.01 * span)
    if over > 0:
        for i in order:
            pos[i] -= over
        prev = hi + 1000000000.0
        for i in order[::-1]:
            pos[i] = min(pos[i], prev - gap)
            prev = pos[i]
    for i in order:
        a.annotate(names[i], xy=(3, ys[i]), xytext=(3.32, pos[i]), fontsize=5.3, color='#4A6F96', va='center', ha='left', annotation_clip=False, arrowprops=dict(arrowstyle='-', color=LIGHT['blue'], lw=0.4, shrinkA=0, shrinkB=0))
    a.set_xlim(-0.2, 3.1)

def ladder(a, rows, fr, fair, ylab, title):
    a.axhspan(fair - 0.0001, fair + 0.0001, color='#999', lw=0.6)
    for r in rows:
        a.plot(x, r, color=LIGHT['blue'], lw=0.7, alpha=0.55)
    a.plot(x, np.mean(rows, 0), color=INK['blue'], lw=2.2, marker='o', ms=4, label='full-corpus mean (18)')
    a.plot(x, fr, color=INK['rose'], lw=1.8, ls='--', marker='s', ms=3.5, label='frontier mean (6)')
    a.set_xticks(x)
    a.set_xticklabels(xl)
    a.set_ylabel(ylab)
    a.set_title(title)
ladder(ax[0], acc, f_acc, 0, 'points (fair = 0)', '(a) Accuracy Gap')
ax[0].set_ylim(-3.5, 19)
ladder(ax[3], lean, f_lean, 50, '% minority picked (fair = 50)', '(d) Tie-Break Lean')
ax[3].set_ylim(48, 76)
ax[0].legend(loc='upper left', handlelength=1.4, fontsize=6.6)
cols = [INK['orange'], INK['teal']]
mk = ['o', '^']
ax[1].axhline(0, color='#999', lw=0.6)
for ((k, v), c, m) in zip(frag.items(), cols, mk):
    ax[1].plot(x, v, color=c, lw=2, marker=m, ms=4, label=k)
ax[1].set_xticks(x)
ax[1].set_xticklabels(['D0', 'D1', 'D2', 'D3'])
ax[1].set_ylabel('points (fair = 0)')
ax[1].set_title('(b) Fragility Gap by contrast type')
ax[1].set_ylim(-5.5, 16.5)
ax[1].legend(loc='upper left', fontsize=6.6, handlelength=1.6)
for ((k, v), c, m) in zip(tieshift.items(), cols, mk):
    ax[2].plot(x, v, color=c, lw=2, marker=m, ms=4, label=k)
ax[2].set_xticks(x)
ax[2].set_xticklabels(['D0', 'D1', 'D2', 'D3'])
ax[2].set_ylabel('points (fair = 0)')
ax[2].set_title('(c) Tie-Shift by contrast type')
ax[2].set_ylim(0, 27)
ax[2].legend(loc='lower left', fontsize=6.6, handlelength=1.6)
endlabels(ax[0], acc)
endlabels(ax[3], lean)
fig.tight_layout(w_pad=0.4)
fig.subplots_adjust(wspace=0.95)
fig.savefig('../../figures/fig_rq1_ladder.pdf', bbox_inches='tight')
