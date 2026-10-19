import os
import numpy as np
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, '..', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
degrees = ['D0', 'D1', 'D2', 'D3']
x = np.arange(len(degrees))
acc_gap_open = [-0.09, 0.92, 3.99, 3.47]
acc_gap_frontier = [-1.2, 0.7, 0.0, 1.1]
tie_lean_open = [51.5, 53.6, 59.1, 63.7]
tie_lean_frontier = [56.7, 61.8, 64.2, 66.3]
COLOR_OPEN = '#3D4F5C'
COLOR_FRONTIER = '#A6483C'
(fig, axes) = plt.subplots(1, 2, figsize=(7, 3.5))
(axA, axB) = axes
axA.plot(x, acc_gap_open, marker='o', markersize=7, linewidth=2.0, color=COLOR_OPEN, label='Open-weight (n=18)')
axA.plot(x, acc_gap_frontier, marker='o', markersize=7, linewidth=2.0, color=COLOR_FRONTIER, label='Frontier (n=6)')
axA.axhline(0, color='gray', linestyle='--', linewidth=1.0, zorder=0)
axA.set_xticks(x)
axA.set_xticklabels(degrees, fontsize=10)
axA.set_ylabel('Accuracy Gap (pts)', fontsize=10)
axA.set_title('Panel A: Accuracy Gap', fontsize=11)
axA.grid(axis='y', color='0.9', linewidth=0.7, zorder=0)
axA.set_axisbelow(True)
axA.tick_params(labelsize=9)
axB.plot(x, tie_lean_open, marker='o', markersize=7, linewidth=2.0, color=COLOR_OPEN, label='Open-weight (n=18)')
axB.plot(x, tie_lean_frontier, marker='o', markersize=7, linewidth=2.0, color=COLOR_FRONTIER, label='Frontier (n=6)')
axB.axhline(50, color='gray', linestyle='--', linewidth=1.0, zorder=0)
axB.set_xticks(x)
axB.set_xticklabels(degrees, fontsize=10)
axB.set_ylabel('Tie-Break Lean (%)', fontsize=10)
axB.set_title('Panel B: Tie-Break Lean', fontsize=11)
axB.grid(False)
axB.tick_params(labelsize=9)
(handles, labels) = axA.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, frameon=True, fontsize=9, bbox_to_anchor=(0.5, 1.06))
fig.suptitle('RQ1 degree-ladder: frontier vs. open-weight', fontsize=13, y=1.14)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'B1_rq1_degree_ladder.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(OUT_DIR, 'B1_rq1_degree_ladder.pdf'), bbox_inches='tight')
print('Saved B1_rq1_degree_ladder.{png,pdf}')
