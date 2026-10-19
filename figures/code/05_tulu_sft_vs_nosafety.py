import os
import numpy as np
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, '..', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
conditions = ['pro-majority\npressure', 'null\n(baseline)', 'pro-minority\npressure']
with_safety = [0.3969, 0.5325, 0.695]
no_safety = [0.4097, 0.5189, 0.6681]
pressure_swing = {'SFT (with safety data)': 0.2981, 'SFT (no safety data)': 0.2584}
COLOR_WITH = '#3D4F5C'
COLOR_NO = '#A6483C'
(fig, axes) = plt.subplots(1, 2, figsize=(11, 5.2))
ax = axes[0]
x = np.arange(len(conditions))
ax.plot(x, with_safety, marker='o', markersize=10, linewidth=2.2, color=COLOR_WITH, label='SFT (with safety data)')
ax.plot(x, no_safety, marker='o', markersize=10, linewidth=2.2, color=COLOR_NO, label='SFT (no safety data)')
ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.0, zorder=0)
ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=11)
ax.set_ylabel('AMB minority-chosen rate', fontsize=11)
ax.set_title('Directional pressure effect', fontsize=13)
ax.legend(loc='upper left', frameon=True, fontsize=10)
ax.grid(axis='y', color='0.9', linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax2 = axes[1]
labels = list(pressure_swing.keys())
vals = list(pressure_swing.values())
bars = ax2.bar(labels, vals, color=[COLOR_WITH, COLOR_NO], width=0.6)
for (b, v) in zip(bars, vals):
    ax2.text(b.get_x() + b.get_width() / 2, v + 0.006, f'{v:.3f}', ha='center', fontsize=11)
ax2.set_ylabel('Pressure "swing"', fontsize=11)
ax2.set_title('Susceptibility to directional pressure', fontsize=13)
ax2.set_ylim(0, 0.32)
ax2.grid(False)
fig.suptitle('Tulu-3-8B-SFT vs. no-safety-data variant — AMB tie-breaking only', fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUT_DIR, '05_tulu_sft_vs_nosafety.png'), dpi=200)
fig.savefig(os.path.join(OUT_DIR, '05_tulu_sft_vs_nosafety.pdf'))
print('Saved 05_tulu_sft_vs_nosafety.{png,pdf}')
