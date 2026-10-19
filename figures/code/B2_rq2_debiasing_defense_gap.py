import os
import numpy as np
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, '..', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
data = {'OLMo-3-7B': {'baseline': (-0.0886, -0.1713, -0.0037), 'LoReFT': (-0.0325, -0.113, 0.0485), 'pressure-mixed': (-0.0092, -0.0802, 0.058)}, 'Gemma-3-12B': {'baseline': (-0.0428, -0.1237, 0.0377), 'LoReFT': (-0.0202, -0.082, 0.0377), 'pressure-mixed': (-0.0167, -0.0643, 0.0309)}, 'Qwen2.5-7B': {'baseline': (-0.0371, -0.0886, 0.0127), 'LoReFT': (-0.0146, -0.0896, 0.0563), 'pressure-mixed': (-0.025, -0.0667, 0.0123)}}
models = list(data.keys())
methods = ['baseline', 'LoReFT', 'pressure-mixed']
METHOD_COLOR = {'baseline': '#4C72B0', 'LoReFT': '#C44E52', 'pressure-mixed': '#55A868'}
(fig, ax) = plt.subplots(figsize=(8, 5))
x = np.arange(len(models))
width = 0.26
for (i, method) in enumerate(methods):
    means = [data[m][method][0] for m in models]
    lo = [data[m][method][0] - data[m][method][1] for m in models]
    hi = [data[m][method][2] - data[m][method][0] for m in models]
    offset = (i - 1) * width
    ax.bar(x + offset, means, width, yerr=[lo, hi], capsize=4, color=METHOD_COLOR[method], label=method, error_kw={'elinewidth': 1.3, 'ecolor': 'black', 'capthick': 1.3})
ax.axhline(0, color='gray', linestyle='--', linewidth=1.2, zorder=0)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylabel('Defense gap (FAS flip-Δ), 95% CI', fontsize=11)
ax.set_title('RQ2 cross-model debiasing: defense-gap comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', frameon=True, fontsize=10)
ax.grid(axis='y', color='0.9', linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'B2_rq2_debiasing_defense_gap.png'), dpi=200)
fig.savefig(os.path.join(OUT_DIR, 'B2_rq2_debiasing_defense_gap.pdf'))
print('Saved B2_rq2_debiasing_defense_gap.{png,pdf}')
