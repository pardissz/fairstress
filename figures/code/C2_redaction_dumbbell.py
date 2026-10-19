import os
import numpy as np
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, '..', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
ROWS = [('gemma', 'GT_bothcorrect', 'disability_normal', 155, 0.868, 0.558), ('gemma', 'GT_bothcorrect', 'blackM_whiteM', 50, 0.615, 0.473), ('gemma', 'GT_bothcorrect', 'blackF_whiteM', 27, 0.707, 0.529), ('gemma', 'GT_bothcorrect', 'older_younger', 165, 0.882, 0.554), ('gemma', 'AMB_chosen', 'disability_normal', 172, 0.82, 0.706), ('gemma', 'AMB_chosen', 'blackM_whiteM', 45, 0.708, 0.679), ('gemma', 'AMB_chosen', 'blackF_whiteM', 35, 0.75, 0.767), ('gemma', 'AMB_chosen', 'older_younger', 171, 0.849, 0.503), ('qwen', 'GT_bothcorrect', 'disability_normal', 154, 0.894, 0.482), ('qwen', 'GT_bothcorrect', 'blackM_whiteM', 49, 0.612, 0.512), ('qwen', 'GT_bothcorrect', 'blackF_whiteM', 28, 0.7, 0.536), ('qwen', 'GT_bothcorrect', 'older_younger', 166, 0.731, 0.587), ('qwen', 'AMB_chosen', 'disability_normal', 171, 0.822, 0.693), ('qwen', 'AMB_chosen', 'blackM_whiteM', 44, 0.645, 0.632), ('qwen', 'AMB_chosen', 'blackF_whiteM', 35, 0.644, 0.578), ('qwen', 'AMB_chosen', 'older_younger', 172, 0.786, 0.527)]
CONTRAST_LABELS = {'disability_normal': 'disability', 'blackM_whiteM': 'blackM_whiteM', 'blackF_whiteM': 'blackF_whiteM', 'older_younger': 'older_younger'}
CONTRAST_ORDER = ['disability_normal', 'blackM_whiteM', 'blackF_whiteM', 'older_younger']
MODEL_COLOR = {'gemma': '#3D4F5C', 'qwen': '#A6483C'}
(fig, axes) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
for (ax, task, title) in zip(axes, ['GT_bothcorrect', 'AMB_chosen'], ['GT (both correct)', 'AMB (null tie-break)']):
    task_rows = [r for r in ROWS if r[1] == task]
    y_labels = []
    y_pos = []
    y = 0
    for contrast in CONTRAST_ORDER:
        for model in ['gemma', 'qwen']:
            row = next((r for r in task_rows if r[0] == model and r[2] == contrast))
            (_, _, _, n, a_orig, e_red) = row
            ax.plot([a_orig, e_red], [y, y], color=MODEL_COLOR[model], linewidth=1.8, zorder=1, alpha=0.6)
            ax.scatter([a_orig], [y], color=MODEL_COLOR[model], marker='o', s=70, zorder=3, label=f'{model} (A′ original)' if y == 0 else None)
            ax.scatter([e_red], [y], color=MODEL_COLOR[model], marker='s', s=60, zorder=3, facecolors='white', edgecolors=MODEL_COLOR[model], linewidths=1.8)
            y_labels.append(f'{CONTRAST_LABELS[contrast]} ({model}, n={n})')
            y_pos.append(y)
            y += 1
        y += 0.4
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=1.0, zorder=0)
    ax.set_xlabel('Separability accuracy', fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.grid(axis='x', color='0.9', linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0.4, 0.95)
legend_elems = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3D4F5C', markersize=9, label='gemma, A′ (original)'), plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='white', markeredgecolor='#3D4F5C', markersize=9, label='gemma, E (redacted)'), plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#A6483C', markersize=9, label='qwen, A′ (original)'), plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='white', markeredgecolor='#A6483C', markersize=9, label='qwen, E (redacted)')]
fig.legend(handles=legend_elems, loc='lower center', ncol=4, frameon=True, fontsize=9, bbox_to_anchor=(0.5, -0.05))
fig.suptitle('Identity-redaction control: separability drops when candidate names are redacted,\nbut does not collapse to chance', fontsize=13)
fig.tight_layout(rect=[0, 0.03, 1, 0.93])
fig.savefig(os.path.join(OUT_DIR, 'C2_redaction_dumbbell.png'), dpi=200, bbox_inches='tight')
print('Saved C2_redaction_dumbbell.png')
