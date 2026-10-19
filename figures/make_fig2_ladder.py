"""Figure 2: every metric across the degree ladder (Type-1 contrasts).

One marker per model and degree, the mean over the 15 full-corpus models
(pink line) and the fair line (green).

Usage: python make_fig2_ladder.py data/fig2_ladder.json FONT_DIR OUT.pdf
FONT_DIR holds Manrope-*.ttf; if it does not exist, DejaVu Sans is used.
"""
import json, sys
import numpy as np
import matplotlib.pyplot as plt
import ai2_figure_style as ai2

data_path, manrope_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
ai2.use(manrope_dir, base_size=7)
D = json.load(open(data_path))
rows = D["per_model"]

# Same ten models and marker shapes as the original figure; colors follow
# model family (OLMo = teal, Tulu-3 lineage = purple, Tulu-3.1 = orange,
# other developers = dark teal and greys).
MODELS = [
    ("OLMo-3-7B",    "o",  ai2.TEAL,      4.2),
    ("OLMo-3.1-32B", "s",  ai2.TEAL,      3.8),
    ("Tulu-3-SFT",   "^",  ai2.PURPLE,    4.2),
    ("Tulu-3-DPO",   "v",  ai2.PURPLE,    4.2),
    ("Tulu-3",       "D",  ai2.PURPLE,    3.6),
    ("Tulu-3.1",     "P",  ai2.ORANGE,    4.6),
    ("Qwen2.5-7B",   "X",  "#6B7A7C",     4.2),
    ("Llama-3.1-8B", "*",  ai2.DARK_TEAL, 5.4),
    ("Gemma-3-27B",  "h",  "#A9B4B6",     4.4),
    ("GPT-4o-mini",  "d",  "#6B7A7C",     4.2),
]
PANELS = [("Accuracy Gap", "AG", "points", 0.0),
          ("Robustness Gap", "RG", "points", 0.0),
          ("Tie-Break Shift", "TS", "points", 0.0),
          ("Tie-Break Lean", "TL", "% minority picked", 50.0)]
x = np.arange(4)
xt = ["D0\nnone", "D1\nname", "D2\ncorrelate", "D3\nlabel"]
MEANS = D["mean_15_models"]

fig, axes = plt.subplots(1, 4, figsize=(6.5, 1.62))
for ax, (title, key, ylab, fair) in zip(axes, PANELS):
    ax.grid(False)
    ax.axhline(fair, color=ai2.GREEN, lw=1.4, zorder=1)
    ax.text(-0.22, fair, "fair", color=ai2.GREEN_TEXT, fontsize=6, fontweight="bold",
            ha="left", va="bottom", zorder=4)
    for name, mk, col, ms in MODELS:
        ax.plot(x, rows[name][key], ls="none", marker=mk, ms=ms, color=col,
                markeredgecolor="white", markeredgewidth=0.3, alpha=0.95, zorder=2)
    mean = MEANS[key]
    ax.plot(x, mean, color=ai2.PINK, lw=2.0, marker="o", ms=4.2,
            markeredgecolor="white", markeredgewidth=0.4, zorder=3)
    ax.set_title(title, color=ai2.METRIC[title], pad=3)
    ax.set_xticks(x, xt, fontsize=5.8, linespacing=1.0)
    ax.set_ylabel(ylab, labelpad=1.5, fontsize=6.3)
    ax.set_xlim(-0.3, 3.3)
    ax.tick_params(axis="both", length=2, pad=1.5)
    ax.tick_params(axis="y", labelsize=6)

handles = [plt.Line2D([], [], ls="none", marker=mk, ms=ms, color=col,
                      markeredgecolor="white", markeredgewidth=0.3)
           for _, mk, col, ms in MODELS]
handles.append(plt.Line2D([], [], color=ai2.PINK, lw=2.0, marker="o", ms=4.2,
                          markeredgecolor="white", markeredgewidth=0.4))
labels = [m[0] for m in MODELS] + ["mean, all 15 models"]
fig.legend(handles, labels, loc="lower center", ncol=11, bbox_to_anchor=(0.5, -0.02),
           fontsize=5.6, handletextpad=0.15, columnspacing=0.75, handlelength=1.2)
fig.tight_layout(w_pad=0.9, rect=(0, 0.08, 1, 1))
fig.savefig(out)
print("wrote", out)
