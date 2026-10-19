"""Figures 3 and 8: net framing (protective% minus negative%) of training-data
passages naming each group, at every OLMo-3 training stage.

Figure 3 shows seven representative groups; Figure 8 shows all 24.
Data: data/framing_by_group_and_stage.json (the paper's stance table).

Usage: python make_framing_figures.py data/framing_by_group_and_stage.json FONT_DIR OUT_FIG3.pdf OUT_FIG8.pdf
"""
import json, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import ai2_figure_style as ai2

data_path, manrope_dir, out3, out8 = sys.argv[1:5]
ai2.use(manrope_dir)
D = json.load(open(data_path))
STAGES = D["stages"]
BY = {g["group"]: g["stages"] for g in D["groups"]}

def row(group):
    net, n = [], []
    for st in STAGES:
        c = BY[group][st]
        if c["n"] is None:
            net.append(np.nan); n.append(0)
        else:
            net.append(c["protective_pct"] - c["negative_pct"]); n.append(c["n"])
    return net, n

# ---------------- Figure 3 (seven groups)
groups = [("transgender", "Transgender"), ("disabled", "Disabled"), ("Black", "Black"),
          ("women", "Women"), ("gay", "Gay"), ("men", "Men"), ("White", "White")]
stages = ["Pretrain", "SFT", "DPO\nchosen", "DPO\nrejected", "RLVR"]
net = np.array([row(k)[0] for k, _ in groups]); ns = np.array([row(k)[1] for k, _ in groups])
out = out3
fig, ax = plt.subplots(figsize=(3.25, 2.2))
norm = TwoSlopeNorm(vmin=-40, vcenter=0, vmax=60)
im = ax.imshow(net, cmap=ai2.DIVERGING_R, norm=norm, aspect="auto")
for i in range(net.shape[0]):
    for j in range(net.shape[1]):
        v = net[i, j]
        rgba = ai2.DIVERGING_R(norm(v))
        lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
        col = "white" if lum < 0.45 else ai2.FOREGROUND
        ax.text(j, i - 0.12, f"{v:+.1f}", ha="center", va="center", fontsize=6.3, color=col, fontweight="bold")
        ax.text(j, i + 0.27, f"(n={ns[i, j]})", ha="center", va="center", fontsize=4.6, color=col)
ax.set_xticks(range(5), stages)
ax.set_yticks(range(len(groups)), [g for _, g in groups])
ax.tick_params(length=0, pad=2)
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
cb.set_label("net framing (protective% $-$ negative%)", fontsize=6)
cb.ax.tick_params(labelsize=6, length=2)
cb.outline.set_visible(False)
fig.tight_layout()
fig.savefig(out)
print("wrote", out)
plt.close(fig)

# ---------------- Figure 8 (all 24 groups)
names = [g["group"] for g in D["groups"]]
net = [row(k)[0] for k in names]; ns = [row(k)[1] for k in names]
out = out8
net, ns = np.array(net), np.array(ns)
order = np.argsort(-np.nan_to_num(net[:, 0], nan=-999))
names = [names[i] for i in order]; net = net[order]; ns = ns[order]
label = {"women": "women", "men": "men", "older": "older people", "younger": "younger people"}
ylabels = [label.get(n, n + (" people" if n not in ("women", "men") else "")) for n in names]

stages = ["Pretrain", "SFT", "DPO chosen", "DPO rejected", "RLVR"]
fig, ax = plt.subplots(figsize=(6.5, 7.2))
norm = TwoSlopeNorm(vmin=-40, vcenter=0, vmax=60)
cmap = ai2.DIVERGING_R.copy(); cmap.set_bad("white")
im = ax.imshow(np.ma.masked_invalid(net), cmap=cmap, norm=norm, aspect="auto")
for i in range(net.shape[0]):
    for j in range(5):
        v = net[i, j]
        if np.isnan(v):
            continue
        r, g, b, _ = cmap(norm(v))
        col = "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45 else ai2.FOREGROUND
        ax.text(j, i - 0.13, f"{v:+.1f}", ha="center", va="center", fontsize=7, color=col, fontweight="bold")
        ax.text(j, i + 0.27, f"(n={ns[i, j]})", ha="center", va="center", fontsize=5.5, color=col)
ax.set_xticks(range(5), stages)
ax.set_yticks(range(len(names)), ylabels)
ax.xaxis.tick_bottom()
ax.tick_params(length=0, pad=3)
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.2)
ax.tick_params(which="minor", length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("net framing (protective% $-$ negative%)")
cb.outline.set_visible(False)
fig.tight_layout()
fig.savefig(out)
print("wrote", out)
