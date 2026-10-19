"""Appendix plots of the paper (Figures 4-7 and 9-11) in Ai2 colors.

Usage:
    python make_appendix_figures.py data/appendix_figures.json FONT_DIR OUT_DIR [name ...]

The numbers come from appendix_figs_data.json. Where the paper has a table, the
values are the table's. Where it does not, they were read from the vector paths
of the original PDFs (see extract/), calibrated against the axis ticks or a
line whose values are known; integer counts reproduce the caption's total of
23,008 excerpts exactly.
"""
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyArrowPatch
import ai2_figure_style as ai2

data_path, manrope_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
ONLY = set(sys.argv[4:])
D = json.load(open(data_path))
ai2.use(manrope_dir, base_size=7)
os.makedirs(out_dir, exist_ok=True)

GREY_DARK = "#6B7A7C"
GREY_LIGHT = "#D5DBDC"
TEAL_MID = "#5E8E91"


def save(fig, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def fair_line(ax, y, label=True, x=None):
    ax.axhline(y, color=ai2.GREEN, lw=1.3, zorder=2.5)
    if label:
        lo, hi = ax.get_xlim()
        ax.text(lo + 0.02 * (hi - lo) if x is None else x, y, "fair", color=ai2.GREEN_TEXT,
                fontsize=5.8, fontweight="bold", ha="left", va="bottom", zorder=5)


# ---------------------------------------------------------------- App. F
def spp_trend():
    d = D["spp_trend"]
    rec = d["recipes"]
    # control in grey; the four interventions in one green ramp, light to dark,
    # in order of increasing constitutional exposure (green = the fair direction)
    cols = [ai2.GREY, "#A4E9D1", ai2.GREEN_LIGHT, ai2.GREEN, ai2.GREEN_TEXT]
    panels = [("Accuracy Gap", None, "AG"), ("Robustness Gap", "(mean of 9 families)", "RG"),
              ("Tie-Break Shift", None, "TS"), ("Tie-Break Lean", "(distance from 50)", "TL")]
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 1.9))
    x = np.arange(5)
    for i, (ax, (title, sub, key)) in enumerate(zip(axes, panels)):
        v = d[key]
        ax.grid(False)
        ax.bar(x, v, width=0.72, color=cols, edgecolor="white", linewidth=0.4, zorder=2)
        for xi, vi in zip(x, v):
            ax.text(xi, vi + 0.25, f"{vi:.1f}", ha="center", va="bottom", fontsize=5.8, zorder=3)
        ax.axhline(0, color=ai2.GREEN, lw=1.3, zorder=1)
        top = max(v) * 1.32
        ax.set_ylim(0, top)
        ax.set_xlim(-0.6, 5.05)
        # "toward fair" arrow, pointing down to the fair line
        ax.annotate("", xy=(4.78, 0.03 * top), xytext=(4.78, 0.80 * top),
                    arrowprops=dict(arrowstyle="-|>", color=ai2.GREEN_TEXT, lw=1.4,
                                    mutation_scale=7, shrinkA=0, shrinkB=0), zorder=4)
        ax.text(4.9, 0.83 * top, "toward fair", ha="right", va="bottom", fontsize=5.8,
                fontweight="bold", color=ai2.GREEN_TEXT)
        ax.set_title(title if sub is None else f"{title}\n", color=ai2.METRIC[title], pad=3,
                     fontsize=7.3, linespacing=1.0)
        if sub:
            ax.text(0.5, 1.035, sub, transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=6, color=ai2.FOREGROUND)
        ax.set_xticks(x, rec, fontsize=6, rotation=28, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", length=0, pad=2)
        ax.tick_params(axis="y", length=2, labelsize=6, pad=1.5)
        if i == 0:
            ax.set_ylabel("points at D3", fontsize=6.5, labelpad=2)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_spp_trend_ai2.pdf")


# ---------------------------------------------------------------- App. J
def training_stages():
    d = D["training_stages"]
    # one color and marker per training stage, the same in every column
    SFT, DPO, FIN, GRPO = ai2.PURPLE, ai2.TEAL, ai2.PINK, ai2.ORANGE
    cols = [
        ("(a) OLMo-3-7B-Instruct stages", [
            ("SFT", d["OLMo-3-7B-SFT"], SFT, "^", "-"),
            ("DPO", d["OLMo-3-7B-DPO"], DPO, "s", "-"),
            ("final (RLVR)", d["OLMo-3-7B"], FIN, "o", "-")]),
        ("(b) Tulu-3-8B SFT, safety data", [
            ("with safety data", d["Tulu-3-SFT"], SFT, "^", "-"),
            ("without safety data", d["Tulu-3-SFT-no-safety (CoT)"], ai2.DARK_TEAL, "D", (0, (3.2, 1.6)))]),
        ("$\\mathbf{(c)}$ Tulu-3-8B stages and Tulu-3.1", [
            ("SFT", d["Tulu-3-SFT"], SFT, "^", "-"),
            ("DPO", d["Tulu-3-DPO"], DPO, "s", "-"),
            ("Tulu-3 (PPO)", d["Tulu-3"], FIN, "o", "-"),
            ("Tulu-3.1 (GRPO)", d["Tulu-3.1"], GRPO, "P", "-")]),
    ]
    rows = [("Accuracy Gap", "AG", 0.0, "points"), ("Robustness Gap", "RG", 0.0, "points"),
            ("Tie-Break Shift", "TS", 0.0, "points"), ("Tie-Break Lean", "TL", 50.0, "% minority picked")]
    fig, axes = plt.subplots(4, 3, figsize=(6.5, 7.0), sharex=True)
    x = np.arange(4)
    for c, (ctitle, series) in enumerate(cols):
        for r, (rtitle, key, fair, unit) in enumerate(rows):
            ax = axes[r, c]
            ax.grid(True, axis="y")
            for name, vals, col, mk, ls in series:
                y = vals[key]
                if c == 1 and key == "TS" and name == "with safety data":
                    y = d["Tulu-3-SFT-CoT-TS"]   # same chain-of-thought pipeline as the dashed line
                ax.plot(x, y, color=col, ls=ls, lw=1.5, marker=mk, ms=3.8,
                        markeredgecolor="white", markeredgewidth=0.4, zorder=3, label=name)
            ax.set_xlim(-0.25, 3.25)
            fair_line(ax, fair, label=(c == 0), x=-0.2)
            ax.tick_params(axis="both", length=2, pad=1.5, labelsize=6)
            if r == 0:
                ax.set_title(ctitle, fontsize=7.2, pad=4)
            if c == 0:
                ax.set_ylabel(f"{rtitle}\n{unit}", fontsize=6.5, labelpad=2, color=ai2.FOREGROUND)
                ax.yaxis.label.set_color(ai2.METRIC[rtitle])
            if r == 0:
                ax.legend(loc="upper left", fontsize=5.8, handlelength=1.8, handletextpad=0.4,
                          borderaxespad=0.2, labelspacing=0.25)
        axes[-1, c].set_xticks(x, ["D0", "D1", "D2", "D3"])
    fig.tight_layout(h_pad=0.6, w_pad=0.8)
    save(fig, "fig_training_stages_ai2.pdf")


# ---------------------------------------------------------------- App. I
def refusal_density():
    d = D["refusal_density"]
    fig, ax = plt.subplots(figsize=(5.35, 1.45))
    labels = ["Semantic clusters that mention\ndeciding on merit\n(9 of 28,000 indexed)",
              "Refusal-training examples from\nWildGuardMix and WildJailbreak\n(in the SFT corpus)"]
    vals = [d["merit_clusters"], d["refusal_examples"]]
    ax.barh([1, 0], vals, height=0.55, color=[ai2.TEAL, ai2.PINK], zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(1, 4e5)
    ax.set_ylim(-0.5, 1.75)
    ax.set_yticks([1, 0], labels, fontsize=6.3)
    for y, v in zip([1, 0], vals):
        ax.text(v * 1.25, y, f"{v:,}", va="center", ha="left", fontsize=6.8, fontweight="bold",
                color=ai2.TEAL if y == 1 else ai2.PINK_DARK)
    ax.text(12, 1.36, "all 9 are themselves refusals that turn to merit language;\nnone shows a decision being made",
            fontsize=5.8, color=GREY_DARK, va="bottom", ha="left")
    ax.set_xlabel("number of items (log scale)", fontsize=6.5, labelpad=2)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", length=2, labelsize=6)
    fig.tight_layout()
    save(fig, "fig5_refusal_density_ai2.pdf")


# ---------------------------------------------------------------- App. H
def loadings():
    d = D["loadings"]
    specs = [("axis1", "Axis 1: confident and positive vs. distressed and negative",
              "Distressed / Negative", "Confident / Positive"),
             ("axis2", "Axis 2: relational and experiential vs. topical and procedural",
              "Topical / Procedural", "Relational / Experiential")]
    fig, axes = plt.subplots(2, 1, figsize=(5.55, 7.6))
    for ax, (key, title, neg, pos) in zip(axes, specs):
        rows = d[key]
        n = len(rows)
        y = np.arange(n)[::-1]
        v = np.array([r["loading"] for r in rows])
        ax.barh(y, v, height=0.72, color=[ai2.PINK if t > 0 else ai2.TEAL for t in v], zorder=2)
        ax.axvline(0, color=ai2.DARK_TEAL, lw=0.6, zorder=3)
        ax.set_yticks(y, [r["feature"] for r in rows], fontsize=6.2)
        ax.set_ylim(-1.9, n - 0.4)
        lo, hi = min(v.min(), -0.02), max(v.max(), 0.02)
        pad = 0.06 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
        ax.text(lo - pad * 0.6, -1.3, "← " + neg, color=ai2.TEAL, fontsize=6.2, fontweight="bold",
                ha="left", va="center")
        ax.text(hi + pad * 0.6, -1.3, pos + " →", color=ai2.PINK_DARK, fontsize=6.2, fontweight="bold",
                ha="right", va="center")
        ax.set_title(title, fontsize=7.2, pad=4)
        ax.set_xlabel("PCA loading", fontsize=6.5, labelpad=2)
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        ax.tick_params(axis="y", length=0, pad=2)
        ax.tick_params(axis="x", length=2, labelsize=6)
    fig.tight_layout(h_pad=1.2)
    save(fig, "fig_loadings_readable_11models_ai2.pdf")


def mean_shift():
    d = D["mean_shift"]
    models = list(d.keys())          # already in the original's order (largest axis-1 shift first)
    n = len(models)
    y = np.arange(n)[::-1].astype(float)
    h = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 3.15))
    a1 = [d[m]["axis1"]["shift"] for m in models]; e1 = [d[m]["axis1"]["se"] for m in models]
    a2 = [d[m]["axis2"]["shift"] for m in models]; e2 = [d[m]["axis2"]["se"] for m in models]
    ax.barh(y + h / 2, a2, height=h, color=ai2.TEAL, zorder=2, label="axis 2 (toward relational)")
    ax.barh(y - h / 2, a1, height=h, color=ai2.PINK, zorder=2, label="axis 1 (toward confident and positive)")
    # standard errors: a white halo under a dark whisker keeps them visible on the dark teal bars
    for yy, vv, ee, halo in ((y + h / 2, a2, e2, True), (y - h / 2, a1, e1, False)):
        layers = (("white", 1.5, 3), (ai2.FOREGROUND, 0.7, 4)) if halo else ((ai2.FOREGROUND, 0.7, 4),)
        for col, lw, z in layers:
            ax.errorbar(vv, yy, xerr=ee, fmt="none", ecolor=col, elinewidth=lw, capsize=1.8,
                        capthick=lw, zorder=z)
    ax.set_yticks(y, models, fontsize=6.5)
    ax.set_xlim(0, 0.205)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("Mean shift from majority-chosen to minority-chosen reasoning", fontsize=6.5, labelpad=2)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", length=2, labelsize=6)
    hl, ll = ax.get_legend_handles_labels()
    ax.legend(hl[::-1], ll[::-1], loc="lower right", fontsize=6.3, handlelength=1.2, borderaxespad=0.3)
    fig.tight_layout()
    save(fig, "fig_mean_shift_vector_11models_ai2.pdf")


# ---------------------------------------------------------------- Supplement
def mfd_vs_stance():
    d = D["mfd_vs_stance"]
    groups = list(d.keys())          # ordered by the MFD ratio, highest first
    hi = {"race:white", "trans_status:cisgender"}
    cols = [ai2.PINK if g in hi else ai2.GREY for g in groups]
    y = np.arange(len(groups))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.5), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.03, wspace=0.08)
    specs = [("mfd", "Method 3: MFD/LIWC lexical density\n(pretraining, n=1000 every group)",
              "MFD Vice/Virtue ratio (higher = more vice-coded)"),
             ("protective_pct", "Method 1: LLM stance classification\n(same group order as left panel)",
              "Method 1: % classified PROTECTIVE (pretraining)")]
    for ax, (key, title, xlab) in zip(axes, specs):
        v = [d[g][key] for g in groups]
        ax.barh(y, v, height=0.72, color=cols, zorder=2)
        ax.set_yticks(y, groups, fontsize=6.2)
        for t, g in zip(ax.get_yticklabels(), groups):
            if g in hi:
                t.set_color(ai2.PINK_DARK); t.set_fontweight("bold")
        ax.set_ylim(-0.7, len(groups) - 0.3)
        ax.set_title(title, fontsize=7, pad=4, linespacing=1.15)
        ax.set_xlabel(xlab, fontsize=6.3, labelpad=2)
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        ax.tick_params(axis="y", length=0, pad=2)
        ax.tick_params(axis="x", length=2, labelsize=6)
    fig.suptitle("Cross-method disagreement: which groups rank as most vice-coded (left)\n"
                 "vs. most protected (right)? White/cisgender highlighted in pink.",
                 fontsize=7, fontweight="normal", linespacing=1.2)
    save(fig, "fig2_mfd_vs_stance_ai2.pdf")


def category_arrows():
    d = D["category_arrows"]
    order = ["OLMo-3-7B-Instruct", "OLMo-3-7B-Instruct-DPO", "OLMo-3-7B-Instruct-SFT",
             "OLMo-3.1-32B-Instruct", "Tulu-3-8B-SFT", "Tulu-3-8B-SFT-no-safety-data",
             "Tulu-3-8B", "Tulu-3-8B-DPO", "Tulu-3.1-8B",
             "Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct"]
    cats = [("race x sex", ai2.GREY, "race × sex"), ("race", ai2.TEAL, "race"),
            ("sex / gender", ai2.PINK, "sex / gender"), ("sexuality", ai2.PURPLE, "sexuality"),
            ("disability", ai2.ORANGE, "disability")]
    fig, axes = plt.subplots(4, 3, figsize=(6.5, 6.9), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, order):
        ax.grid(False)
        ax.axhline(0, color=ai2.GRID, lw=0.7, zorder=0)
        ax.axvline(0, color=ai2.GRID, lw=0.7, zorder=0)
        for key, col, _ in cats:
            r = d[m][key]
            (x0, y0), (x1, y1) = r["start"], r["end"]
            ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                         lw=1.4, color=col, shrinkA=0, shrinkB=0, zorder=3))
            ax.plot(x0, y0, "o", ms=2.8, color=col, markeredgecolor="white", markeredgewidth=0.4, zorder=4)
        ax.set_title(m, fontsize=6.6, pad=2.5)
        ax.set_xlim(-0.82, 0.62)
        ax.set_ylim(-0.48, 0.42)
        ax.tick_params(length=2, labelsize=5.8, pad=1.5)
        ax.spines["left"].set_color("#B7C0C1"); ax.spines["bottom"].set_color("#B7C0C1")
    leg = axes.flat[-1]
    leg.axis("off")
    handles = [Line2D([], [], color=c, lw=1.6, marker=">", ms=3.5, label=l) for _, c, l in cats[::-1]]
    leg.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=7,
               title="category", title_fontsize=7, handlelength=2.0, labelspacing=0.5, frameon=False)
    leg.text(0.5, -0.14, "dot = majority-chosen reasoning\narrow tip = minority-chosen reasoning",
             transform=leg.transAxes, ha="center", va="bottom", fontsize=6, color=GREY_DARK,
             linespacing=1.3)
    leg.xaxis.set_tick_params(labelbottom=True)
    axes[2, 2].xaxis.set_tick_params(labelbottom=True)
    fig.supxlabel("Axis 1 (confident and positive to the right)", fontsize=6.8, y=0.005)
    fig.supylabel("Axis 2 (relational and experiential upward)", fontsize=6.8, x=0.005)
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    save(fig, "fig_category_arrows_pooled_no_age_religion_ai2.pdf")


ALL = dict(spp_trend=spp_trend, training_stages=training_stages, refusal_density=refusal_density,
           loadings=loadings, mean_shift=mean_shift, mfd_vs_stance=mfd_vs_stance,
           category_arrows=category_arrows)
for name, fn in ALL.items():
    if not ONLY or name in ONLY:
        fn()
