"""Ai2 figure style for the FairStress paper.

Usage:
    import ai2_figure_style as ai2
    ai2.use(manrope_dir="ai2style/manrope")   # path to the template's Manrope TTFs
    ...plot as usual, using ai2.PINK, ai2.TEAL, ai2.METRIC, ai2.DIVERGING, ...

Brand hex values are the ones Ai2 uses in its own figure code
(allenai/OLMo scripts/flops_by_perf_figure.py, allenai/olmocr
scripts/plots/plot_olmocr2_timeline.py) and in the Ai2 LaTeX template.
"""
import os
import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

# --- Ai2 brand colors (exact) -------------------------------------------
PINK = "#F0529C"         # Ai2 pink: minority-coded / overcorrecting / highlight
PINK_LIGHT = "#F697C4"
PINK_DARK = "#A51C5C"    # Ai2's darker pink, for text
TEAL = "#105257"         # Ai2 teal: majority-coded / stereotypical
DARK_TEAL = "#0A3235"    # Ai2 dark teal: ink, reference lines
GREEN = "#0FCB8C"        # fills and large marks only (2.1:1 on white)
GREEN_LIGHT = "#6FE0BA"
PURPLE = "#B11BE8"
ORANGE = "#F65834"
YELLOW = "#FFF500"       # highlights only
OFF_WHITE = "#FAF2E9"    # Ai2 beige background
FOREGROUND = "#1C2B33"   # Ai2 template text color

# darker shades of green / orange for thin lines and small text on white
GREEN_TEXT = "#007248"
ORANGE_TEXT = "#B63517"
GREY = "#A9B4B6"         # context series (other models), recessive
GRID = "#E3E6E6"

# The four metrics: same colors as the metric names in the paper text
METRIC = {
    "Accuracy Gap": ORANGE,
    "Robustness Gap": PINK_DARK,
    "Tie-Break Shift": PURPLE,
    "Tie-Break Lean": GREEN_TEXT,
}

# Up to three series that must be told apart (validated for color-vision
# deficiency, all pairs): pink, teal, purple. More series -> grey + highlight.
CATEGORICAL = [PINK, TEAL, PURPLE]

# Diverging: teal (negative / stereotypical) - beige - pink (positive / overcorrecting)
DIVERGING = LinearSegmentedColormap.from_list(
    "ai2_diverging", [TEAL, "#8FB1B3", OFF_WHITE, PINK, PINK_DARK])
DIVERGING_R = DIVERGING.reversed()
# Sequential / ordinal: one hue, light to dark
SEQUENTIAL_PINK = LinearSegmentedColormap.from_list(
    "ai2_pink", [OFF_WHITE, PINK_LIGHT, PINK, PINK_DARK])
SEQUENTIAL_TEAL = LinearSegmentedColormap.from_list(
    "ai2_teal", [OFF_WHITE, "#8FB1B3", TEAL, DARK_TEAL])


def use(manrope_dir=None, base_size=7.5):
    """Apply the style. Pass the folder holding Manrope-*.ttf to use the
    template's sans-serif in figures (falls back to DejaVu Sans)."""
    family = "DejaVu Sans"
    if manrope_dir and os.path.isdir(manrope_dir):
        for f in os.listdir(manrope_dir):
            if f.lower().endswith(".ttf"):
                font_manager.fontManager.addfont(os.path.join(manrope_dir, f))
        family = "Manrope"
    mpl.rcParams.update({
        "font.family": [family, "DejaVu Sans"],   # DejaVu fills glyphs Manrope lacks (e.g. ↔)
        "font.size": base_size,
        "axes.titlesize": base_size + 0.5,
        "axes.titleweight": "bold",
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 0.5,
        "ytick.labelsize": base_size - 0.5,
        "legend.fontsize": base_size - 0.5,
        "text.color": FOREGROUND,
        "axes.labelcolor": FOREGROUND,
        "xtick.color": FOREGROUND,
        "ytick.color": FOREGROUND,
        "axes.edgecolor": "#B7C0C1",
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
        "lines.linewidth": 1.6,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        # mathtext in the same font, so "$\\mathbf{(c)}$" can sidestep Manrope's (c) -> © ligature
        "mathtext.fontset": "custom",
        "mathtext.rm": family,
        "mathtext.it": family + ":italic",
        "mathtext.bf": family + ":bold",
    })
