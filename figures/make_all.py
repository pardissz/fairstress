"""Regenerate every plot of the paper into pdf/ (and png/ previews).

Usage: python make_all.py [FONT_DIR]
FONT_DIR holds the Manrope TTFs (OFL; github.com/sharanda/manrope or Google Fonts).
Without it, the figures fall back to DejaVu Sans; numbers and layout are the same.
Figure 1 is drawn by hand (pdf/fig1_framework.pdf) and is not regenerated here.
"""
import os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "fonts")
PDF, PNG = os.path.join(HERE, "pdf"), os.path.join(HERE, "png")
os.makedirs(PDF, exist_ok=True); os.makedirs(PNG, exist_ok=True)
tmp = tempfile.mkdtemp()

def run(*args):
    subprocess.run([sys.executable, *args], check=True, cwd=HERE)

run("make_fig2_ladder.py", "data/fig2_ladder.json", FONTS, os.path.join(PDF, "fig2_degree_ladder.pdf"))
run("make_framing_figures.py", "data/framing_by_group_and_stage.json", FONTS,
    os.path.join(PDF, "fig3_framing_7_groups.pdf"), os.path.join(PDF, "fig8_framing_24_groups.pdf"))
run("make_appendix_figures.py", "data/appendix_figures.json", FONTS, tmp)
RENAME = {
    "fig_spp_trend_ai2.pdf": "fig4_pretraining_recipes.pdf",
    "fig_loadings_readable_11models_ai2.pdf": "fig5_reasoning_axes.pdf",
    "fig_mean_shift_vector_11models_ai2.pdf": "fig6_reasoning_shift_per_model.pdf",
    "fig_category_arrows_pooled_no_age_religion_ai2.pdf": "fig7_reasoning_shift_per_category.pdf",
    "fig2_mfd_vs_stance_ai2.pdf": "fig9_mfd_vs_stance.pdf",
    "fig5_refusal_density_ai2.pdf": "fig10_refusal_vs_merit.pdf",
    "fig_training_stages_ai2.pdf": "fig11_training_stages.pdf",
}
for src, dst in RENAME.items():
    shutil.move(os.path.join(tmp, src), os.path.join(PDF, dst))

# PNG previews (needs pdftoppm from poppler; skipped if missing)
if shutil.which("pdftoppm"):
    for f in sorted(os.listdir(PDF)):
        if f.endswith(".pdf") and f != "fig1_framework.pdf":   # fig1 preview is kept as shipped
            stem = os.path.join(PNG, f[:-4])
            subprocess.run(["pdftoppm", "-png", "-r", "200", "-singlefile", os.path.join(PDF, f), stem], check=True)
print("done:", PDF)
