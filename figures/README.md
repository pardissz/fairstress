# Figures

Every figure in the paper, in the Ai2 colors, with the code and the numbers to redraw it.
The numbers are small JSON files, so nothing here needs the 13M-item corpus.

| Paper | File | Script | Data |
|---|---|---|---|
| Figure 1, the FairStress framework | `pdf/fig1_framework.pdf` | drawn by hand (not generated) | – |
| Figure 2, every metric across the degree ladder | `pdf/fig2_degree_ladder.pdf` | `make_fig2_ladder.py` | `data/fig2_ladder.json` |
| Figure 3, training-data framing, 7 groups | `pdf/fig3_framing_7_groups.pdf` | `make_framing_figures.py` | `data/framing_by_group_and_stage.json` |
| Figure 4, pretraining-time alignment recipes | `pdf/fig4_pretraining_recipes.pdf` | `make_appendix_figures.py` | `data/appendix_figures.json` |
| Figure 5, the two reasoning-style axes | `pdf/fig5_reasoning_axes.pdf` | `make_appendix_figures.py` | `data/appendix_figures.json` |
| Figure 6, reasoning shift per model | `pdf/fig6_reasoning_shift_per_model.pdf` | `make_appendix_figures.py` | `data/appendix_figures.json` |
| Figure 7, reasoning shift per category | `pdf/fig7_reasoning_shift_per_category.pdf` | `make_appendix_figures.py` | `data/appendix_figures.json` |
| Figure 8, training-data framing, all 24 groups | `pdf/fig8_framing_24_groups.pdf` | `make_framing_figures.py` | `data/framing_by_group_and_stage.json` |
| Figure 9, MFD vice/virtue vs. stance | `pdf/fig9_mfd_vs_stance.pdf` | `make_appendix_figures.py` | `data/appendix_figures.json` |
| Figure 10, refusal training vs. merit-based decisions | `pdf/fig10_refusal_vs_merit.pdf` | `make_appendix_figures.py` | `data/appendix_figures.json` |
| Figure 11, training stages and algorithms | `pdf/fig11_training_stages.pdf` | `make_appendix_figures.py` | `data/appendix_figures.json` |

`png/` holds previews of the same figures.

## Redraw everything

```bash
python make_all.py            # writes pdf/ and png/
python make_all.py path/to/manrope
```

The figures use Manrope, the font of the Ai2 paper template. Download it (SIL Open Font License)
and pass its folder; without it, matplotlib falls back to DejaVu Sans and only the font changes.
PNG previews need `pdftoppm` (poppler).

`ai2_figure_style.py` holds the palette and matplotlib settings: Ai2 pink for the minority-coded or
overcorrecting side, Ai2 teal for the majority-coded or stereotypical side, green for the fair reference,
and one color per metric (Accuracy Gap orange, Robustness Gap dark pink, Tie-Break Shift purple,
Tie-Break Lean green).
