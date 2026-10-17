"""Fig. 2 (protocol figure), Pass 30a / Pass 31: flow of the deployment rules with the measured basis of each rule.
Style helpers from tools/figlib.py (same as draw_fig1.py / draw_fig2.py).
Pass 32 ([internal decision], renumbering): box labels R1 (requirement T, optional) -> P (premise, optional), R2 -> R1 (screen),
R3 -> R2 (scoped retraining); "screened again by R2" -> "by R1". Option --no-p is an alias of --no-r1. Pass 32 (journal review 2, A-1, A-4, [internal decision]): R1 box with
per-operator basis and held-out threshold values (Q_ROBUST.json sha 581598a8), R2 box as an untested observation with
the D counts 9/16/55 of 66, "Deploy" -> "Accept checkpoint" (editor item 1). The Numbers paragraph below uses the
pre-Pass-32 rule names.

Run:  python -s tools/draw_fig_protocol.py                 -> figures/fig2_protocol_draft.pdf (R1 as an optional dashed box)
      python -s tools/draw_fig_protocol.py --no-r1 --final -> figures/fig2_protocol.pdf (R2 -> R3 only, v3)

Numbers (no new number): R1 basis from 4f (tiles within 0.5 GiB, thin-vessel ratio preserved for four models except
UCMNet at 256 px); R2 basis from 4g / Table tab:collapse_deploy (agreement AUC 0.973 in setting D) and the post hoc
threshold 0.738 of results/qdeploy/tau/TAU_RESULT.json (sha 8c7c55e0); R3 basis from Table
tab:collapse_deploy (S-l50c 57 of 66; S-w4c 9 of 66, S-a6c 6 of 66, S-w4g 38 of 66).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figlib import *

NO_R1 = "--no-r1" in sys.argv or "--no-p" in sys.argv
FINAL = "--final" in sys.argv
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures", "fig2_protocol.pdf" if FINAL else "fig2_protocol_draft.pdf")

FS, FL, FM = 15, 13.5, 12.5
W = 820 if NO_R1 else 990
H = 405
fig, ax = canvas(W, H)
RULE_FC = "#EAF1F8"; RULE_EC = "#2E5C8A"; OPT_FC = "#F4F4F4"


def gbox(x0, y0, x1, y1, title, lines, fc=RULE_FC, ec=RULE_EC, ls="-", lw=1.6):
    box(ax, x0, y0, x1, y1, fc=fc, ec=ec, lw=lw, ls=ls, r=6)
    text(ax, (x0 + x1) / 2, y1 - 17, title, fs=FS, weight="bold")
    ax.plot([x0 + 8, x1 - 8], [y1 - 31, y1 - 31], color=ec, lw=0.8, zorder=5)
    for j, s in enumerate(lines):
        text(ax, x0 + 10, y1 - 48 - 19 * j, s, fs=FL, ha="left")


yc = 295                      # main flow line
x = 10
# trained checkpoint
box(ax, x, yc - 40, x + 100, yc + 40, fc=C["gray_fc"], ec="black", r=5)
text(ax, x + 50, yc, "Trained fp32\ncheckpoint", fs=FL)
x += 100
if not NO_R1:
    arrow(ax, [(x, yc), (x + 22, yc)])
    x += 22
    gbox(x, yc - 85, x + 205, yc + 85, "P (premise, optional)",
         ["Tile inference within the", "memory budget, tile size", "checked per model",
          "Basis: 0.5 GiB tiles preserve", "thin-vessel ratio for four", "models, except UCMNet at 256 px"],
         fc=OPT_FC, ec="#555555", ls=(0, (5, 3)))
    x += 205
arrow(ax, [(x, yc), (x + 22, yc)])
x += 22
XC0 = x
box(ax, x, yc - 40, x + 112, yc + 40, fc=C["pool_fc"], ec=C["pool_ec"], lw=1.6, r=5)
text(ax, x + 56, yc, "Compress in the\ndeployed graph\n(w4, a6, l50)", fs=FM)
x += 112
XC1 = x
arrow(ax, [(x, yc), (x + 22, yc)])
x += 22
XR2 = x
gbox(x, yc - 100, x + 262, yc + 100, "R1 (requirement Q)",
     ["Label-free screen of every", "compressed checkpoint: flag if",
      "output agreement $\\leq$ 0.738 (post hoc)",
      "Basis: AUC 0.973 in setting D,", "above both controls for w4, l50,", "level with SQNR for a6",
      "Held-out models / datasets:", "sens. 0.86, spec. 0.92 / 0.89"])
x += 262
XR2e = x
arrow(ax, [(x, yc), (x + 90, yc)])
text(ax, x + 45, yc + 16, "not flagged", fs=FM, color=C["note"])
x += 90
box(ax, x, yc - 30, x + 90, yc + 30, fc="white", ec="black", r=5)
text(ax, x + 45, yc, "Accept\ncheckpoint", fs=FL)

# R2 (formerly R3) below R1 (formerly R2)
y3 = 10
# Pass 34 ([internal decision], 2026-10-17, SGR result): R2 box partially verified. Before: title "R2 (observation, not tested)",
# lines "Compression-aware training only" / "for settings that fail the screen" / "In D, 9 (w4), 16 (a6), and 55 (l50)" /
# "of 66 cells collapse" / "C1, C3: mixed modes of setting O"; side note "No pruning-only or / prune-then-fine-tune / variant was run".
# Numbers: RESULT_SGR.json (sha d55a2476) aux.main.B1 26/55, aux.w4.B5 13/15.
gbox(XR2, y3, XR2 + 262, y3 + 150, "R2 (partially verified)",
     ["Per-channel scales first, then", "retrain only failing checkpoints:", "fp32 outputs as targets, no labels,",
      "mask and bit width fixed", "Restored 26 of 55 pruning cells"],
     fc=RULE_FC)
text(ax, XR2e + 12, y3 + 75, "Per-channel 4-bit\nscales restored 13 of\n15 per-tensor w4 cells",
     fs=FM, ha="left", va="center", color=C["note"])
xm = XR2 + 131
arrow(ax, [(xm, yc - 100), (xm, y3 + 150)])
text(ax, xm + 8, (yc - 100 + y3 + 150) / 2, "flagged", fs=FM, ha="left", color=C["note"])
# retrained checkpoint goes back to compression and screening
xb = (XC0 + XC1) / 2
arrow(ax, [(XR2, y3 + 75), (xb, y3 + 75), (xb, yc - 40)])
text(ax, xb - 8, y3 + 75, "retrained checkpoint,\nscreened again by R1", fs=FM, ha="right", va="center", color=C["note"])

save(fig, OUT)
print("saved", os.path.normpath(OUT))
