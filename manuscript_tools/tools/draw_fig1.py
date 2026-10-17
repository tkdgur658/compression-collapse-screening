"""Fig. 1 redraw (D6): one encoder stage i, predictive-residual skip and train-only boundary head.
Run:  python -s tools/draw_fig1.py   ->  figures/fig1_teaser_v1fix.pdf
Drawn from code/pbe_seg/models/pbe/src/net.py, matching.py, adapter.py; spec rows 1.1-1.15 of D6_FIGURE_SPEC.md."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figlib import *

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "fig1_teaser_v1fix.pdf")
W, H, OY = 1560, 800, 60
fig, ax = canvas(W, H)
ax.set_ylim(-OY, H - OY)
FS, FL, FM = 20, 17.5, 16.5          # main / small / smallest font sizes


def gbox(x0, y0, x1, y1, lines, fc=C["gray_fc"], ec="black", fs=FS, lw=1.2, rot=0, ls="-"):
    box(ax, x0, y0, x1, y1, fc=fc, ec=ec, lw=lw, r=5, ls=ls)
    text(ax, (x0 + x1) / 2, (y0 + y1) / 2, "\n".join(lines), fs=fs, rot=rot)


yc = 570          # main flow line
# ------------------------------------------------------------ input, 1x1 conv, RepConv  (net.py:114-115)
cube(ax, 15, 530, 60, 80, 28)
text(ax, 62, 495, "Input", fs=FS)
arrow(ax, [(103, yc), (130, yc)], lw=2.2)
gbox(130, 485, 182, 655, ["1×1 Conv"], fc=C["conv_fc"], rot=90)
arrow(ax, [(182, yc), (200, yc)])
gbox(200, 455, 262, 685, ["3×3 RepConv", "(DW, unfused)"], rot=90, fs=FL)
XJ = 296
arrow(ax, [(262, yc), (XJ, yc)])
text(ax, XJ + 9, yc + 15, "$x$", fs=FS + 2, ha="left")
dot(ax, XJ, yc)

# ------------------------------------------------------------ eBPB (stages 1-2 only)  (net.py:55-70, 116-117, 146)
box(ax, 318, 335, 940, 722, fc="none", ec="#1F3A5F", lw=1.6, ls=(0, (5, 3)), r=8, z=1)
text(ax, 330, 704, "eBPB (stages 1–2 only)", fs=FS, ha="left", weight="bold")
ytop, ybot = 680, 455
line(ax, [(XJ, yc), (XJ, ytop), (905, ytop)])
arrow(ax, [(905, ytop), (905, yc + 8)])
line(ax, [(XJ, yc), (XJ, ybot)])
arrow(ax, [(XJ, ybot), (335, ybot)])
gbox(335, 405, 430, 505, ["$\\mathcal{P}$: Avg", "Pool", "(4×4)"], fc=C["pool_fc"], ec=C["pool_ec"], lw=2.0)
arrow(ax, [(430, ybot), (455, ybot)])
cube(ax, 455, 441, 28, 28, 12)
text(ax, 478, 380, "$\\frac{H}{4}\\times\\frac{W}{4}\\times C_i$", fs=FL + 1)
arrow(ax, [(495, ybot), (525, ybot)])
gbox(525, 405, 662, 505, ["$\\psi$: 3×3 DW Conv", "+ 1×1 Conv", "(BN, ReLU)"], fs=FL)
arrow(ax, [(662, ybot), (685, ybot)])
gbox(685, 405, 785, 505, ["$\\mathcal{U}$: Bilinear", "Upsample"], fs=FL)
arrow(ax, [(785, ybot), (815, ybot)])
cube(ax, 815, 405, 50, 100, 22)
text(ax, 842, 380, "$H\\times W\\times C_i$", fs=FL + 1)
line(ax, [(887, ybot), (905, ybot)])
arrow(ax, [(905, ybot), (905, yc - 8)])
node(ax, 905, yc, "+", r=9)
# upper branch: U(P x) -> subtract from x  =>  H x
arrow(ax, [(468, 481), (468, 525)])
gbox(415, 525, 522, 592, ["$\\mathcal{U}$: Bilinear", "Upsample"], fs=FL)
arrow(ax, [(468, 592), (468, ytop - 9)])
node(ax, 468, ytop, "-", r=9)
text(ax, 492, ytop - 26, "$\\mathcal{H}x = x - \\mathcal{U}\\mathcal{P}x$", fs=FS, ha="left")

# ------------------------------------------------------------ PRCM -> 2x2 max-pool -> e_i  (net.py:118, 160-161)
arrow(ax, [(914, yc), (985, yc)])
gbox(985, yc - 32, 1072, yc + 32, ["Channel", "gate"], fc=C["prcm_fc"], fs=FL)  # D18 (pass 20): was "PRCM"
arrow(ax, [(1072, yc), (1100, yc)])
gbox(1100, yc - 36, 1190, yc + 36, ["2×2", "Max Pool"], fs=FL)
XT = 1222
arrow(ax, [(1190, yc), (XT - 3, yc)])
dot(ax, XT, yc, r=3.5)
text(ax, XT + 4, yc + 24, "$e_i$", fs=FS + 2, ha="left")
text(ax, XT - 12, yc - 63, "$\\frac{H}{2}\\times\\frac{W}{2}\\times C_i$", fs=FL + 1, ha="right")

# ------------------------------------------------------------ stage i+1 and predictive-residual skip  (net.py:147-151, 160-168)
YB, YR = 300, 190         # bus y, row-2 y
line(ax, [(XT, yc), (XT, YB), (336, YB), (336, YR)])
arrow(ax, [(336, YR), (362, YR)])
gbox(362, YR - 42, 498, YR + 42, ["Encoding", "stage $i+1$"], fc="white", fs=FL + 1)
arrow(ax, [(498, YR), (525, YR)])
gbox(525, YR - 32, 620, YR + 32, ["2×2", "Max Pool"], fs=FL)
arrow(ax, [(620, YR), (650, YR)])
cube(ax, 650, YR - 22, 40, 44, 16)
text(ax, 676, YR + 54, "$e_{i+1}$", fs=FS + 2)
text(ax, 676, YR - 56, "$\\frac{H}{4}\\times\\frac{W}{4}\\times C_{i+1}$", fs=FL + 1)
box(ax, 760, 112, 1050, 262, fc="none", ec="#1F3A5F", lw=1.6, ls=(0, (5, 3)), r=8, z=1)
text(ax, 772, 244, "Predictive-residual skip", fs=FS, ha="left", weight="bold")
arrow(ax, [(711, YR), (780, YR)])
gbox(780, YR - 40, 868, YR + 30, ["$W_i$:", "1×1 Conv"], fs=FL)
arrow(ax, [(868, YR - 5), (888, YR - 5)])
gbox(888, YR - 40, 984, YR + 30, ["$\\mathcal{U}$: Bilinear", "Upsample"], fs=FL)
XS = 1018
arrow(ax, [(984, YR - 5), (XS - 9, YR - 5)])
node(ax, XS, YR - 5, "-", r=9)
arrow(ax, [(XS, YB), (XS, YR + 4)])
text(ax, XS + 14, 224, "$e_i$", fs=FL + 1, ha="left", color=C["note"])
text(ax, XS + 36, YR + 12, "$s_i$", fs=FS + 2, ha="left")

# ---- skip to decoder (solid, deployed): added to the decoder output after its upsample (net.py:170-173)
XB = 1100
line(ax, [(XS + 9, YR - 5), (XB, YR - 5)])
dot(ax, XB, YR - 5, r=3.5)
yP = 85
arrow(ax, [(XB, YR - 5), (XB, yP + 9)])
node(ax, XB, yP, "+", r=9)
arrow(ax, [(790, yP), (XB - 9, yP)])
text(ax, 940, yP - 24, "decoder output (after upsample)", fs=FL)
arrow(ax, [(XB + 9, yP), (XB + 70, yP)])
text(ax, XB + 38, yP - 24, "to decoder", fs=FL)

# ---- train-only head (dashed): h_i, p_i, S, L_S  (net.py:152, 181; matching.py:14-19, 41-49)
arrow(ax, [(XB, YR - 5), (XB + 40, YR - 5)], ls="--", lw=1.3)
gbox(XB + 40, YR - 40, XB + 135, YR + 30, ["$h_i$:", "1×1 Conv"], fs=FL, ls="--")
arrow(ax, [(XB + 135, YR - 5), (XB + 175, YR - 5)], ls="--", lw=1.3)
text(ax, XB + 178, YR - 5, "$p_i=\\sigma(h_i(s_i))$", fs=FS, ha="left")
arrow(ax, [(1385, YR - 5), (1472, YR - 5)], ls="--", lw=1.3)
gbox(1472, YR - 55, 1552, YR + 45, ["$\\mathcal{S}(p_i,b_i)$:", "Cross-Cov", "Matching"], fs=FM, ls="--")
box(ax, 1360, 305, 1552, 395, fc="white", ec="black", lw=1.2, ls="--", r=5, z=3)
text(ax, 1456, 350, "$b_i$: GT morphological\ngradient (3×3 dilate\n− erode, nearest)", fs=FM)
arrow(ax, [(1512, 305), (1512, YR + 45)], ls="--", lw=1.3)
arrow(ax, [(1512, YR - 55), (1512, 115)], ls="--", lw=1.3)
text(ax, 1552, 80, "$\\mathcal{L}_S=\\frac{1}{3}\\Sigma_{i=1}^{3}(1-\\mathcal{S}(p_i,b_i))$", fs=FS, ha="right")
text(ax, 1552, 44, "$\\mathcal{L}=\\mathcal{L}_{seg}+0.1\\,\\mathcal{L}_S$", fs=FL, ha="right")

# ---- legend / notes
notes = [
    "$\\oplus$: element-wise sum;  $\\ominus$: element-wise subtraction;  dashed arrows and boxes: training-only (removed at inference), $i=1,2,3$;  solid: kept in the deployed network.",
    "$W_i$: bias-free 1×1 conv, zero-initialised;  $\\mathcal{U}$ (skip): bilinear resize to the size of $e_i$ (×2);  $s_4=e_4$ (plain, no $W_4$).  $C_i=128,192,256,384,512$ for $i=1..5$.",
    "$e_i$ is the stage output after the 2×2 max-pool.  eBPB appears at stages 1–2 only (stages 3–5: 1×1 Conv, RepConv, channel gate).  RepConv = 3×3 DW + 1×1 DW + identity BN branches, kept unfused.",
]
for j, s in enumerate(notes):
    text(ax, 12, 12 - 26 * j + 12, s, fs=FM, ha="left", color=C["note"])

save(fig, OUT)
print("saved")
