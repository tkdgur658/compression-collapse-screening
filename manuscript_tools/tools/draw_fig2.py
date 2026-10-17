"""Fig. 2 redraw (D6): overall architecture, drawn from code/pbe_seg/models/pbe/src/net.py.
Run:  python -s tools/draw_fig2.py   ->  figures/fig2_model_v1fix.pdf
Spec rows: D6_FIGURE_SPEC.md section 1 (2.1-2.18).  net.py line numbers in comments."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figlib import *

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "fig2_model_v1fix.pdf")
W, H = 576, 640
P = 72            # row pitch
BH = 19           # box half height
Y = [H - 82 - P * k for k in range(5)]   # Y[k]: resolution level H/2^k (k = 0..4)
XE, XD = 64, 504  # encoder / decoder column centres
BW = 50           # box half width (encoder / decoder column)
fig, ax = canvas(W, H)

CH = [128, 192, 256, 384, 512]            # c_list (adapter.py:8)
ENC = [("eBPB Block", "ebpb"), ("eBPB Block", "ebpb"), ("Basic Block", "basic"),
       ("Basic Block", "basic"), ("Basic Block", "basic")]   # net.py:125-126, 146


def blk(x, y, name, kind, sub, hw=BW):
    box(ax, x - hw, y - BH, x + hw, y + BH, fc=C[kind + "_fc"], ec=C[kind + "_ec"], lw=2.0, r=5)
    text(ax, x, y + 5, name, fs=12.5)
    text(ax, x, y - 8, sub, fs=9.5, color=C["note"])


# ---------------- encoder column (encoder1..5 at H, H/2, H/4, H/8, H/16) -----------
res_in = ["H", "H/2", "H/4", "H/8", "H/16"]
for k, (nm, kd) in enumerate(ENC):
    blk(XE, Y[k], nm, kd, "Enc. %d: %d ch, %s" % (k + 1, CH[k], res_in[k]))
text(ax, XE, Y[0] + 50, "Input Image", fs=12.5)
arrow(ax, [(XE, Y[0] + 42), (XE, Y[0] + BH)])

# 4 max-pools, taps e_1..e_4 after the pool (net.py:160-161)
skip_y = {}
for k in range(1, 5):
    yt = Y[k] + 36
    skip_y[k] = yt
    arrow(ax, [(XE, Y[k - 1] - BH), (XE, Y[k] + BH)])
    text(ax, XE + 7, Y[k] + 47, "2×2 Max Pool", fs=9.5, ha="left")
    dot(ax, XE, yt)
    text(ax, XE + 7, Y[k] + 27, "$e_%d$: $H/%d$, %d ch" % (k, 2 ** k, CH[k - 1]),
         fs=9.5, ha="left", color=C["note"])

# ---------------- level-4 row: encoder5 -> bottleneck -> decoder1 -> (+s4) -> decoder2 ---------
BNX, D1X, P4X = 198, 318, 396
box(ax, BNX - 44, Y[4] - BH, BNX + 44, Y[4] + BH, fc=C["bott_fc"], ec=C["bott_ec"], lw=2.2, r=5)
text(ax, BNX, Y[4] + 5, "Bottleneck", fs=12.5)
text(ax, BNX, Y[4] - 8, "512 ch, H/16", fs=9.5, color=C["note"])
arrow(ax, [(XE + BW, Y[4]), (BNX - 44, Y[4])])
blk(D1X, Y[4], "Basic Block", "basic", "Dec. 1: 384 ch, H/16", hw=44)
arrow(ax, [(BNX + 44, Y[4]), (D1X - 44, Y[4])])
node(ax, P4X, Y[4], "+")
arrow(ax, [(D1X + 44, Y[4]), (P4X - 7, Y[4])])
blk(XD, Y[4], "Basic Block", "basic", "Dec. 2: 256 ch, H/16")
arrow(ax, [(P4X + 7, Y[4]), (XD - BW, Y[4])])

# ---------------- decoder column ------------------------------------------------------------
dec = {3: "Dec. 3: 192 ch, H/8", 2: "Dec. 4: 128 ch, H/4"}
for k in (3, 2):
    blk(XD, Y[k], "Basic Block", "basic", dec[k])
# final 1x1 conv at H/2 (net.py:131, 174)
box(ax, XD - BW, Y[1] - BH, XD + BW, Y[1] + BH, fc=C["conv_fc"], ec="black", lw=1.6, r=5)
text(ax, XD, Y[1] + 5, "1×1 Conv", fs=12.5)
text(ax, XD, Y[1] - 8, "128→1 ch, H/2", fs=9.5, color=C["note"])
text(ax, XD, Y[0], "Output", fs=12.5)
# 3 upsamples with the skip-add (+) nodes (net.py:170-173)
for k in (3, 2, 1):                       # k = skip index; arrow from the box at level k+1 up to the box at level k
    y_lo = Y[k + 1] + BH
    y_hi = Y[k] - BH
    py = Y[k] - 36
    arrow(ax, [(XD, y_lo), (XD, py - 7)])
    arrow(ax, [(XD, py + 7), (XD, y_hi)])
    node(ax, XD, py, "+")
    text(ax, XD + 7, (y_lo + py - 7) / 2 + 1, "2×2 Upsample", fs=9.5, ha="left")
# fourth upsample after the final 1x1 conv (net.py:174)
arrow(ax, [(XD, Y[1] + BH), (XD, Y[0] - 12)])
text(ax, XD + 7, (Y[1] + BH + Y[0] - 12) / 2, "2×2 Upsample", fs=9.5, ha="left")

# ---------------- skips -----------------------------------------------------------------------
XM = {1: 168, 2: 218, 3: 268}         # x of the (-) nodes / predictive arrows
XV = {1: 442, 2: 424, 3: 406}         # elbow x of the skip lines
SK = "-."                              # dash-dot as in the original
for k in (1, 2, 3):
    yk = skip_y[k]
    ypl = Y[k] - 36                    # y of the (+) node in the decoder column
    line(ax, [(XE, yk), (XM[k] - 7, yk)], ls=SK)
    line(ax, [(XM[k] + 7, yk), (XV[k], yk), (XV[k], ypl)], ls=SK)
    arrow(ax, [(XV[k], ypl), (XD - 7, ypl)], ls=SK)
    node(ax, XM[k], yk, "-")
    text(ax, XM[k] + 26, yk + 8, "$s_%d$" % k, fs=10.5)
    # predictive arrow from e_{k+1} (branch point on the line of skip k+1) up to the (-) node
    y_src = skip_y[k + 1]
    dot(ax, XM[k], y_src, r=2.4)
    arrow(ax, [(XM[k], y_src), (XM[k], yk - 7)], ls=SK)
    ym = (y_src + yk) / 2
    box(ax, XM[k] - 24, ym - 9, XM[k] + 24, ym + 9, fc=C["gray_fc"], ec="black", lw=1.0, r=2, z=4)
    text(ax, XM[k], ym, "$W_%d$, $\\mathcal{U}$" % k, fs=10)
# plain skip s4 = e4: from tap e4 to the plus node after decoder 1
y4 = skip_y[4]
line(ax, [(XE, y4), (P4X, y4)], ls=SK)
arrow(ax, [(P4X, y4), (P4X, Y[4] + 7)], ls=SK)
text(ax, 150, y4 + 8, "$s_4 = e_4$ (plain)", fs=10.5, ha="left")
text(ax, 300, Y[1] + 62, "Predictive-residual skip", fs=11, style="italic")
text(ax, 300, Y[1] + 49, "$s_i = e_i - \\mathcal{U}(W_i e_{i+1})$, $i=1,2,3$", fs=10, color=C["note"])

# ---------------- train-only heads (dashed): h_i on s_1..s_3, L_S -----------------------------
for k in (1, 2, 3):
    yk = skip_y[k]
    xb = 338
    ytop = yk - 27
    line(ax, [(xb, yk), (xb, ytop)], ls="--", lw=1.0)
    box(ax, xb - 33, ytop - 22, xb + 33, ytop, fc="white", ec="black", lw=1.0, ls="--", r=3, z=4)
    text(ax, xb, ytop - 11, "$h_%d$ → $\\mathcal{L}_S$" % k, fs=9.5)
    dot(ax, xb, yk, r=2.0)


# ---------------- legend ----------------------------------------------------------------------
def small(x0, x1, yc, l1, l2=None, fc=C["conv_fc"], ec="black", lw=1.1, z=3, h=15, fs=9.5):
    box(ax, x0, yc - h, x1, yc + h, fc=fc, ec=ec, lw=lw, r=3, z=z)
    if l2:
        text(ax, (x0 + x1) / 2, yc + 5.5, l1, fs=fs)
        text(ax, (x0 + x1) / 2, yc - 6.5, l2, fs=fs - 1.5, color=C["note"])
    else:
        text(ax, (x0 + x1) / 2, yc, l1, fs=fs)


def chain(y, label, kind, items):
    box(ax, 10, y - 17, 100, y + 17, fc=C[kind + "_fc"], ec=C[kind + "_ec"], lw=2.0, r=5)
    text(ax, 55, y, label, fs=12)
    text(ax, 115, y, ":=", fs=15)
    x = 134
    for i, (w, l1, l2, fc) in enumerate(items):
        small(x, x + w, y, l1, l2, fc=fc)
        if i < len(items) - 1:
            arrow(ax, [(x + w, y), (x + w + 12, y)], lw=1.0, head=5)
        x += w + 12


ylg = Y[4] - 62
chain(ylg, "eBPB Block", "ebpb", [(78, "1×1 Conv", None, C["conv_fc"]),
                                  (116, "3×3 DW RepConv", "(unfused)", C["gray_fc"]),
                                  (130, "eBPB: $\\mathcal{P}$, $\\psi$, $\\mathcal{U}$", "(4×4 avg-pool)", C["pool_fc"]),
                                  (80, "Channel gate", None, C["prcm_fc"])])  # D18 (pass 20): was "PRCM" (code name, net.py:41)
ylg -= 42
chain(ylg, "Basic Block", "basic", [(78, "1×1 Conv", None, C["conv_fc"]),
                                    (116, "3×3 DW RepConv", "(unfused)", C["gray_fc"]),
                                    (80, "Channel gate", None, C["prcm_fc"])])  # D18 (pass 20): was "PRCM" (code name, net.py:41)
ylg -= 42
chain(ylg, "Bottleneck", "bott", [(116, "7×7 DW Conv", "BN, ReLU", C["gray_fc"]),
                                  (96, "1×1 Conv", "BN, ReLU", C["conv_fc"])])
ny = ylg - 36
notes = [
    "$\\oplus$: element-wise sum; $\\ominus$: subtraction. Skip taps $e_i$ are taken after the 2×2 max-pool (4 pools, none after encoder 5).",
    "Each skip is added to the decoder output after its upsample ($s_4$: right after decoder 1). Channel gate: from global context on a learned basis.",
    "Dashed boxes: train-only heads $h_i$ (1×1 conv) on $s_1$–$s_3$, $\\mathcal{L}_S=\\frac{1}{3}\\Sigma_{i=1}^{3}(1-\\mathcal{S}(p_i,b_i))$, weight 0.1;",
    "deep-supervision heads are also train-only and omitted. Channels: encoder 128/192/256/384/512, decoder 384/256/192/128.",
]
for j, s in enumerate(notes):
    text(ax, 10, ny - 14 * j, s, fs=9.2, ha="left", color=C["note"])

save(fig, OUT)
print("saved", os.path.abspath(OUT), "canvas", W, H, "lowest text y", ny - 14 * (len(notes) - 1))
