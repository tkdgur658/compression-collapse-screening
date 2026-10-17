"""Fig. 1 (teaser), Pass 32 ([internal decision] = [id], decision [id], 2026-10-16): collapse maps of settings D and O.

Run:  python -s tools/draw_fig1_collapse_map.py  ->  figures/fig1_collapse_map.pdf and figures/fig1_collapse_map.png

Left panel: setting D (deployed graph), label L80, rows = 11 models (PBE-Net first, then the order of
Table tab:main_polyp), columns = 6 datasets x 3 operators (w4, a6, l50). A cell is magenta (collapsed) when
"model|ds|D_op" is in RESULT_QDEPLOY.json table_i.L80.D.collapsed_cells, green (intact) otherwise, grey when no
checkpoint of that model and dataset entered the analysis (none expected).
Right panel: the same map for setting O (training graph, per-tensor scales), table_i.L80.O.collapsed_cells.
The rule flow of the Pass 32 prep version is now Fig. 2 (tools/draw_fig_protocol.py).

Data only from results/qdeploy/result/RESULT_QDEPLOY.json (sha256 a5d323e7..., equal to the server file
<WORK>/qdeploy/RESULT_QDEPLOY.json, checked 2026-10-16) and, for the number of splits per cell,
results/qdeploy/ITEMS.json (sha256 cc16bb78..., listed in FROZEN_SHA256SUMS) minus RESULT_QDEPLOY.json
"excluded" (malunet__isic18__k4). The script stops if a sha256 or a count differs.
Style helpers from tools/figlib.py.
"""
import os, sys, json, hashlib
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figlib import *

HERE = os.path.dirname(os.path.abspath(__file__))
QD = os.path.normpath(os.path.join(HERE, "..", "..", "results", "qdeploy"))
RES = os.path.join(QD, "result", "RESULT_QDEPLOY.json")
ITEMS = os.path.join(QD, "ITEMS.json")
SHA_RES = "a5d323e7c52fa375f2b1c69f3770f7393025e8a82286be9b37f8ac6a481f971e"
SHA_ITEMS = "1ba600df885eaec2bb366fd852d4e6aff6dd6b69274738b7b79f6166532c89df"
OUT = os.path.join(HERE, "..", "figures", "fig1_collapse_map.pdf")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


if sha(RES) != SHA_RES or sha(ITEMS) != SHA_ITEMS:
    sys.exit("sha256 mismatch: RESULT_QDEPLOY.json %s, ITEMS.json %s" % (sha(RES), sha(ITEMS)))
R = json.load(open(RES, encoding="utf-8"))
items = json.load(open(ITEMS, encoding="utf-8"))

MODELS = [("pbe", "PBE-Net"), ("maresunet", "MAResUNet"), ("utanet", "UTANet"), ("mk_unet", "MK-UNet"),
          ("mhorunet", "MHorUNet"), ("tinyunet", "TinyU-Net"), ("ucmnet", "UCMNet"), ("lbunet", "LB-UNet"),
          ("cmunext", "CMUNeXt"), ("egeunet", "EGE-UNet"), ("malunet", "MALUNet")]
DATASETS = [("PH2", "PH2"), ("isic17", "ISIC 2017"), ("isic18", "ISIC 2018"), ("kvasir", "Kvasir-SEG"),
            ("cvc", "CVC-ClinicDB"), ("etis", "ETIS-Larib")]
OPS = ["w4", "a6", "l50"]

# number of analysed splits per (model, dataset)
excluded = set(R["excluded"].keys())
nsplit = Counter((it["model"], it["ds"]) for it in items if it["tag"] not in excluded)
assert len(items) == R["n_items"] == 330 and sum(nsplit.values()) == R["n_used"] == 329, "item counts"
assert excluded == {"malunet__isic18__k4"} and nsplit[("malunet", "isic18")] == 4, "exclusion"
assert set(m for m, _ in nsplit) == set(m for m, _ in MODELS) and set(d for _, d in nsplit) == set(d for d, _ in DATASETS)


def grid(setting):
    cc = set(R["table_i"]["L80"][setting]["collapsed_cells"])
    known = set("%s|%s|%s_%s" % (m, d, setting, o) for m, _ in MODELS for d, _ in DATASETS for o in OPS)
    assert cc <= known, "unknown cell key"
    g = {}
    for m, _ in MODELS:
        for d, _ in DATASETS:
            for o in OPS:
                k = "%s|%s|%s_%s" % (m, d, setting, o)
                g[(m, d, o)] = "missing" if nsplit[(m, d)] == 0 else ("collapsed" if k in cc else "intact")
    return g


GD, GO = grid("D"), grid("O")
nD = sum(v == "collapsed" for v in GD.values()); nO = sum(v == "collapsed" for v in GO.values())
pD = sum(GD[("pbe", d, o)] == "collapsed" for d, _ in DATASETS for o in OPS)
pO = sum(GO[("pbe", d, o)] == "collapsed" for d, _ in DATASETS for o in OPS)
nmiss = sum(v == "missing" for v in GD.values())
assert len(GD) == 198 and nmiss == 0, "grid size / missing cells"
assert (nD, pD) == (80, 10) and R["table_i"]["L80"]["D"]["all"] == "80/198" and R["table_i"]["L80"]["D"]["pbe"] == "10/18"
assert (nO, pO) == (102, 18) and R["table_i"]["L80"]["O"]["all"] == "102/198" and R["table_i"]["L80"]["O"]["pbe"] == "18/18"
print("check: D %d/198 (PBE-Net %d/18), O %d/198 (PBE-Net %d/18), missing cells %d, MALUNet ISIC 2018 splits %d"
      % (nD, pD, nO, pO, nmiss, nsplit[("malunet", "isic18")]))


# ------------------------------------------------------------------ drawing (points, y up)
# Pass 32 ([internal decision], item 1): Fig. 1 = collapse maps only, settings D and O side by side; the
# R1 -> R2 flow strip of the Pass 32 prep version moved to Fig. 2 (tools/draw_fig_protocol.py).
COL = {"collapsed": "#C2187A", "intact": "#3A9A55", "missing": "#BDBDBD"}
FS, FL, FM = 17, 15.5, 14
CW, CH, GAP = 26, 17, 8            # cell width, cell height, gap between datasets
XL = 108                          # left edge of the first cell grid (row labels to its left)
GW = 18 * CW + 5 * GAP            # grid width
XL2 = XL + GW + 70                # left edge of the second grid (after the row counts of the first)
W, H = XL2 + GW + 52, 300
fig, ax = canvas(W, H)


def colx(x0, j, o):
    return x0 + j * (3 * CW + GAP) + o * CW


def panel(g, x0, top, title, labels):
    text(ax, x0, top, title, fs=FS, ha="left", va="center", weight="bold")
    yh = top - 22
    for j, (d, dn) in enumerate(DATASETS):
        text(ax, colx(x0, j, 0) + 1.5 * CW, yh, dn, fs=FM - 1.5, va="center")
        for o, op in enumerate(OPS):
            text(ax, colx(x0, j, o) + CW / 2, yh - 18, op, fs=FM - 1.5, va="center", color=C["note"])
    y0 = yh - 30
    for i, (m, mn) in enumerate(MODELS):
        y = y0 - (i + 1) * CH
        if labels:
            text(ax, x0 - 6, y + CH / 2, mn, fs=FM, ha="right", va="center", weight="bold" if m == "pbe" else "normal")
        for j, (d, _) in enumerate(DATASETS):
            for o, op in enumerate(OPS):
                ax.add_patch(Rectangle((colx(x0, j, o), y), CW, CH, fc=COL[g[(m, d, op)]], ec="white", lw=1.2, zorder=3))
        k = sum(g[(m, d, op)] == "collapsed" for d, _ in DATASETS for op in OPS)
        text(ax, x0 + GW + 6, y + CH / 2, "%d/18" % k, fs=FM, ha="left", va="center")
    ax.add_patch(Rectangle((x0 - 1, y0 - CH), GW + 2, CH, fc="none", ec="black", lw=1.4, zorder=4))  # PBE-Net row
    return y0 - len(MODELS) * CH


ybot = panel(GD, XL, H - 16, "Setting D, deployed graph: %d of 198 cells collapsed" % nD, True)
panel(GO, XL2, H - 16, "Setting O, training graph: %d of 198 cells collapsed" % nO, False)
yl = ybot - 22
for k, (lab, key) in enumerate([("collapsed (Dice retention below 0.8, L80)", "collapsed"), ("intact", "intact")]):
    xx = XL + k * 330
    ax.add_patch(Rectangle((xx, yl - 7), 16, 14, fc=COL[key], ec="white", zorder=3))
    text(ax, xx + 22, yl, lab, fs=FM, ha="left", va="center")

# figlib.save writes the full canvas; here the empty margin is cropped (bbox tight, 2 pt pad).
fig.savefig(OUT, format="pdf", bbox_inches="tight", pad_inches=2 / 72.0)
fig.savefig(OUT[:-4] + ".png", format="png", dpi=200, bbox_inches="tight", pad_inches=2 / 72.0)
print("saved", os.path.normpath(OUT), "and .png")
