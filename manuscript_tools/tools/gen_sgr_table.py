"""Pass 34 ([internal decision], 2026-10-17): Appendix table of the pre-registered R2 repair test (08_PREREG_SGR)
-> tables/tb_app_sgr.tex. Reads results/sgr/RESULT_SGR.json only (sha256 checked), recounts every
"cells repaired" value from cell_mean_R >= 0.8 and asserts it against the stored counts and the judge block.
Derived values (not stored as such in the json, computed here from the stored cell means):
  B1 on the 13 subset cells (main B1 cell means restricted to the subset cell keys),
  B1 at n = 100 on the six PBE-Net cells (main B1 cell means restricted to pbe|*).
Run:  python -s tools/gen_sgr_table.py
"""
import hashlib, io, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "results", "sgr", "RESULT_SGR.json"))
SHA = "d55a2476df3d1fa0602eecb848e9dd6b6eaf12ce3d1f022c62194e5fbd602247"
OUT = os.path.join(HERE, "..", "tables", "tb_app_sgr.tex")

raw = open(SRC, "rb").read()
assert hashlib.sha256(raw).hexdigest() == SHA, "RESULT_SGR.json sha256 mismatch"
d = json.loads(raw.decode("utf8"))
aux, judge = d["aux"], d["judge"]


def rep(cm):
    return sum(1 for v in cm.values() if v >= 0.8)


# recount and assert every stored cell count
for blk in ("main", "sub", "w4", "naux"):
    for arm, v in aux[blk].items():
        assert rep(v["cell_mean_R"]) == v["cells_repaired"], (blk, arm)
        assert len(v["cell_mean_R"]) == v["n_cells"], (blk, arm)
for arm in ("B0", "B1", "B2", "B3", "SGR"):
    assert judge["counts"][arm]["repaired"] == aux["main"][arm]["cells_repaired"]
for arm in ("A1", "A2", "A3", "B4", "SGR"):
    assert judge["counts"]["sub_" + arm]["repaired"] == aux["sub"][arm]["cells_repaired"]
assert d["fails"] == {"main": ["egeunet__isic18__k3"]} and d["missing"] == {}

sub_keys = sorted(aux["sub"]["B4"]["cell_mean_R"])
assert len(sub_keys) == 13
b1_main = aux["main"]["B1"]["cell_mean_R"]
b1_sub = sum(1 for k in sub_keys if b1_main[k] >= 0.8)
pbe_keys = sorted(k for k in b1_main if k.startswith("pbe|"))
assert len(pbe_keys) == 6 and pbe_keys == sorted(aux["naux"]["SGR"]["cell_mean_R"])
b1_pbe = sum(1 for k in pbe_keys if b1_main[k] >= 0.8)
# the subset SGR cells are the main SGR cells (same runs)
for k in sub_keys:
    assert abs(aux["sub"]["SGR"]["cell_mean_R"][k] - aux["main"]["SGR"]["cell_mean_R"][k]) < 1e-12
for k in pbe_keys:
    assert abs(aux["naux"]["SGR"]["cell_mean_R"][k] - aux["main"]["SGR"]["cell_mean_R"][k]) < 1e-12

flags = aux["rescreen_flags"]
tf = aux["tuned_fraction_range"]


def pct(x):
    p = 100.0 * x
    if p >= 99.95:
        return "%.3f" % p
    return ("%.2f" % p) if p < 1 else ("%.1f" % p)


def frac(arm):
    if arm not in tf:
        return "--"
    lo, hi = tf[arm]
    return "%s--%s" % (pct(lo), pct(hi))


def row(block, arm_key, label, desc, blk, flag_key=None, tf_key=None, cells=None, ckpt=None):
    if cells is None:
        v = aux[blk][arm_key]
        cells = "%d/%d" % (v["cells_repaired"], v["n_cells"])
        ckpt = "%d/%d" % (v["ckpt_R_ge_0.8"], v["n_ckpt_with_value"])
    fk = flag_key if flag_key is not None else arm_key
    fl = ("%d/%d" % (flags[fk]["flagged"], flags[fk]["n"])) if fk in flags else "--"
    return r" %s & %s & %s & %s & %s & %s & %s \\" % (block, label, desc, cells, ckpt, fl, frac(tf_key if tf_key is not None else arm_key))


R = []
R.append(row("Main", "B0", "B0", "No repair (pruned as in setting D)", "main"))
R.append(row("", "B1", "B1", "Output matching of all remaining weights, channel scales, and biases, mask fixed", "main"))
R.append(row("", "B2", "B2", "Global magnitude pruning, one threshold over all layers", "main"))
R.append(row("", "B3", "B3", "Channel-statistics correction, no gradient step", "main"))
R.append(row("", "SGR", "SGR", "Attribution, reallocation, matching in attributed layers", "main"))
R.append(r"\hline")
R.append(row("Subset", None, "B1", "As in the main block", None, flag_key="-", tf_key="-",
             cells="%d/13" % b1_sub, ckpt="--"))
R.append(row("", "B4", "B4", "B1 trained on the training masks (label bound)", "sub"))
R.append(row("", "SGR", "SGR", "As in the main block", "sub", flag_key="-", tf_key="-"))
R.append(row("", "A1", "A1", "SGR without reallocation", "sub"))
R.append(row("", "A2", "A2", "SGR without matching", "sub"))
R.append(row("", "A3", "A3", "SGR with randomly drawn layers", "sub"))
R.append(r"\hline")
R.append(row("4-bit", "B0w", "B0", "No repair (per-tensor 4-bit weights)", "w4"))
R.append(row("", "B1w", "B1", "Output matching of all latent weights, straight-through rounding", "w4"))
R.append(row("", "B3w", "B3", "Bias correction only", "w4"))
R.append(row("", "B5", "B5", "Per-channel 4-bit requantization, no training", "w4"))
R.append(row("", "SGRw", "SGR", "Attribution and matching of attributed tensors, no reallocation", "w4"))
R.append(r"\hline")
R.append(row("Images", "SGR_n10", "SGR", "$n=10$", "naux"))
R.append(row("", "SGR_n30", "SGR", "$n=30$", "naux"))
R.append(row("", "SGR", "SGR", "$n=100$ (main block)", "naux", flag_key="-", tf_key="-"))
R.append(row("", "B1_n10", "B1", "$n=10$", "naux"))
R.append(row("", "B1_n30", "B1", "$n=30$", "naux"))
R.append(row("", None, "B1", "$n=100$ (main block)", None, flag_key="-", tf_key="-",
             cells="%d/6" % b1_pbe, ckpt="--"))

tex = r"""%% GENERATED by tools/gen_sgr_table.py from results/sgr/RESULT_SGR.json (sha256 d55a2476...), do not edit by hand.
\begin{table*}[tp]
\centering
\caption{\label{tab:app_sgr}Pre-registered repair test of pruning collapse in setting D. Main: the 55 cells collapsed under layer-wise 50\%% pruning, all 11 models, 274 of 275 checkpoints with a value. Subset: the 13 main cells of PBE-Net, UTANet, and MAResUNet. 4-bit: the 15 cells of the same three models collapsed under per-tensor 4-bit weights (S-w4g), reported only. Images: the six PBE-Net cells with $n$ unlabeled training images per checkpoint ($n=100$ elsewhere). A cell is repaired when its Dice retention averaged over the splits reaches 0.8. Checkpoints count checkpoints with a retention of at least 0.8. Flags count checkpoints that the screen R1 still flags after repair, reported only. Tuned gives the range over checkpoints of the gradient-trained share of the inference parameters, in percent. A dash marks a value that the result file does not record for that row.}
\scriptsize
\begin{tblr}{colspec={Q[l] Q[l] X[3.2,l] Q[c] Q[c] Q[c] Q[c]}, width=\textwidth, colsep=3pt, rowsep=0pt}
\hline
 Block & Arm & Change & Cells repaired & Checkpoints & Flags & Tuned (\%%) \\
\hline
%s
\hline
\end{tblr}
\end{table*}
""" % "\n".join(R)
io.open(OUT, "w", encoding="utf8", newline="\n").write(tex)
print("wrote", os.path.normpath(OUT))
print("derived: B1 on subset %d/13, B1 n=100 on PBE-Net %d/6" % (b1_sub, b1_pbe))
print("tuned B1 %s, SGR %s" % (frac("B1"), frac("SGR")))
print("H:", json.dumps(judge["hyp"]))
