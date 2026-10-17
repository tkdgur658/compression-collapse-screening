"""Pass 33 (duplicate audit, [internal decision], 2026-10-17): appendix tables of the duplicate-image audit and
Table 1 recomputed without the duplicated test images.

Inputs (read only):
  review2/DUP_AUDIT_P33.json   output of tools/dup_audit_server.py, run on gpu-server (CPU, nice 10, 2 threads) over the
                               protocol-v1 caches <ARCHIVE>/cache/v1/<ds>/ and the split folders
  ../data/main/<model>/<ds>/k<split>/results.json   test_per_image_iou and test_names (local copy of the Table 1 runs)
Outputs:
  tables/tb_app_dup.tex          tab:app_dup_audit (counts by dataset, pair type and split) and tab:app_dup_groups
  tables/tb_app_table1_dedup.tex tab:app_table1_dedup (Table 1 without the duplicated test images)
  review2/TABLE1_DEDUP_DIFF.md   every cell of both tables and every change of a significance mark
The pre-registered Table 1 (tables/tb_main.tex, tools/gen_table1.py) is not touched.
Duplicated test image = test image of split k whose cache array is byte-identical to a train or val image of split k.
Test and Holm correction: tools/sig_table1.py (holm(), ROWS, DS, ALPHA imported unchanged).
Run:  python -s tools/gen_dup_tables.py
"""
import glob
import io
import json
import os
import sys

import numpy as np
from scipy import stats

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sig_table1 as S

ROOT = os.path.abspath(os.path.join(HERE, ".."))
AUDIT = os.path.join(ROOT, "review2", "DUP_AUDIT_P33.json")
RUNS = S.RUNS
OUT_AUD = os.path.join(ROOT, "tables", "tb_app_dup.tex")
OUT_T1 = os.path.join(ROOT, "tables", "tb_app_table1_dedup.tex")
OUT_MD = os.path.join(ROOT, "review2", "TABLE1_DEDUP_DIFF.md")
DSN = {"PH2": "PH2", "isic17": "ISIC 2017", "isic18": "ISIC 2018", "kvasir": "Kvasir-SEG", "cvc": "CVC-ClinicDB", "etis": "ETIS-Larib"}
NAME = dict(S.ROWS); NAME["pbe"] = "PBE-Net"; NAME["egeunet"] = "EGE-UNet"
SETN = {"train": "tr", "val": "va", "test": "te"}
SPLITS = S.SPLITS
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


def load_audit():
    A = json.load(io.open(AUDIT, encoding="utf8"))
    for ds, R in A["datasets"].items():
        assert R["union_same_names_all_splits"] and R["union_same_content_all_splits"], ds
        for k, d in R["splits"].items():
            # the three hash levels agree, and no file name is shared between train, val and test of a split
            assert d["cache"] == d["file"] == d["pixel"], (ds, k)
            assert not any(d["name_overlap"].values()), (ds, k)
    return A


def per_image():
    """{(model, ds, k): (names, per-image IoU array, data_cache dict)} from the Table 1 result files"""
    v = {}
    for f in glob.glob(os.path.join(RUNS, "*", "*", "k*", "results.json")):
        r = json.load(open(f))
        if r.get("status") != "complete":
            continue
        row = f.replace("\\", "/").split("/")[-4]
        key = (row, r["config"]["dataset"], r["config"]["split"])
        iou = np.array(r["test_per_image_iou"], float)
        assert len(iou) == len(r["test_names"]) and abs(iou.mean() - r["test"]["iou"]) < 1e-9, f
        v[key] = (list(r["test_names"]), iou, r["data_cache"])
    return v


def cells_from(v, drop):
    cells = {}
    for r, _ in S.ROWS:
        for d, _ in S.DS:
            x = []
            for k in SPLITS:
                names, iou, _ = v[(r, d, k)]
                keep = np.array([n not in drop.get((d, k), ()) for n in names])
                x.append(iou[keep].mean())
            cells[(r, d)] = np.array(x, float)
    return cells


def tests(cells):
    res = {}
    for d, _ in S.DS:
        others = [r for r, _ in S.ROWS if r != "pbe"]
        p = [stats.ttest_rel(cells[("pbe", d)], cells[(r, d)]).pvalue for r in others]
        pa = S.holm(p)
        for r, pr, pp in zip(others, p, pa):
            diff = cells[("pbe", d)].mean() - cells[(r, d)].mean()
            res[(r, d)] = dict(diff=diff, p=float(pr), padj=float(pp),
                               mark=(("down" if diff > 0 else "up") if pp < S.ALPHA else ""))
    return res


def th(x):
    """thousands separator as in tables/tab_settings.tex"""
    return "{:,}".format(x).replace(",", "{,}")


def _join(xs):
    xs = list(xs)
    return " and ".join(xs) if len(xs) == 2 else ", ".join(xs)


def fmt_cell(m, s, bold, mark):
    txt = "%.3f $\\pm$ %.3f" % (m, s)
    if bold:
        txt = "\\textbf{%s}" % txt
    sup = {"down": "$^{\\blacktriangledown}$", "up": "$^{\\blacktriangle}$", "": "\\phantom{$^{\\blacktriangledown}$}"}[mark]
    return "{%s%s}" % (txt, sup)


def main():
    A = load_audit()
    v = per_image()
    # the result files point at the audited cache files, and their test names are the cache test names
    for (r, d, k), (names, _, dc) in v.items():
        sp = A["datasets"][d]["splits"][str(k)]
        assert dc == sp["cache_files"], (r, d, k)
        assert len(names) == sp["n"]["test"], (r, d, k)
    drop = {(d, int(k)): set(sp["test_removed"]) for d, R in A["datasets"].items() for k, sp in R["splits"].items()}
    n_drop = sum(len(x) for x in drop.values())
    n_test = sum(sp["n"]["test"] for R in A["datasets"].values() for sp in R["splits"].values())

    # ---------- audit tables ----------
    pairs = [("train_val", "Train--val"), ("train_test", "Train--test"), ("val_test", "Val--test"),
             ("within_train", "In train"), ("within_val", "In val"), ("within_test", "In test")]
    rows = []
    for d, _ in S.DS:
        R = A["datasets"][d]
        sp = [R["splits"][str(k)] for k in SPLITS]
        n = sp[0]["n"]
        cells = ["/".join(str(s["cache"][p]) for s in sp) for p, _ in pairs]
        rows.append(" & ".join([DSN[d],
                                "%s (%s)" % (th(R["n_union"]), th(R["union_n_distinct_cache"]))] + cells) + r" \\")
    grows = []
    for d, _ in S.DS:
        R = A["datasets"][d]
        for g in R["union_dup_groups_cache"]:
            place, iou = [], None
            for k in SPLITS:
                for x in R["splits"][str(k)]["dup_groups_mask_iou"]:
                    mem = {n: s for s, n in x["members"]}
                    if set(mem) == set(g):
                        place.append("/".join(SETN[mem[n]] for n in g))
                        iou = x["mask_iou"] if iou is None else iou
                        assert iou == x["mask_iou"]
            assert len(place) == 5
            grows.append(" & ".join([DSN[d], ", ".join(n.replace("_", "\\_") for n in g), "%.3f" % iou,
                                     ] + place) + r" \\")
    tex = ["% GENERATED by tools/gen_dup_tables.py from review2/DUP_AUDIT_P33.json (tools/dup_audit_server.py on gpu-server), do not edit by hand.",
           r"""\begin{table*}[tp]
\centering
\caption{\label{tab:app_dup_audit}Byte-identical images in the protocol-v1 splits. Two images count as identical when the sha256 hashes of their $256\times256$ uint8 cache arrays agree. These arrays are the input of training and evaluation. Hashes of the original files and of their decoded pixel arrays give the same counts in every cell. A pair count is the number of images of the second set with an identical image in the first set of the same split. ``In'' columns count extra copies within one set. Each cell lists splits 1 to 5. Set sizes are given in Table~ef{tab:settings}. No file name is shared between the sets of a split.}
\scriptsize
\begin{tblr}{colspec={Q[l] Q[c] Q[c] Q[c] Q[c] Q[c] Q[c] Q[c] Q[c]}, colsep=2.5pt, rowsep=0pt}
\hline
 Dataset & Images (distinct) & """ + " & ".join(h for _, h in pairs) + r""" \\
\hline
""" + "\n".join(" " + r for r in rows) + r"""
\hline
\end{tblr}
\end{table*}
""",
           r"""\begin{table*}[tp]
\centering
\caption{\label{tab:app_dup_groups}The %s groups of byte-identical images behind Table~\ref{tab:app_dup_audit}. Each group holds two file names with identical image content but different masks. Mask IoU is the IoU between the two cache masks. Columns 1 to 5 give the sets of the two files in each split, in the order of the file names (tr training, va validation, te test).}
\scriptsize
\begin{tblr}{colspec={Q[l] Q[l] Q[c] Q[c] Q[c] Q[c] Q[c] Q[c]}, colsep=3pt, rowsep=0pt}
\hline
 Dataset & Files & Mask IoU & 1 & 2 & 3 & 4 & 5 \\
\hline
""" % WORDS[len(grows)] +"\n".join(" " + r for r in grows) + r"""
\hline
\end{tblr}
\end{table*}
"""]
    io.open(OUT_AUD, "w", encoding="utf8", newline="\n").write("\n".join(tex))

    # ---------- Table 1 without duplicated test images ----------
    c0, r0 = S.compute()  # pre-registered Table 1 (unchanged source)
    c0b = cells_from(v, {})
    assert all(np.allclose(c0[x], c0b[x], atol=1e-9) for x in c0)
    c1 = cells_from(v, drop)
    r1 = tests(c1)
    assert all(r0[x]["mark"] == tests(c0)[x]["mark"] for x in r0)
    best = {d: max(c1[(r, d)].mean() for r, _ in S.ROWS) for d, _ in S.DS}
    body = []
    for r, _ in S.ROWS:
        cs = []
        for d, _ in S.DS:
            m, s = c1[(r, d)].mean(), c1[(r, d)].std(ddof=1)
            cs.append(fmt_cell(m, s, abs(m - best[d]) < 1e-12, "" if r == "pbe" else r1[(r, d)]["mark"]))
        body.append("%s & %s \\\\" % (NAME[r], " & ".join(cs)))
    nsig0 = sum(x["mark"] != "" for x in r0.values()); nsig1 = sum(x["mark"] != "" for x in r1.values())
    changed = [(x, r0[x]["mark"], r1[x]["mark"]) for x in r0 if r0[x]["mark"] != r1[x]["mark"]]
    bold0 = {d: max((c0[(r, d)].mean(), r) for r, _ in S.ROWS)[1] for d, _ in S.DS}
    bold1 = {d: max((c1[(r, d)].mean(), r) for r, _ in S.ROWS)[1] for d, _ in S.DS}
    removed_txt = _join("%s split %d" % (DSN[d], k) for (d, k), x in sorted(drop.items(), key=lambda t: ([dd for dd, _ in S.DS].index(t[0][0]), t[0][1])) if x)
    t1 = "%% GENERATED by tools/gen_dup_tables.py (per-image IoU of <workspace>/data/main results.json, duplicates from review2/DUP_AUDIT_P33.json, test and Holm of tools/sig_table1.py), do not edit by hand.\n" + r"""\begin{table*}[tp]
\centering
\caption{\label{tab:app_table1_dedup}Table~\ref{tab:main_polyp} repeated without the %s test images that are byte-identical to a training image of the same split (one each in %s, Table~\ref{tab:app_dup_groups}). No other split loses an image. The models, the test, and the mark definitions follow Table~\ref{tab:main_polyp}. Significance marks after Holm correction: %d of 60 here, %d of 60 in Table~\ref{tab:main_polyp}.}
\small
\begin{tblr}{colspec={ X[c,1] X[c,1] X[c,1] X[c,1] X[c,1] X[c,1] X[c,1] }, width=\textwidth, colsep=0.5pt}
\hline
 Model & PH2 & ISIC 2017 & ISIC 2018 & Kvasir-SEG & CVC-ClinicDB & ETIS-Larib \\
\hline
""" % (WORDS.get(n_drop, n_drop), removed_txt, nsig1, nsig0) + "\n".join(body) + r"""
\hline
\end{tblr}
\end{table*}
"""
    io.open(OUT_T1, "w", encoding="utf8", newline="\n").write(t1)

    # ---------- diff report ----------
    L = ["# TABLE1_DEDUP_DIFF (generated by `tools/gen_dup_tables.py`, Pass 33 duplicate audit)\n",
         "Source: per-image test IoU (`test_per_image_iou`, `test_names`) of `<workspace>/data/main/*/*/k*/results.json`; all 330 result files point at the audited cache files (`data_cache` equal to `review2/DUP_AUDIT_P33.json`). "
         "Removed: test images of split k whose cache array is byte-identical to a train or val image of split k: %d of %d test images (%s). "
         "Test: paired t-test over the five splits, Holm over ten baselines per dataset, alpha 0.05 (`tools/sig_table1.py`).\n" % (n_drop, n_test, removed_txt),
         "Removed images: " + "; ".join("%s k%d: %s" % (DSN[d], k, ", ".join(sorted(x))) for (d, k), x in sorted(drop.items()) if x) + "\n",
         "Significant after Holm: %d of 60 (Table 1) -> %d of 60 (without duplicates). Changed marks: %d. Bold (highest mean) changed: %s.\n"
         % (nsig0, nsig1, len(changed), [d for d in bold0 if bold0[d] != bold1[d]] or "none"),
         "| Model | Dataset | Table 1 mean | dedup mean | Table 1 Holm p | dedup Holm p | Table 1 mark | dedup mark |", "|---|---|---|---|---|---|---|---|"]
    for r, _ in S.ROWS:
        for d, _ in S.DS:
            a, b = c0[(r, d)].mean(), c1[(r, d)].mean()
            if r == "pbe":
                L.append("| %s | %s | %.4f | %.4f | - | - | - | - |" % (NAME[r], DSN[d], a, b))
            else:
                L.append("| %s | %s | %.4f | %.4f | %.4f | %.4f | %s | %s |" % (NAME[r], DSN[d], a, b, r0[(r, d)]["padj"], r1[(r, d)]["padj"],
                                                                         r0[(r, d)]["mark"] or "-", r1[(r, d)]["mark"] or "-"))
    L.append("\n## Changed significance marks\n")
    L += ["- %s on %s: %s -> %s" % (NAME[r], DSN[d], a or "none", b or "none") for (r, d), a, b in changed] or ["None."]
    io.open(OUT_MD, "w", encoding="utf8", newline="\n").write("\n".join(L) + "\n")
    print("removed", n_drop, "of", n_test, "| significant", nsig0, "->", nsig1, "| changed marks", changed)
    print("max |mean change| %.4f" % max(abs(c0[x].mean() - c1[x].mean()) for x in c0))


if __name__ == "__main__":
    main()
