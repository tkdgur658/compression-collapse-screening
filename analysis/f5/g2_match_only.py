# Manuscript-match check only (PREREG_F5.md sec. 4.2 assert), run while the sec. 6 synthetic check has NOT passed.
# Writes only the point-value comparison against Table tab:tiling_fives (already published values).
# No CI, no T classification, no T statement and no overlap sentence is written (sec. 6: no sec. 3-4 analysis before all R1-R4 pass).
# Usage: python -I g2_match_only.py G2_DIR TEX OUT.md
import os, sys
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f5_boot as fb
import f5_summary as fs


def main():
    g2, tex, out = sys.argv[1:4]
    models = {m: fb.load_model(os.path.join(g2, m))[0] for m in fs.G2_MODELS}
    res = fb.analyze(models)
    n_rows, bad = fs.manuscript_check(res, tex)
    table = fs.parse_tex_table(tex)
    L = ["# G2 recomputation: manuscript-match check only (PREREG_F5.md sec. 4.2)", "",
         "Status: the sec. 6 synthetic reverse-check FAILED (F5_SYNTH.md, 10/15). Under sec. 6 no sec. 3-4 analysis is run before all "
         "R1-R4 pass, so this file contains ONLY the point-value code check against the already published Table tab:tiling_fives. "
         "No bootstrap CI, T classification, T statement or overlap sentence was written or inspected.", "",
         f"Inputs: {g2}/{{ucmnet,utanet,maresunet}}/{{native,rs256,rs512,rs1024}}/results.json (read-only copies of gpu-server g2a). "
         f"Table source: {tex}.", "",
         f"Manuscript match ({n_rows} rows x peak, thin3 clDice, Dice, ratio, seam relative, 2 GiB, 0.5 GiB): "
         + ("**PASS**" if not bad else f"**FAIL** ({len(bad)} mismatches)"), ""]
    L += ["| model | variant | peak | thin3 clDice | Dice | ratio | seam | 2 GiB | 0.5 GiB |", "|---|---|---|---|---|---|---|---|---|"]
    for (m, vn), w in sorted(table.items()):
        L.append(f"| {m} | {vn} | {w['peak']} | {w['thin3']} | {w['dice']} | {w['ratio']} | {w['seam']} | {w['2 GiB']} | {w['0.5 GiB']} |")
    L += ["", "(values above are the manuscript cells; each was compared with the f5_boot point value formatted the same way)", ""]
    if bad: L += ["Mismatches:"] + ["- " + b for b in bad]
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("match:", "PASS" if not bad else "FAIL", len(bad))
    assert not bad, bad


if __name__ == "__main__":
    main()
