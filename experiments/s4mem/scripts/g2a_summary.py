# G2 (FIVES) summarizer: reads g2a/<model>/<variant>/results.json, writes g2a/SUMMARY.md with the mechanical verdict.
# Usage: python -I g2a_summary.py [G2A_DIR]      (rules: G2_PREREG.md Part B)
import os as _ccs_os  # path configuration of the released copy (README, Configuration)
CCS_REPO_ROOT = _ccs_os.path.normpath(_ccs_os.path.join(_ccs_os.path.dirname(_ccs_os.path.abspath(__file__)), '../../..'))
CCS_WORK_ROOT = _ccs_os.environ.get('CCS_WORK', _ccs_os.path.join(CCS_REPO_ROOT, 'experiments'))
import os, sys, json, glob
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else _ccs_os.path.join(CCS_WORK_ROOT, 's4mem/g2a')
MODELS = ["ucmnet", "utanet", "maresunet"]
VARIANTS = ["native", "rs256", "rs512", "rs1024"]
BUDGETS = {"2 GiB": 2048.0, "0.5 GiB": 512.0}
RATIO = 0.8


def med(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


def fmt(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}"


def load():
    R = {}
    for m in MODELS:
        for v in VARIANTS:
            p = f"{OUT}/{m}/{v}/results.json"
            if os.path.exists(p):
                R[(m, v)] = json.load(open(p))
    return R


def grid_of(vn):
    # seam-grid stride of a tiled variant: stride = patch*(1-overlap)
    if vn.startswith("tile_"):
        p = int(vn.split("_p")[1].split("_")[0]); o = int(vn.split("_o")[1].split("_")[0])
        return p * (100 - o) // 100
    return None


def main():
    R = load()
    L = ["# G2 summary (FIVES, memory axis)", "", f"Rules: G2_PREREG.md Part B. Generated from {OUT}. Ratio threshold {RATIO}.", ""]
    rows = {}   # (model, variant_name) -> dict
    for m in MODELS:
        nat = R.get((m, "native"))
        if nat:
            for vn, lst in nat["per_image"].items():
                rows[(m, vn)] = {"res": lst, "mem": nat["memory"].get(vn, {}), "mode": nat["modes"].get(vn)}
            for vn in nat["memory"]:
                if (m, vn) not in rows:
                    rows[(m, vn)] = {"res": [], "mem": nat["memory"][vn], "mode": nat["modes"].get(vn)}
        for rv in VARIANTS[1:]:
            r = R.get((m, rv))
            if r:
                for vn in r["memory"]:
                    rows[(m, vn)] = {"res": r["per_image"].get(vn, []), "mem": r["memory"][vn], "mode": r["modes"].get(vn)}
    missing = [f"{m}/{v}" for m in MODELS for v in VARIANTS if (m, v) not in R]
    complete = not missing
    L.append(f"Jobs present: {len(R)}/{len(MODELS) * len(VARIANTS)}" + ("" if complete else f"  (missing: {', '.join(missing)})"))
    L.append("")

    # ---- table
    L += ["## Per-variant results (median over the 100 validation images; absolute values first)", "",
          "| model | variant | peak MiB (alloc, batch 1) | thin3 clDice | clDice | Dice | thin3 / full_fp32 | exec 2 GiB | exec 0.5 GiB | mode |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    ref = {}
    for m in MODELS:
        r = rows.get((m, "full_fp32"))
        ref[m] = med([d["thin3_cldice"] for d in r["res"]]) if r and r["res"] else None
    table = {}
    for (m, vn), d in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        t3 = med([x["thin3_cldice"] for x in d["res"]]); cl = med([x["cldice"] for x in d["res"]]); dc = med([x["dice"] for x in d["res"]])
        peak = d["mem"].get("peak_alloc_mib"); oom = bool(d["mem"].get("oom"))
        ratio = (t3 / ref[m]) if (t3 is not None and ref[m]) else None
        ex = {b: (peak is not None and not oom and peak <= cap) for b, cap in BUDGETS.items()}
        table[(m, vn)] = {"t3": t3, "ratio": ratio, "peak": peak, "oom": oom, "exec": ex, "has_res": bool(d["res"])}
        L.append(f"| {m} | {vn} | {'OOM (>24 GiB)' if oom else fmt(peak, 0)} | {fmt(t3)} | {fmt(cl)} | {fmt(dc)} | {fmt(ratio)} | "
                 f"{'yes' if ex['2 GiB'] else 'no'} | {'yes' if ex['0.5 GiB'] else 'no'} | {d['mode']} |")
    L.append("")

    # ---- C rule
    L += ["## C rule (mechanical)", "",
          f"C holds at budget B for a model iff EVERY S variant executable within B (peak allocated GPU memory at batch 1 <= B) has "
          f"thin3 clDice < {RATIO} x that of full_fp32 (unlimited full-resolution inference) of the same model. "
          f"C holds overall at B iff it holds for every model. Holding at 2 GiB implies holding at 0.5 GiB (subset).", ""]
    verdict = {}
    for b in BUDGETS:
        per_model = {}
        for m in MODELS:
            if ref[m] is None:
                per_model[m] = ("NO_REF", []); continue
            ex = [(vn, table[(m, vn)]) for (mm, vn) in table if mm == m and table[(m, vn)]["exec"][b]]
            ex = [(vn, t) for vn, t in ex if t["ratio"] is not None]
            bad = [(vn, t["ratio"]) for vn, t in ex if t["ratio"] >= RATIO]
            if not ex: per_model[m] = ("VACUOUS", [])
            elif bad: per_model[m] = ("FAILS", bad)
            else: per_model[m] = ("HOLDS", [])
        verdict[b] = per_model
        L.append(f"### Budget {b}")
        for m in MODELS:
            st, bad = per_model[m]
            ex_names = [vn for (mm, vn) in table if mm == m and table[(m, vn)]["exec"][b]]
            L.append(f"- {m}: C {st}; executable variants: {', '.join(ex_names) if ex_names else 'none'}"
                     + (f"; variants with ratio >= {RATIO}: " + ", ".join(f"{vn} ({r:.2f})" for vn, r in bad) if bad else ""))
        L.append("")
    def overall(b):
        sts = [verdict[b][m][0] for m in MODELS]
        if any(s == "NO_REF" for s in sts): return "INCOMPLETE (no full-resolution reference for some model)"
        if any(s == "FAILS" for s in sts): return "C FAILS (memory axis closes on FIVES)"
        if any(s == "VACUOUS" for s in sts): return "C HOLDS but VACUOUS for some model (no S variant executable) - flag"
        return "C HOLDS"
    L += ["## Verdict", ""]
    for b in BUDGETS:
        L.append(f"- Budget {b}: **{overall(b)}**")
    L.append(f"- Jobs complete: {'yes' if complete else 'NO - verdict is provisional (' + ', '.join(missing) + ')'}")
    L.append("")
    # 50% overlap rule
    L += ["## 50%-overlap tiling vs full-resolution (memory-independent), thin3 ratio", "",
          "| model | variant | ratio | peak MiB | exec 2 GiB | exec 0.5 GiB |", "|---|---|---|---|---|---|"]
    ge = []
    for (m, vn), t in sorted(table.items()):
        if vn.startswith("tile_") and "_o50_" in vn:
            L.append(f"| {m} | {vn} | {fmt(t['ratio'])} | {fmt(t['peak'], 0)} | {'yes' if t['exec']['2 GiB'] else 'no'} | {'yes' if t['exec']['0.5 GiB'] else 'no'} |")
            if t["ratio"] is not None and t["ratio"] >= RATIO and t["exec"]["2 GiB"]: ge.append((m, vn))
    L += ["", f"50%-overlap tiling executable within 2 GiB with ratio >= {RATIO}: {ge if ge else 'none'}  ->  "
              + ("memory axis closes on FIVES" if ge else "no closure from this line"), ""]
    # robustness: excluding 512 patches
    rob = {}
    for b in BUDGETS:
        bad = [(m, vn) for (m, vn), t in table.items() if t["exec"][b] and t["ratio"] is not None and t["ratio"] >= RATIO and "_p512_" not in vn]
        rob[b] = bad
    L += ["Robustness (descriptive): offending executable variants excluding patch-512 (512 uses weights trained on 256 crops): "
          + "; ".join(f"{b}: {rob[b] if rob[b] else 'none'}" for b in BUDGETS), ""]

    # ---- seam
    L += ["## Tile-seam connectivity (clDice in +-8 px band around tile seams vs tile interiors; medians over images)", "",
          "| model | variant | band clDice | interior clDice | band/interior | same grid, full_fp32 band/interior | relative | seam break (<0.8) |",
          "|---|---|---|---|---|---|---|---|"]
    breaks = []
    for (m, vn), d in sorted(rows.items()):
        g = grid_of(vn)
        if g is None or not d["res"]: continue
        fr = rows.get((m, "full_fp32"))
        def bi(res):
            b = med([x["seam"][str(g)]["band"] for x in res]); i = med([x["seam"][str(g)]["interior"] for x in res])
            return b, i, (b / i if (b is not None and i) else None)
        b, i, r = bi(d["res"])
        fb = bi(fr["res"])[2] if fr and fr["res"] else None
        rel = (r / fb) if (r is not None and fb) else None
        brk = rel is not None and rel < 0.8
        if brk: breaks.append((m, vn))
        L.append(f"| {m} | {vn} | {fmt(b)} | {fmt(i)} | {fmt(r)} | {fmt(fb)} | {fmt(rel)} | {'YES' if brk else 'no'} |")
    L += ["", f"Seam breaks flagged: {breaks if breaks else 'none'} (descriptive; the verdict above is the thin3 ratio rule only).", ""]

    # ---- OOM boundary
    L += ["## Whole-image forward: memory vs side (batch 1, MiB allocated; OOM = does not fit 24 GiB)", "",
          "| model | prec | " + " | ".join(str(s) for s in range(256, 2049, 256)) + " |", "|---|---|" + "---|" * 8]
    for m in MODELS:
        nat = R.get((m, "native"))
        if not nat: continue
        for prec, pr in nat.get("probes", {}).items():
            cells = []
            for s in range(256, 2049, 256):
                c = pr.get(str(s), {})
                cells.append("OOM" if c.get("oom") else fmt(c.get("peak_alloc_mib"), 0))
            L.append(f"| {m} | {prec} | " + " | ".join(cells) + " |")
    L.append("")
    # ---- training
    L += ["## Training", "", "| model | variant | params | train min | best val Dice (own res) @ step | skipped updates |", "|---|---|---|---|---|---|"]
    for (m, v), r in sorted(R.items()):
        L.append(f"| {m} | {v} | {r['n_params']:,} | {r['train_seconds'] / 60:.1f} | {fmt(r.get('best_val_dice'), 4)} @ {r.get('best_step')} | {r.get('skipped_nonfinite_updates')} |")
    nf = [(m, v, r.get("nonfinite_images")) for (m, v), r in sorted(R.items()) if r.get("nonfinite_images") and any(r["nonfinite_images"].values())]
    L += ["", f"Images with non-finite outputs (AMP/others): {nf if nf else 'none'}", ""]
    open(OUT + "/SUMMARY.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L[-40:]))


if __name__ == "__main__":
    main()
