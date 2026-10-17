# F5 synthetic reverse-check (PREREG_F5.md sec. 6, frozen sha 42dc2398...). CPU only, run once before any real analysis.
# Usage: python -I f5_synth.py OUT.md [REF_SERVER_DIR]
#   REF_SERVER_DIR holds unmodified copies of the server files scripts/g2a_train.py and scripts/g1.py (default ../ref_server).
# The seam metric is G2 eval_task/_band/_region_cl (g2a_train.py) with g1.py, imported unchanged.
import os, sys, json, hashlib, time
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(HERE, "..", "ref_server")
sys.path.insert(0, HERE)
sys.path.insert(0, REF)
import numpy as np
import cv2
cv2.setNumThreads(1)
from scipy import ndimage as ndi
from scipy import stats, optimize
import f5_boot as fb
import f5_summary as fs
import g1
import g2a_train as g2

DATA_SEED = 1            # synthetic data generator (separate from the frozen bootstrap seed 20261016)
N_IMG, N_SIM = 100, 200
CS = (0.70, 0.80, 1.00)
MEM_OK = {"peak_alloc_mib": 100.0}
S = g2.SIZE


def rec_scalar(x, seam=None):
    seam = seam or {str(g): {"band": 1.0, "n_band": 1, "interior": 1.0, "n_int": 1} for g in g2.GRIDS}
    return {"dice": x, "cldice": x, "thin3_cldice": x, "thin3_nskel": 1, "thin5_cldice": x, "thin5_nskel": 1, "seam": seam}


def rows_from(per):
    """per: {variant: list of records}; builds f5_boot rows (memory within both budgets, all finite)."""
    rows = {}
    for vn, lst in per.items():
        recs = [dict(r, idx=i) for i, r in enumerate(lst)]
        rows[vn] = {"recs": recs, "mem": dict(MEM_OK), "mode": "gpu", "nonfinite": 0, "src": "synthetic"}
    return rows


# ------------------------------------------------------------------ R1 -------------------------------------------
def true_ratio(c, sd=0.1):
    """Population ratio of medians: c * median(F * L) / median(F), F ~ Beta(5,5), L ~ lognormal(0, sd)."""
    z, w = np.polynomial.hermite_e.hermegauss(200); w = w / w.sum()
    cdf = lambda m: float(np.sum(w * stats.beta.cdf(m * np.exp(-sd * z), 5, 5)))
    mfl = optimize.brentq(lambda m: cdf(m) - 0.5, 1e-6, 5.0, xtol=1e-14)
    return c * mfl / stats.beta.median(5, 5)


def r1(rng):
    out = {}
    for c in CS:
        tr = true_ratio(c); cover = 0; cls = {"preserved": 0, "not preserved": 0, "indeterminate": 0}
        for _ in range(N_SIM):
            full = rng.beta(5, 5, N_IMG)
            var = c * full * rng.lognormal(0.0, 0.1, N_IMG)
            rows = rows_from({"full_fp32": [rec_scalar(float(x)) for x in full],
                              "tile_p256_o0_fp32": [rec_scalar(float(x)) for x in var]})
            res = fb.analyze({"syn": rows})
            t = res["models"]["syn"]["variants"]["tile_p256_o0_fp32"]["ratio"]
            cover += int(t[1] <= tr <= t[2])
            _, cl = fs.t_statement("syn", res["models"]["syn"])
            cls[cl["tile_p256_o0_fp32"]] += 1
        out[c] = {"true_ratio": tr, "coverage": cover / N_SIM, "classes": cls}
    return out


# ------------------------------------------------------------------ R2 -------------------------------------------
def vessel_map(rng, n_lines=60):
    """2048^2 map of random polylines, width 1-5 px (cv2 thickness), smooth random-walk vertices."""
    img = np.zeros((S, S), np.uint8); lines = []
    for _ in range(n_lines):
        w = int(rng.integers(1, 6)); nv = int(rng.integers(3, 7))
        p = rng.uniform(0, S, 2); ang = rng.uniform(0, 2 * np.pi); pts = [p.copy()]
        for _ in range(nv - 1):
            ang += rng.normal(0, 0.6); L = rng.uniform(150, 500)
            p = np.clip(p + L * np.array([np.cos(ang), np.sin(ang)]), 0, S - 1); pts.append(p.copy())
        P = np.round(np.array(pts)).astype(np.int32)                      # (x, y) for cv2
        cv2.polylines(img, [P.reshape(-1, 1, 2)], False, 1, thickness=w, lineType=cv2.LINE_8)
        lines.append((P, w))
    return img.astype(bool), lines


def seam_lines(g):
    return np.arange(g, S, g)


def gaps_on_seams(G, g=256, half=2):
    """(B): 4-px gap wherever a vessel crosses a seam line of grid g (columns/rows k-2..k+1 removed)."""
    cut = np.zeros((S, S), bool)
    for k in seam_lines(g):
        cut[:, k - half:k + half] = True; cut[k - half:k + half, :] = True
    P = G & ~cut
    n_gaps = int(ndi.label(G & cut, structure=np.ones((3, 3)))[1])
    return P, n_gaps


def gaps_off_seams(G, lines, n_gaps, rng, g=256, min_d=16):
    """(C): the same number of 4-px gaps, every removed pixel >= min_d px from every seam line of grid g."""
    k = seam_lines(g)
    far = lambda a0, a1: np.min(np.abs(np.arange(a0, a1)[:, None] - k[None, :])) >= min_d
    P = G.copy(); used = np.zeros((S, S), bool); placed = 0; tries = 0
    while placed < n_gaps and tries < 200000:
        tries += 1
        Pl, w = lines[int(rng.integers(len(lines)))]
        j = int(rng.integers(len(Pl) - 1)); a, b = Pl[j].astype(float), Pl[j + 1].astype(float)
        t = rng.uniform(0, 1); x, y = np.round(a + t * (b - a)).astype(int)
        dx, dy = abs(b[0] - a[0]), abs(b[1] - a[1]); e = w + 3
        if dx >= dy: y0, y1, x0, x1 = y - e, y + e, x - 2, x + 2               # mostly horizontal vessel: 4 columns cut
        else: y0, y1, x0, x1 = y - 2, y + 2, x - e, x + e                     # mostly vertical vessel: 4 rows cut
        if y0 < 0 or x0 < 0 or y1 > S or x1 > S: continue
        if not (far(y0, y1) and far(x0, x1)): continue
        if used[max(y0 - 8, 0):y1 + 8, max(x0 - 8, 0):x1 + 8].any(): continue
        win = P[y0:y1, x0:x1]
        if not win.any(): continue
        P[y0:y1, x0:x1] = False; used[y0:y1, x0:x1] = True; placed += 1
    return P, placed


def seam_eval(G, P, g=256):
    """G2 eval_task on (full = GT, variant = P); then f5_boot seam relative on the variant's own grid."""
    g2._W.clear(); g2._W["g1"] = g1; g2._W["msk"] = G[None]; g2._W["bands"] = {}
    _, o = g2.eval_task((0, {"full_fp32": np.packbits(G), f"tile_p{g}_o0_fp32": np.packbits(P)}))
    rows = rows_from({vn: [d] for vn, d in o.items()})
    res = fb.analyze({"syn": rows})
    v = res["models"]["syn"]["variants"][f"tile_p{g}_o0_fp32"]["seam"]
    fband = o["full_fp32"]["seam"][str(g)]["band"]; fint = o["full_fp32"]["seam"][str(g)]["interior"]
    return {"s": v["s"][0], "flag": v["flag"], "band": v["band"][0], "interior": v["interior"][0],
            "full_band": fband, "full_interior": fint, "thin3_nskel": o["full_fp32"]["thin3_nskel"]}


def r2(rng):
    G, lines = vessel_map(rng)
    A = seam_eval(G, G.copy())
    PB, nB = gaps_on_seams(G)
    Bq = seam_eval(G, PB)
    PC, nC = gaps_off_seams(G, lines, nB, rng)
    C = seam_eval(G, PC)
    return {"vessel_frac": float(G.mean()), "A": A, "B": dict(Bq, n_gaps=nB), "C": dict(C, n_gaps=nC)}


# ------------------------------------------------------------------ R3 -------------------------------------------
def seam_rec(vals):
    """vals: {grid: (band, interior)}"""
    return {str(g): {"band": float(b), "n_band": 1, "interior": float(i), "n_int": 1} for g, (b, i) in vals.items()}


def r3(rng):
    n = N_IMG
    full_t3 = rng.beta(5, 5, n)
    out = {}
    # case 1: whole-image ratio identical (same per-image thin3), seam differs
    it = rng.uniform(0.85, 0.95, (len(g2.GRIDS), n)); bd = it * rng.uniform(0.93, 0.99, it.shape)
    fseam = [{g: (bd[k, i], it[k, i]) for k, g in enumerate(g2.GRIDS)} for i in range(n)]
    t3 = full_t3 * 0.95 * rng.lognormal(0, 0.05, n)
    low = rng.uniform(0.80, 0.90, n)
    o0 = [dict(rec_scalar(float(t3[i])), seam=seam_rec({g: ((bd[k, i] * low[i]) if g == 256 else bd[k, i], it[k, i]) for k, g in enumerate(g2.GRIDS)})) for i in range(n)]
    o50 = [dict(rec_scalar(float(t3[i])), seam=seam_rec(fseam[i])) for i in range(n)]
    full = [dict(rec_scalar(float(full_t3[i])), seam=seam_rec(fseam[i])) for i in range(n)]
    res = fb.analyze({"syn": rows_from({"full_fp32": full, "tile_p256_o0_fp32": o0, "tile_p256_o50_fp32": o50})})
    out["case1"] = {"sentences": fs.overlap_sentences("syn", res["models"]["syn"])[:2], "contrast": res["models"]["syn"]["contrasts"]["256"]}
    # case 2: seam relative identical (same band/interior on both own grids, same full values on both grids), ratio differs
    it1 = rng.uniform(0.85, 0.95, n); bd1 = it1 * rng.uniform(0.93, 0.99, n); fac = rng.uniform(0.85, 0.95, n)
    fs2 = [seam_rec({g: (bd1[i], it1[i]) for g in g2.GRIDS}) for i in range(n)]
    vs2 = [seam_rec({g: (bd1[i] * fac[i], it1[i]) for g in g2.GRIDS}) for i in range(n)]
    t50 = full_t3 * 0.95 * rng.lognormal(0, 0.05, n); t0 = t50 * 1.10
    full = [dict(rec_scalar(float(full_t3[i])), seam=fs2[i]) for i in range(n)]
    o0 = [dict(rec_scalar(float(t0[i])), seam=vs2[i]) for i in range(n)]
    o50 = [dict(rec_scalar(float(t50[i])), seam=vs2[i]) for i in range(n)]
    res = fb.analyze({"syn": rows_from({"full_fp32": full, "tile_p256_o0_fp32": o0, "tile_p256_o50_fp32": o50})})
    out["case2"] = {"sentences": fs.overlap_sentences("syn", res["models"]["syn"])[:2], "contrast": res["models"]["syn"]["contrasts"]["256"]}
    return out


# ------------------------------------------------------------------ R4 -------------------------------------------
def r4(rng):
    full = [rec_scalar(float(x)) for x in rng.beta(5, 5, N_IMG)]
    res = fb.analyze({"syn": rows_from({"full_fp32": full, "tile_p256_o0_fp32": [dict(r) for r in full]})})
    return res["models"]["syn"]["variants"]["tile_p256_o0_fp32"]["ratio"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    out = sys.argv[1]
    t0 = time.time()
    rng = np.random.default_rng(DATA_SEED)
    R1 = r1(rng); R2 = r2(rng); R3 = r3(rng); R4 = r4(rng)
    L = ["# F5 synthetic reverse-check (PREREG_F5.md sec. 6)", "",
         f"Generated by f5_synth.py, {time.strftime('%Y-%m-%d %H:%M:%S')}, {time.time() - t0:.0f} s CPU wall. "
         f"Bootstrap B = {fb.B}, seed {fb.SEED} (frozen functions f5_boot.analyze / f5_summary.classify, t_statement, overlap_sentences). "
         f"Synthetic data generator numpy default_rng({DATA_SEED}).", "",
         "Code sha256: " + ", ".join(f"{os.path.basename(p)} {sha(p)}" for p in
                                    (os.path.join(HERE, "f5_boot.py"), os.path.join(HERE, "f5_summary.py"), os.path.abspath(__file__),
                                     os.path.join(REF, "g2a_train.py"), os.path.join(REF, "g1.py"))), "",
         "| check | criterion | value | result |", "|---|---|---|---|"]
    allp = []

    def add(name, crit, val, ok):
        allp.append(bool(ok)); L.append(f"| {name} | {crit} | {val} | {'PASS' if ok else 'FAIL'} |")

    for c in CS:
        d = R1[c]
        add(f"R1 c={c:.2f} coverage", "95% CI covers true ratio-of-medians in [0.90, 0.99] of 200 datasets",
            f"{d['coverage']:.3f} (true ratio {d['true_ratio']:.4f})", 0.90 <= d["coverage"] <= 0.99)
    d = R1[0.70]; add("R1 c=0.70 class", "\"not preserved\" ≥ 95%", f"{d['classes']['not preserved'] / N_SIM:.3f} {d['classes']}", d["classes"]["not preserved"] / N_SIM >= 0.95)
    d = R1[1.00]; add("R1 c=1.00 class", "\"preserved\" ≥ 95%", f"{d['classes']['preserved'] / N_SIM:.3f} {d['classes']}", d["classes"]["preserved"] / N_SIM >= 0.95)
    d = R1[0.80]; mode = max(d["classes"], key=lambda k: d["classes"][k])
    add("R1 c=0.80 class", "\"indeterminate\" is the mode", f"mode = {mode} {d['classes']}",
        d["classes"]["indeterminate"] > max(d["classes"]["preserved"], d["classes"]["not preserved"]))
    A, Bq, C = R2["A"], R2["B"], R2["C"]
    add("R2 (A) pred = GT", "s ∈ [0.98, 1.02]", f"s = {A['s']:.4f}", 0.98 <= A["s"] <= 1.02)
    add("R2 (B) band drop", "band clDice(v) < band clDice(full)", f"{Bq['band']:.4f} vs {Bq['full_band']:.4f} ({Bq['n_gaps']} gaps)", Bq["band"] < Bq["full_band"])
    add("R2 (B) interior", "|interior(v) − interior(full)| ≤ 0.01", f"{abs(Bq['interior'] - Bq['full_interior']):.4f}", abs(Bq["interior"] - Bq["full_interior"]) <= 0.01)
    add("R2 (B) s", "s < 0.8", f"s = {Bq['s']:.4f}", Bq["s"] < 0.8)
    add("R2 (B) flag", "seam break flag on", f"flag = {Bq['flag']}", Bq["flag"] is True)
    add("R2 (C) off-seam gaps", f"s ≥ 0.9 with the same number of gaps ≥ 16 px from all seams ({C['n_gaps']}/{Bq['n_gaps']} placed)",
        f"s = {C['s']:.4f} (interior {C['interior']:.4f} vs {C['full_interior']:.4f}; band {C['band']:.4f} vs {C['full_band']:.4f})",
        C["s"] >= 0.9 and C["n_gaps"] == Bq["n_gaps"])
    s1, s2 = R3["case1"]["sentences"], R3["case2"]["sentences"]
    j1, j2 = " ".join(s1), " ".join(s2)
    add("R3 case 1 (ratio equal, seam differs)", "\"raises seam continuity\" + \"does not differ detectably\"",
        "<br>".join(s1), "overlap raises seam continuity" in j1 and "does not differ detectably" in j1)
    add("R3 case 2 (seam equal, ratio differs)", "\"no detectable seam effect\" + direction + \"not attributed to overlap\"",
        "<br>".join(s2), "no detectable seam effect" in j2 and ("higher at 0% than at 50% overlap" in j2 or "higher at 50% than at 0% overlap" in j2)
        and "not attributed to overlap" in j2)
    add("R4 pairing", "variant = full → ratio CI = [1, 1] exactly", f"r = {R4[0]!r}, CI = [{R4[1]!r}, {R4[2]!r}]", R4[1] == 1.0 and R4[2] == 1.0)
    L += ["", f"**Overall: {'PASS' if all(allp) else 'FAIL'}** ({sum(allp)}/{len(allp)} criteria)", "",
          "R2 synthetic map: 60 random polylines (3-6 vertices, steps 150-500 px, cv2 thickness 1-5), "
          f"vessel fraction {R2['vessel_frac']:.4f}, GT thin3 skeleton {A['thin3_nskel']} px. Full-resolution reference = GT. "
          "Grid 256 (variant tile_p256_o0_fp32, own grid = stride 256). (B) removes columns/rows k-2..k+1 at every seam line k = 256, 512, ..., 1792; "
          "gap count = 8-connected components of GT within those strips. (C) cuts 4-px strips across the vessel at random polyline points, "
          "every removed pixel ≥ 16 px from all grid-256 seam lines.", ""]
    json.dump({"R1": {str(k): v for k, v in R1.items()}, "R2": R2, "R3": R3, "R4": R4, "pass": all(allp)},
              open(os.path.splitext(out)[0] + ".json", "w"), indent=1, default=float)
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
