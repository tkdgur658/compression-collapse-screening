# F5 paired percentile bootstrap (PREREG_F5.md sec. 3, frozen sha 42dc2398...).
# Usage: python -I f5_boot.py OUT.json MODEL=DIR [MODEL=DIR ...]
#   DIR holds <variant>/results.json for variant in native, rs256, rs512, rs1024 (G2 results.json format, `per_image`).
# CPU only. No threads beyond 1. Writes nothing except OUT.json.
import os, sys, json
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
sys.dont_write_bytecode = True
import warnings
import numpy as np

B = 2000                                  # sec. 3
SEED = 20261016                           # sec. 3, numpy.random.default_rng
THR = 0.8                                 # sec. 3/4 ratio and seam thresholds
BUDGETS = {"2 GiB": 2048.0, "0.5 GiB": 512.0}
TRAIN_VARIANTS = ("native", "rs256", "rs512", "rs1024")
PATCHES, OVERLAPS = (256, 512), (0, 50)
FULL = "full_fp32"


# ------------------------------------------------------------------ loading --------------------------------------
def load_model(d):
    """Rows of one model, built the same way as g2a_summary.py: native per_image variants, memory-only variants,
    then resize variants. Returns {variant: {"recs": [...sorted by idx], "mem": {}, "mode": str, "nonfinite": int}},
    plus the list of training variants whose results.json is missing."""
    rows, missing = {}, []
    for tv in TRAIN_VARIANTS:
        p = os.path.join(d, tv, "results.json")
        if not os.path.exists(p):
            missing.append(tv); continue
        r = json.load(open(p))
        names = list(r["per_image"]) + [vn for vn in r["memory"] if vn not in r["per_image"]]
        for vn in names:
            recs = sorted(r["per_image"].get(vn, []), key=lambda x: x["idx"])
            rows[vn] = {"recs": recs, "mem": r["memory"].get(vn, {}), "mode": r.get("modes", {}).get(vn),
                        "nonfinite": int(r.get("nonfinite_images", {}).get(vn, 0) or 0), "src": tv}
    return rows, missing


def grid_of(vn):
    """Seam-grid stride of a tiled variant (G2 definition, O7): stride = patch * (1 - overlap)."""
    if vn.startswith("tile_"):
        p = int(vn.split("_p")[1].split("_")[0]); o = int(vn.split("_o")[1].split("_")[0])
        return p * (100 - o) // 100
    return None


def is_amp(vn):
    return vn.endswith("_amp")


# ------------------------------------------------------------------ image sets and index matrices -----------------
def image_sets(models):
    """All-image index list (sorted idx) and I3 = images with thin3_nskel > 0 in every variant of every model."""
    ref = None
    for m, rows in models.items():
        for vn, d in rows.items():
            if not d["recs"]: continue
            ids = [x["idx"] for x in d["recs"]]
            if ref is None: ref = ids
            assert ids == ref, f"image set differs: {m}/{vn}"
    assert ref is not None, "no per-image results"
    ok = np.ones(len(ref), bool)
    for m, rows in models.items():
        for vn, d in rows.items():
            if not d["recs"]: continue
            ok &= np.array([(x.get("thin3_nskel") or 0) > 0 for x in d["recs"]])
    i3 = [j for j in range(len(ref)) if ok[j]]
    return ref, i3


def make_index_matrices(n_all, n_i3, b=B, seed=SEED):
    """One generator; the all-image matrix is drawn first, then the I3 matrix (sec. 3). Shared by every model/variant."""
    rng = np.random.default_rng(seed)
    m_all = rng.integers(0, n_all, size=(b, n_all))
    m_i3 = rng.integers(0, n_i3, size=(b, n_i3))
    return m_all, m_i3


# ------------------------------------------------------------------ primitives ----------------------------------
def _f(x):
    return np.nan if x is None else float(x)


def vec(recs, key, pos=None):
    v = np.array([_f(x.get(key)) for x in recs], dtype=np.float64)
    return v if pos is None else v[pos]


def seam_vec(recs, g, part):
    return np.array([_f(x["seam"][str(g)][part]) for x in recs], dtype=np.float64)


def nanmed(x, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(x, axis=axis)


def boot_median(x, M):
    """Point median (None/NaN dropped, as g2a_summary.med) and its B replicate medians under index matrix M."""
    return float(nanmed(x)), nanmed(x[M], axis=1)


def ci(reps):
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return float(lo), float(hi)


def triple(pt, reps):
    lo, hi = ci(reps)
    return [pt, lo, hi]


# ------------------------------------------------------------------ analysis -------------------------------------
def analyze_model(rows, m_all, m_i3, i3):
    """Medians, ratio r(v), seam relative s(v), overlap contrasts for one model, all with paired percentile CIs."""
    out = {"variants": {}, "contrasts": {}}
    reps = {}                                        # variant -> {"t3": reps, "s": reps}
    full = rows.get(FULL)
    have_ref = bool(full and full["recs"])
    if have_ref:
        f_t3_pt, f_t3_r = boot_median(vec(full["recs"], "thin3_cldice", i3), m_i3)
        f_bi = {}
        for g in sorted({grid_of(vn) for vn in rows if grid_of(vn)}):
            fb_pt, fb_r = boot_median(seam_vec(full["recs"], g, "band"), m_all)
            fi_pt, fi_r = boot_median(seam_vec(full["recs"], g, "interior"), m_all)
            f_bi[g] = (fb_pt / fi_pt, fb_r / fi_r)
    for vn, d in rows.items():
        mem = d["mem"]; peak = mem.get("peak_alloc_mib"); oom = bool(mem.get("oom"))
        o = {"peak_alloc_mib": peak, "oom": oom, "mode": d["mode"], "nonfinite": d["nonfinite"],
             "n": len(d["recs"]), "exec": {b: (peak is not None and not oom and peak <= cap) for b, cap in BUDGETS.items()}}
        out["variants"][vn] = o
        if not d["recs"]:
            o["status"] = "no_results"; continue
        if is_amp(vn) and d["nonfinite"] > 0:          # sec. 3: "non-finite k/100", no ratio and no CI
            o["status"] = "nonfinite"; o["nonfinite_label"] = f"non-finite {d['nonfinite']}/{len(d['recs'])}"; continue
        o["status"] = "ok"
        recs = d["recs"]
        t3_pt, t3_r = boot_median(vec(recs, "thin3_cldice", i3), m_i3)
        med = {"thin3_cldice": triple(t3_pt, t3_r)}
        for k in ("cldice", "dice", "thin5_cldice"):
            pt, r = boot_median(vec(recs, k), m_all)
            med[k] = triple(pt, r)
        o["median"] = med
        reps[vn] = {"t3": t3_r}
        if have_ref:
            rr = t3_r / f_t3_r                       # numerator and denominator from the same resample
            o["ratio"] = triple(t3_pt / f_t3_pt, rr)
            reps[vn]["r"] = rr; reps[vn]["r_pt"] = t3_pt / f_t3_pt
        else:
            o["ratio"] = None
        g = grid_of(vn)
        if g is not None:
            b_pt, b_r = boot_median(seam_vec(recs, g, "band"), m_all)
            i_pt, i_r = boot_median(seam_vec(recs, g, "interior"), m_all)
            sm = {"grid": g, "band": triple(b_pt, b_r), "interior": triple(i_pt, i_r), "band_over_interior": triple(b_pt / i_pt, b_r / i_r)}
            if have_ref:
                fpt, frep = f_bi[g]
                s_pt = (b_pt / i_pt) / fpt; s_r = (b_r / i_r) / frep
                sm["full_band_over_interior"] = triple(fpt, frep)
                sm["s"] = triple(s_pt, s_r); sm["flag"] = bool(s_pt < THR)
                reps[vn]["s"] = s_r; reps[vn]["s_pt"] = s_pt
            o["seam"] = sm
    # overlap contrasts (fp32), sec. 3: delta = o50 - o0, paired (same replicate rows)
    for p in PATCHES:
        a, b = f"tile_p{p}_o0_fp32", f"tile_p{p}_o50_fp32"
        if a in reps and b in reps and "s" in reps[a] and "s" in reps[b]:
            out["contrasts"][str(p)] = {
                "dseam": triple(reps[b]["s_pt"] - reps[a]["s_pt"], reps[b]["s"] - reps[a]["s"]),
                "dratio": triple(reps[b]["r_pt"] - reps[a]["r_pt"], reps[b]["r"] - reps[a]["r"])}
    out["has_ref"] = have_ref
    return out


def analyze(models, b=B, seed=SEED):
    """models: {name: rows}. Index matrices are generated once (all -> I3) and shared by every model and variant."""
    ids, i3 = image_sets(models)
    m_all, m_i3 = make_index_matrices(len(ids), len(i3), b, seed)
    res = {"B": b, "seed": seed, "n_all": len(ids), "n_i3": len(i3), "i3_idx": [ids[j] for j in i3], "models": {}}
    for m, rows in models.items():
        res["models"][m] = analyze_model(rows, m_all, m_i3, i3)
    return res


def main():
    out = sys.argv[1]
    models, missing = {}, {}
    for a in sys.argv[2:]:
        m, d = a.split("=", 1)
        models[m], missing[m] = load_model(d)
    res = analyze(models)
    res["missing_inputs"] = missing
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
