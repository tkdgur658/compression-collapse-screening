# F5 summary (PREREG_F5.md sec. 4 and sec. 7, frozen sha 42dc2398...). Mechanical strings only.
# Usage (server):  python -I f5_summary.py --g2-dir <WORK>/s4mem/g2a --pbe-dir <WORK>/s4mem/f5 \
#                    --tex 4f_analysis_tiling.tex --out F5_SUMMARY.md --boot-json F5_BOOT.json [--amend0 AMEND0_SHA.json] [--sha-paths P.json]
# Without --pbe-dir only the sec. 4.2 recomputation of the three G2 models is written.
import os as _ccs_os  # path configuration of the released copy (README, Configuration)
CCS_REPO_ROOT = _ccs_os.path.normpath(_ccs_os.path.join(_ccs_os.path.dirname(_ccs_os.path.abspath(__file__)), '../..'))
CCS_WORK_ROOT = _ccs_os.environ.get('CCS_WORK', _ccs_os.path.join(CCS_REPO_ROOT, 'experiments'))
CCS_ARCHIVE_ROOT = _ccs_os.environ.get('CCS_ARCHIVE', _ccs_os.path.join(CCS_REPO_ROOT, 'archive'))
import os, sys, json, re, hashlib, argparse
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f5_boot as fb

BASE = _ccs_os.path.join(CCS_WORK_ROOT, 's4mem')
CODE_V2 = _ccs_os.path.join(CCS_ARCHIVE_ROOT, 'code_v2')
G2_MODELS = ["ucmnet", "utanet", "maresunet"]
DISPLAY = {"pbe": "PBE-Net", "ucmnet": "UCMNet", "utanet": "UTANet", "maresunet": "MAResUNet"}
PBE_VARIANTS = fb.TRAIN_VARIANTS
THR = fb.THR
E_BUDGET = "0.5 GiB"

# sec. 7 table: logical name -> (default server path, expected sha256 prefix; None = recorded by AMENDMENT 0)
SHA_TABLE = [
    ("scripts/f5_train.py", BASE + "/scripts/f5_train.py", "910fb8608f9c1c55"),
    ("scripts/g1.py", BASE + "/scripts/g1.py", "1625914b9c55d2db"),
    ("scripts/g2a_summary.py", BASE + "/scripts/g2a_summary.py", "8b91ae05a77f4de9"),
    ("code_v2/models/pbe/src/net.py", CODE_V2 + "/models/pbe/src/net.py", "de4dcb9c59d4a49c"),
    ("code_v2/models/pbe/src/matching.py", CODE_V2 + "/models/pbe/src/matching.py", "3af81e50e47e4499"),
    ("code_v2/models/pbe/adapter.py", CODE_V2 + "/models/pbe/adapter.py", "23e061bbb1a88cf6"),
    ("code_v2/common/losses.py", CODE_V2 + "/common/losses.py", "365a663073c303f8"),
    ("g2a/val_list.txt", BASE + "/g2a/val_list.txt", "93ddae6ab100eb37"),
    ("g2a/split.json", BASE + "/g2a/split.json", "840d22ff92ff5339"),
    ("scripts/f5_boot.py", BASE + "/scripts/f5_boot.py", None),
    ("scripts/f5_summary.py", BASE + "/scripts/f5_summary.py", None),
    ("scripts/f5_queue.sh", BASE + "/scripts/f5_queue.sh", None),
    ("PREREG_F5.md", BASE + "/PREREG_F5.md", "42dc239842e778bfd2329fd88584cbc176356639f833eb9664873382dcfd4c80"),
]


# ------------------------------------------------------------------ formatting ----------------------------------
def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def fci(t):
    return "n/a" if t is None else f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def fsci(t):
    return "n/a" if t is None else f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def fpeak(v):
    if v["oom"]: return "OOM / non-executable"
    return "n/a" if v["peak_alloc_mib"] is None else f"{v['peak_alloc_mib']:.0f}"


# ------------------------------------------------------------------ rules (sec. 4) ------------------------------
def classify(t):
    """sec. 4.1 (O2 (a)). t = [point, lo, hi]."""
    pt, lo, hi = t
    if pt >= THR and lo >= THR: return "preserved"
    if hi < THR: return "not preserved"
    return "indeterminate"


def classify_point(pt):
    """O2 (b), G2 B6 continuity (point only)."""
    return "preserved (point)" if pt >= THR else "not preserved (point)"


def t_set(mres):
    """E = fp32 tiled variants with peak <= 512 MiB (O4: p512 included automatically when it fits). AMP excluded (O3)."""
    return sorted(vn for vn, v in mres["variants"].items()
                  if vn.startswith("tile_") and vn.endswith("_fp32") and v["exec"][E_BUDGET])


def t_statement(model, mres):
    name = DISPLAY.get(model, model)
    E = t_set(mres)
    if not E:
        return f"{name}: no tiled fp32 variant fits 0.5 GiB (VACUOUS)", {}
    cls = {}
    for vn in E:
        v = mres["variants"][vn]
        cls[vn] = classify(v["ratio"]) if (v.get("status") == "ok" and v.get("ratio")) else "indeterminate (no result)"
    pres = [vn for vn in E if cls[vn] == "preserved"]
    if len(pres) == len(E):
        s = (f"Tile inference of {name} within 0.5 GiB keeps thin-vessel clDice at ≥ 0.8 of full-resolution inference "
             f"(all executable tiled variants, CI lower bounds ≥ 0.8).")
    elif pres:
        rest = [f"{vn} ({cls[vn]}, r = {fci(mres['variants'][vn].get('ratio'))})" for vn in E if vn not in pres]
        s = f"Tile inference of {name} within 0.5 GiB preserves at {', '.join(pres)}, except {', '.join(rest)}."
    else:
        s = f"Tile inference of {name} within 0.5 GiB does not keep thin-vessel clDice at 0.8 of full resolution."
    return s, cls


def overlap_sentences(model, mres):
    """sec. 4.3: overlap effect attributed to seam relative only; delta-ratio gets numbers and the allowed wording only."""
    name = DISPLAY.get(model, model)
    L = []
    for p in map(str, fb.PATCHES):
        c = mres["contrasts"].get(p)
        if not c:
            L.append(f"- {name}, patch {p}: contrast not computable (missing fp32 o0/o50 result or full-resolution reference)."); continue
        ds, dr = c["dseam"], c["dratio"]
        if ds[1] > 0: st = "overlap raises seam continuity"
        elif ds[2] < 0: st = "overlap lowers seam continuity"
        else: st = "no detectable seam effect"
        L.append(f"- {name}, patch {p}: {st} (Δseam = {fsci(ds)}).")
        if dr[1] <= 0 <= dr[2]:
            L.append(f"- {name}, patch {p}: whole-image ratio does not differ detectably between 0% and 50% overlap (Δratio = {fsci(dr)}).")
        else:
            d = "higher at 50% than at 0% overlap" if dr[0] > 0 else "higher at 0% than at 50% overlap"
            L.append(f"- {name}, patch {p}: whole-image ratio is {d} (Δratio = {fsci(dr)}), not attributed to overlap.")
    return L


# ------------------------------------------------------------------ tables --------------------------------------
HDR = ("| variant | peak MiB (alloc, batch 1) | exec 2 GiB | exec 0.5 GiB | thin3 clDice [CI] | clDice [CI] | Dice [CI] | thin5 clDice [CI] "
       "| ratio r [CI] | T class (CI rule) | point-only (G2 B6) | seam grid | band [CI] | interior [CI] | seam relative s [CI] | seam break (<0.8) | note |")
SEP = "|" + "---|" * 17


def row(vn, v, in_e, cls):
    ex = v["exec"]
    note = []
    if "_p512_" in vn: note.append("weights trained on 256 crops")
    if v.get("nonfinite") and not fb.is_amp(vn): note.append(f"non-finite {v['nonfinite']}/{v['n']} (fp32)")
    if v.get("mode") and v["mode"] != "gpu": note.append(f"mode {v['mode']}")
    lead = f"| {vn} | {fpeak(v)} | {'yes' if ex['2 GiB'] else 'no'} | {'yes' if ex['0.5 GiB'] else 'no'} |"
    if v.get("status") == "nonfinite":
        return lead + f" {v['nonfinite_label']} |" + " — |" * 11 + f" {'; '.join(note)} |"
    if v.get("status") != "ok":
        return lead + " no results |" + " — |" * 11 + f" {'; '.join(note)} |"
    m = v["median"]; r = v.get("ratio")
    tc = cls.get(vn, "—") if in_e else "— (not a T target)"
    pc = classify_point(r[0]) if (r and in_e) else "—"
    sm = v.get("seam")
    if sm:
        seam = f" {sm['grid']} | {fci(sm['band'])} | {fci(sm['interior'])} | {fci(sm.get('s'))} | {'YES' if sm.get('flag') else 'no'} |"
    else:
        seam = " — | — | — | — | — |"
    return (lead + f" {fci(m['thin3_cldice'])} | {fci(m['cldice'])} | {fci(m['dice'])} | {fci(m['thin5_cldice'])} | {fci(r)} | {tc} | {pc} |"
            + seam + f" {'; '.join(note)} |")


def model_block(model, mres):
    name = DISPLAY.get(model, model)
    stmt, cls = t_statement(model, mres)
    E = set(t_set(mres))
    V = mres["variants"]
    order_full = [vn for vn in ("full_fp32", "full_amp") if vn in V]
    tiles_fp32 = sorted(vn for vn in V if vn.startswith("tile_") and vn.endswith("_fp32"))
    tiles_amp = sorted(vn for vn in V if vn.startswith("tile_") and vn.endswith("_amp"))
    resize = sorted((vn for vn in V if vn.startswith("rs")), key=lambda s: int(s[2:].split("_")[0]))
    L = [f"### {name}", "", "Absolute values (medians over images, paired percentile 95% CI) precede ratios. "
         "thin3 medians over I3; other medians over all images.", ""]
    L += ["Full-resolution reference and tiled fp32 (T candidates):", "", HDR, SEP]
    L += [row(vn, V[vn], vn in E, cls) for vn in order_full + tiles_fp32]
    L += ["", "AMP rows (auxiliary, descriptive; excluded from T, O3):", "", HDR, SEP]
    L += [row(vn, V[vn], False, cls) for vn in tiles_amp] if tiles_amp else ["| (none) |" + " |" * 16]
    L += ["", "Block \"resize, not tiled\" (not a T target, O8):", "", HDR, SEP]
    L += [row(vn, V[vn], False, cls) for vn in resize] if resize else ["| (none) |" + " |" * 16]
    L += ["", f"E (fp32 tiled, peak ≤ 512 MiB) = {sorted(E) if E else '∅'}", "",
          f"**T statement ({name}):** {stmt}", "", "Overlap attribution (sec. 4.3; fp32; T statements do not use overlap):", ""]
    L += overlap_sentences(model, mres)
    L.append("")
    return L, stmt, cls


# ------------------------------------------------------------------ manuscript assert (sec. 4.2) ---------------
TEX_MODEL = {"UCMNet": "ucmnet", "UTANet": "utanet", "MAResUNet": "maresunet"}


def tex_variant(s):
    s = s.strip().replace("\\%", "%")
    if s == "Full image": return "full_fp32"
    m = re.fullmatch(r"Resize (\d+)", s)
    if m: return f"rs{m.group(1)}_fp32"
    m = re.fullmatch(r"Tiles (\d+), (\d+)%", s)
    if m: return f"tile_p{m.group(1)}_o{m.group(2)}_fp32"
    raise ValueError("unknown variant label: " + s)


def parse_tex_table(path):
    txt = open(path, encoding="utf-8").read()
    i = txt.index("\\label{tab:tiling_fives}")
    body = txt[i:txt.index("\\end{tblr}", i)]
    rows, cur = {}, None
    for ln in body.splitlines():
        ln = ln.strip()
        if not ln.endswith("\\\\") or "&" not in ln or ln.startswith("Model"): continue
        c = [x.strip() for x in ln[:-2].split("&")]
        if c[0]: cur = TEX_MODEL[c[0]]
        rows[(cur, tex_variant(c[1]))] = {"peak": c[2], "thin3": c[3], "dice": c[4], "ratio": c[5], "seam": c[6], "2 GiB": c[7], "0.5 GiB": c[8]}
    return rows


def manuscript_check(res, tex):
    table = parse_tex_table(tex)
    bad = []
    for (m, vn), want in sorted(table.items()):
        v = res["models"][m]["variants"].get(vn)
        if v is None or v.get("status") != "ok":
            bad.append(f"{m}/{vn}: no computed row"); continue
        sm = v.get("seam")
        got = {"peak": fpeak(v), "thin3": f3(v["median"]["thin3_cldice"][0]), "dice": f3(v["median"]["dice"][0]),
               "ratio": f3(v["ratio"][0]), "seam": f3(sm["s"][0]) if sm else "--",
               "2 GiB": "Yes" if v["exec"]["2 GiB"] else "No", "0.5 GiB": "Yes" if v["exec"]["0.5 GiB"] else "No"}
        for k in want:
            if got[k] != want[k]: bad.append(f"{m}/{vn} {k}: manuscript {want[k]} vs computed {got[k]}")
    return len(table), bad


# ------------------------------------------------------------------ sha (sec. 7) and markers --------------------
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def sha_block(paths_override, amend0):
    L = ["| file | path | expected (prefix) | computed | status |", "|---|---|---|---|---|"]
    ok_all = True
    for name, path, exp in SHA_TABLE:
        path = paths_override.get(name, path)
        if exp is None: exp = amend0.get(name)
        if not os.path.exists(path):
            got, st = "—", "MISSING"
        else:
            got = sha256(path)
            st = "UNRECORDED (AMENDMENT 0 pending)" if exp is None else ("OK" if got.startswith(exp) else "DIFFERS")
        ok_all &= st == "OK"
        L.append(f"| {name} | {path} | {exp or '—'} | {got} | {st} |")
    L.append("")
    L.append("Validity: " + ("valid (all sec. 7 sha256 match)" if ok_all else "**invalid until amendment**"))
    return L, ok_all


def jobs_complete(pbe_dir):
    """f5_queue.sh markers: markers/pbe__<v>.DONE|.FAIL|.DROP and f5/ALL_DONE ("ALL_DONE ..." on normal end)."""
    mk = os.path.join(pbe_dir, "markers")
    st = {}
    for v in PBE_VARIANTS:
        k = os.path.join(mk, f"pbe__{v}")
        st[v] = ("FAIL" if os.path.exists(k + ".FAIL") else "DONE" if os.path.exists(k + ".DONE")
                 else "DROP" if os.path.exists(k + ".DROP") else "absent")
    ad = os.path.join(pbe_dir, "ALL_DONE")
    ad_txt = open(ad).read().strip() if os.path.exists(ad) else ""
    done = ad_txt.startswith("ALL_DONE") and all(s in ("DONE", "DROP") for s in st.values())
    drops = [v for v, s in st.items() if s == "DROP"]
    return done, st, ad_txt, drops


# ------------------------------------------------------------------ main ----------------------------------------
LIMIT = ("Limitations (sec. 4.4): 1 dataset (FIVES), 1 seed, 100 train-internal validation images, the same validation images "
         "used for checkpoint selection (optimism, identical for every variant); the official FIVES test set was not used. "
         "Patch-512 rows use weights trained on 256 crops (G2 B3 limitation).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g2-dir", required=True); ap.add_argument("--pbe-dir", default=None)
    ap.add_argument("--tex", required=True); ap.add_argument("--out", required=True); ap.add_argument("--boot-json", required=True)
    ap.add_argument("--sha-paths", default=None); ap.add_argument("--amend0", default=None)
    ap.add_argument("--title", default="F5 summary (PBE-Net under the FIVES T protocol)")
    a = ap.parse_args()
    paths_override = json.load(open(a.sha_paths)) if a.sha_paths else {}
    amend0 = json.load(open(a.amend0)) if a.amend0 else {}

    models, missing = {}, {}
    if a.pbe_dir:
        models["pbe"], missing["pbe"] = fb.load_model(os.path.join(a.pbe_dir, "pbe"))
    for m in G2_MODELS:
        models[m], missing[m] = fb.load_model(os.path.join(a.g2_dir, m))
    res = fb.analyze(models)                     # one pair of index matrices (all -> I3) for every model and variant
    res["missing_inputs"] = missing

    L = [f"# {a.title}", "", "Rules: PREREG_F5.md (frozen) sec. 3-4, 7. Generated mechanically by f5_summary.py.",
         f"Bootstrap: paired percentile, B = {res['B']}, numpy.random.default_rng({res['seed']}), index matrices all ({res['n_all']} images) "
         f"then I3 ({res['n_i3']} images with thin3_nskel > 0), shared by every model and variant. Threshold {THR}.", ""]
    L += ["## sha256 check (sec. 7)", ""]
    sl, sha_ok = sha_block(paths_override, amend0); L += sl; L.append("")
    if a.pbe_dir:
        done, st, ad_txt, drops = jobs_complete(a.pbe_dir)
        L += [f"Jobs complete: {'YES' if done else 'NO'}  (markers: {st}; ALL_DONE: {ad_txt or 'absent'})",
              f"Dropped variants (S2/O5): {drops if drops else 'none'}",
              f"Missing results.json: {missing['pbe'] if missing['pbe'] else 'none'}" + ("" if done else "  -> judgments are provisional"), ""]
    L += [LIMIT, ""]

    statements = {}
    if a.pbe_dir:
        L += ["## 4.1 PBE-Net (primary)", ""]
        bl, stmt, cls = model_block("pbe", res["models"]["pbe"]); L += bl
        statements["pbe"] = {"T": stmt, "classes": cls}

    n_rows, bad = manuscript_check(res, a.tex)
    L += ["## 4.2 Same rule, recomputed for the three G2 models (no retraining, no re-inference)", "",
          f"Manuscript match (Table tab:tiling_fives, {n_rows} rows x peak, thin3 clDice, Dice, ratio, seam, budgets; "
          f"source {os.path.basename(a.tex)}): " + ("**PASS**" if not bad else f"**FAIL** ({len(bad)} mismatches; recorded as a bug, interpretation halted)"), ""]
    if bad:
        L += ["- " + b for b in bad] + [""]
    else:
        for m in G2_MODELS:
            bl, stmt, cls = model_block(m, res["models"][m]); L += bl
            statements[m] = {"T": stmt, "classes": cls}
        L += ["Statement under review in the manuscript: \"tile inference within 0.5 GiB preserves thin-vessel topology, "
              "with UCMNet at 256-pixel patches as the one exception\". Replaced or confirmed by the T statements above (sec. 4.2).", ""]

    res["statements"] = statements
    res["manuscript_match"] = {"rows": n_rows, "mismatches": bad}
    res["sha_valid"] = sha_ok
    json.dump(res, open(a.boot_json, "w"), indent=1)
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", a.out, a.boot_json)
    assert not bad, "manuscript mismatch: " + "; ".join(bad)


if __name__ == "__main__":
    main()
