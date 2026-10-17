"""F4 ablation analysis (PREREG_ABL section 4), plus the synthetic reverse-check (section 5).
  python abl_analyze.py synthetic   -> BASE/synthetic.log (lines prefixed SYNTHETIC); exit 1 if any check fails
  python abl_analyze.py real        -> BASE/abl_result.{json,md}; only after ALL_DONE and test_eval.json of every DONE run;
                                       refuses if abl_result.json exists (run once).
Metric: per-run test mean IoU. Test: two-sided paired t over splits k = 1..5 (variant k vs full v1 k).
Family: Holm step-down over all tested (variant, dataset) cells (12 if complete), alpha 0.05.
Significant drop: Holm p < 0.05 and mean(variant - full) < 0. Effect: significant drop on >= 2 of 3 datasets."""
import json
import math
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abl_common as A  # noqa: E402

ALPHA = 0.05


def paired_t(v, f):
    """manual paired t: returns (mean diff, t, p two-sided, ci95 half-width)."""
    d = np.asarray(v, float) - np.asarray(f, float)
    n = len(d)
    m = d.mean()
    sd = d.std(ddof=1)
    if sd == 0:
        return m, (math.inf if m != 0 else 0.0), (0.0 if m != 0 else 1.0), 0.0
    t = m / (sd / math.sqrt(n))
    p = 2 * stats.t.sf(abs(t), n - 1)
    return m, t, p, stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)


def holm(ps):
    ps = np.asarray(ps, float)
    m = len(ps)
    order = np.argsort(ps, kind='mergesort')
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[j]))
        adj[j] = run
    return adj


def decide(cells):
    """cells: {(v, d): (variant list or None, full list)} -> table rows + per-variant verdict."""
    tested = [(key, val) for key, val in cells.items() if val[0] is not None and len(val[0]) == len(A.SPLITS)]
    rows = {}
    for key, (vv, ff) in tested:
        m, t, p, hw = paired_t(vv, ff)
        rows[key] = {'mean_diff': m, 't': t, 'p_raw': p, 'ci95': [m - hw, m + hw]}
    adj = holm([rows[key]['p_raw'] for key, _ in tested]) if tested else []
    for (key, _), a in zip(tested, adj):
        r = rows[key]
        r['p_holm'] = float(a)
        r['sig_drop'] = bool(a < ALPHA and r['mean_diff'] < 0)
        r['sig_up'] = bool(a < ALPHA and r['mean_diff'] > 0)
    for key in cells:
        if key not in rows:
            rows[key] = {'status': 'not tested'}
    verdict = {}
    for v in sorted({key[0] for key in cells}):
        drops = sum(1 for d in A.DATASETS if rows.get((v, d), {}).get('sig_drop'))
        ups = sum(1 for d in A.DATASETS if rows.get((v, d), {}).get('sig_up'))
        verdict[v] = {'sig_drop_datasets': drops, 'sig_up_datasets': ups,
                      'effect': drops >= 2, 'harmful': ups >= 2,
                      'class': 'effect' if drops >= 2 else ('harmful' if ups >= 2 else 'no effect claim')}
    return rows, verdict, len(tested)


def full_values(metric='iou'):
    return {d: [A.ref_results('pbe', d, k)['test'][metric] for k in A.SPLITS] for d in A.DATASETS}


def synthetic():
    log = []
    ok_all = True

    def say(name, ok, msg):
        nonlocal ok_all
        ok_all &= bool(ok)
        line = f'SYNTHETIC {name}: {"PASS" if ok else "FAIL"} {msg}'
        print(line, flush=True)
        log.append(line)

    # 1. real-data path: Table 1 CVC PBE-Net vs MK-UNet (raw p 0.0075, diff +0.023, TABLE1_SIG.md)
    pb = [A.ref_results('pbe', 'cvc', k)['test']['iou'] for k in A.SPLITS]
    mk = [A.ref_results('mk_unet', 'cvc', k)['test']['iou'] for k in A.SPLITS]
    m, t, p, _ = paired_t(pb, mk)
    say('1_table1_repro', abs(p - 0.0075) < 5e-4 and round(m, 3) == 0.023, f'p={p:.4f} diff={m:+.4f} (expect 0.0075, +0.023)')
    full = full_values()
    sd_full = {d: float(np.std(full[d], ddof=1)) for d in A.DATASETS}
    rs = np.random.RandomState(7919)
    vs = list(A.VARIANTS)

    # 2. planted drop on V1 (cvc, isic17)
    cells = {}
    for v in vs:
        for d in A.DATASETS:
            f = np.array(full[d])
            if v == 'pbe_idskip' and d in ('cvc', 'isic17'):
                cells[(v, d)] = (list(f - 0.05 + rs.normal(0, 0.005, 5)), list(f))
            else:
                cells[(v, d)] = (list(f + rs.normal(0, 0.005, 5)), list(f))
    rows, ver, n = decide(cells)
    say('2_planted_drop', ver['pbe_idskip']['effect'] and not any(ver[v]['effect'] for v in vs[1:]) and n == 12,
        f'classes={ {v: ver[v]["class"] for v in vs} } family={n}')

    # 3. null: 1000 families, variant = full + N(0, SD_full)
    fam_effect = fam_any = 0
    for _ in range(1000):
        cells = {(v, d): (list(np.array(full[d]) + rs.normal(0, sd_full[d], 5)), full[d]) for v in vs for d in A.DATASETS}
        rows, ver, n = decide(cells)
        fam_effect += any(ver[v]['effect'] for v in vs)
        fam_any += any(r.get('p_holm', 1) < ALPHA for r in rows.values())
    say('3_null_fwer', fam_effect / 1000 <= 0.05 and fam_any / 1000 <= 0.05,
        f'P(>=1 effect)={fam_effect / 1000:.3f} P(any Holm-sig cell)={fam_any / 1000:.3f} (<= 0.05)')

    # 4. sign: variant = full + 0.05
    cells = {(v, d): (list(np.array(full[d]) + 0.05 + rs.normal(0, 0.005, 5)), full[d]) for v in vs for d in A.DATASETS}
    rows, ver, n = decide(cells)
    say('4_sign', all(ver[v]['class'] == 'harmful' and not ver[v]['effect'] for v in vs) and not any(r.get('sig_drop') for r in rows.values()),
        f'classes={sorted({ver[v]["class"] for v in vs})}')

    # 5. Holm hand case and manual-t vs ttest_rel
    ps = [0.001, 0.004, 0.01, 0.02, 0.03, 0.04, 0.05, 0.2, 0.3, 0.5, 0.7, 0.9]
    hand = [0.012, 0.044, 0.10, 0.18, 0.24, 0.28, 0.30, 1, 1, 1, 1, 1]
    got = holm(ps[::-1])[::-1]
    dmax = 0.0
    for _ in range(200):
        a = rs.normal(0.8, 0.03, 5)
        b = a + rs.normal(0.01, 0.02, 5)
        dmax = max(dmax, abs(paired_t(b, a)[2] - stats.ttest_rel(b, a).pvalue))
    say('5_holm_t', np.allclose(got, hand, atol=1e-12) and dmax < 1e-12, f'holm={np.round(got, 4).tolist()} max|p_manual-p_scipy|={dmax:.2e}')

    # 6. incomplete bundle: one (v, d) with 4 splits -> not tested, family 11
    cells = {(v, d): (list(np.array(full[d]) + rs.normal(0, 0.005, 5)), full[d]) for v in vs for d in A.DATASETS}
    cells[('pbe_se', 'isic18')] = (cells[('pbe_se', 'isic18')][0][:4], full['isic18'])
    rows, ver, n = decide(cells)
    say('6_incomplete', n == 11 and rows[('pbe_se', 'isic18')] == {'status': 'not tested'}, f'family={n}')
    print('SYNTHETIC ALL', 'PASS' if ok_all else 'FAIL', flush=True)
    log.append(f'SYNTHETIC ALL {"PASS" if ok_all else "FAIL"}')
    with open(os.path.join(A.BASE, 'synthetic.log'), 'a') as fh:
        fh.write('\n'.join(log) + '\n')
    return ok_all


def real():
    if not os.path.exists(os.path.join(A.BASE, 'ALL_DONE')):
        raise SystemExit('refuse: ALL_DONE missing')
    dst = os.path.join(A.BASE, 'abl_result.json')
    if os.path.exists(dst):
        raise SystemExit(f'refuse: {dst} exists (analysis runs once)')
    full = full_values('iou')
    full_d = full_values('dice')
    cells, extra = {}, {}
    for v in A.VARIANTS:
        for d in A.DATASETS:
            iou, dice, pinf = [], [], None
            for k in A.SPLITS:
                rd = A.run_dir(v, d, k)
                te = os.path.join(rd, 'test_eval.json')
                if os.path.exists(os.path.join(rd, 'DONE')) and os.path.exists(te):
                    t = json.load(open(te))['test']
                    iou.append(t['iou'])
                    dice.append(t['dice'])
                    pinf = json.load(open(os.path.join(rd, 'results.json')))['params_inference']
            cells[(v, d)] = (iou if len(iou) == 5 else None, full[d])
            extra[(v, d)] = {'n_splits': len(iou), 'variant_iou': iou, 'full_iou': full[d],
                             'variant_mean_sd': [float(np.mean(iou)), float(np.std(iou, ddof=1))] if len(iou) > 1 else None,
                             'full_mean_sd': [float(np.mean(full[d])), float(np.std(full[d], ddof=1))],
                             'dice_diff': float(np.mean(dice) - np.mean(full_d[d])) if len(dice) == 5 else None,
                             'params_inference': pinf}
    rows, ver, n = decide(cells)
    out = {'family_size': n, 'cells': {f'{v}|{d}': {**rows[(v, d)], **extra[(v, d)]} for (v, d) in cells},
           'verdict': ver, 'n_effect_components': sum(ver[v]['effect'] for v in ver)}
    out['prereg_consequence'] = ('0 effects: remove component claims from title and contributions'
                                 if out['n_effect_components'] == 0 else
                                 'only components with effect may be claimed as measured under protocol v1')
    with open(dst, 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    md = ['| variant | dataset | n | mean diff | 95% CI | raw p | Holm p | sig | variant mean+-SD | full mean+-SD | Dice diff | params_inf |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for (v, d), r in rows.items():
        e = extra[(v, d)]
        if 'p_raw' in r:
            sig = 'drop' if r['sig_drop'] else ('up' if r['sig_up'] else '-')
            md.append(f'| {v} | {d} | 5 | {r["mean_diff"]:+.4f} | [{r["ci95"][0]:+.4f}, {r["ci95"][1]:+.4f}] | {r["p_raw"]:.4f} | '
                      f'{r["p_holm"]:.4f} | {sig} | {e["variant_mean_sd"][0]:.4f}+-{e["variant_mean_sd"][1]:.4f} | '
                      f'{e["full_mean_sd"][0]:.4f}+-{e["full_mean_sd"][1]:.4f} | {e["dice_diff"]:+.4f} | {e["params_inference"]} |')
        else:
            md.append(f'| {v} | {d} | {e["n_splits"]} | not tested | | | | | | | | |')
    md.append('')
    md.append(f'family size {n}; verdict: ' + json.dumps({v: ver[v]['class'] for v in ver}))
    md.append(out['prereg_consequence'])
    with open(os.path.join(A.BASE, 'abl_result.md'), 'w') as fh:
        fh.write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    if sys.argv[1] == 'synthetic':
        sys.exit(0 if synthetic() else 1)
    elif sys.argv[1] == 'real':
        real()
    else:
        raise SystemExit('usage: abl_analyze.py synthetic|real')
