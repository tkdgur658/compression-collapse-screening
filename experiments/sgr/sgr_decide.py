# -*- coding: utf-8 -*-
"""PREREG_SGR decision (pure python, no torch).
  --selftest : §9.1 synthetic reverse check, 4 fake scenarios through the SAME judge() -> SELFTEST.json, synthetic.log
  --decide   : ONCE, only after out/ALL_DONE (run_decide.sh enforces) -> RESULT_SGR.json / RESULT_SGR.md"""
import os as _ccs_os  # path configuration of the released copy (README, Configuration)
CCS_REPO_ROOT = _ccs_os.path.normpath(_ccs_os.path.join(_ccs_os.path.dirname(_ccs_os.path.abspath(__file__)), '../..'))
CCS_WORK_ROOT = _ccs_os.environ.get('CCS_WORK', _ccs_os.path.join(CCS_REPO_ROOT, 'experiments'))
import os
import sys
import json
import glob
import math
import time
import random
import argparse

sys.dont_write_bytecode = True
BASE = _ccs_os.path.join(CCS_WORK_ROOT, 'sgr')
OUT = BASE + '/out/items'
THR = 0.8
H_R1_MIN_FRAC = 0.7
H_R4_RATIO = 0.8
ALPHA = 0.05
B_BOOT = 10000
SEED = 20261017
MAIN_ARMS = ['B0', 'B1', 'B2', 'B3', 'SGR']
SUB_ARMS = ['B4', 'A1', 'A2', 'A3']
W4_ARMS = ['B0w', 'B1w', 'B3w', 'B5', 'SGRw']
NAUX_ARMS = ['SGR_n10', 'SGR_n30', 'B1_n10', 'B1_n30']


def jload(p):
    with open(p, encoding='utf8') as f:
        return json.load(f)


def sanitize(o):
    if isinstance(o, float):
        return o if (o == o and o not in (float('inf'), float('-inf'))) else None
    if isinstance(o, dict):
        return {str(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    return o


def dump_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf8') as f:
        json.dump(sanitize(obj), f, indent=1, sort_keys=True, allow_nan=False)
    os.replace(tmp, path)


def cell_of(tag):
    m, d, k = tag.split('__')
    return '%s|%s' % (m, d)


def cell_means(Rarm, tags):
    """F3 cell_table rule: mean over the splits that have a value; cells with no value -> None."""
    acc = {}
    for t in tags:
        acc.setdefault(cell_of(t), [])
        v = Rarm.get(t)
        if v is not None:
            acc[cell_of(t)].append(v)
    return {c: ({'mean_R': sum(v) / len(v), 'n_splits': len(v)} if v else {'mean_R': None, 'n_splits': 0}) for c, v in acc.items()}


def repaired(Rarm, tags):
    cm = cell_means(Rarm, tags)
    return {c: (None if v['mean_R'] is None else v['mean_R'] >= THR) for c, v in cm.items()}, cm


def mcnemar_one_sided(b, c):
    """exact one-sided McNemar (SGR better): P(Binom(b+c, 1/2) >= b); b + c = 0 -> 1."""
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(b, n + 1)) / 2.0 ** n


def holm(pvals, alpha=ALPHA):
    keys = sorted(pvals, key=lambda k: pvals[k])
    m = len(keys)
    rej, stop = {}, False
    for i, k in enumerate(keys):
        if not stop and pvals[k] <= alpha / (m - i):
            rej[k] = True
        else:
            stop = True
            rej[k] = False
    return rej


def boot_p(sgr, b4, cells, B=B_BOOT, seed=SEED):
    """cell-paired bootstrap, p = Pr*(n_SGR* - 0.8 n_B4* <= 0)."""
    rng = random.Random(seed)
    n = len(cells)
    bad = 0
    for _ in range(B):
        s = [cells[rng.randrange(n)] for _ in range(n)]
        d = sum(1 for c in s if sgr[c]) - H_R4_RATIO * sum(1 for c in s if b4[c])
        if d <= 0:
            bad += 1
    return bad / B


def judge(R, cells_main, tags_main, cells_sub, tags_sub, complete):
    """R[arm][tag] = retention (val Dice_q / val Dice_fp32) or None.  complete = {'main': bool, 'sub': bool}."""
    out = {'counts': {}, 'hyp': {}}
    rep = {}
    for arm in MAIN_ARMS:
        r, cm = repaired(R.get(arm, {}), tags_main)
        rep[arm] = r
        out['counts'][arm] = {'repaired': sum(1 for c in cells_main if r.get(c)), 'n_cells': len(cells_main),
                              'cells_without_value': sum(1 for c in cells_main if r.get(c) is None)}
    n55 = len(cells_main)
    need = int(math.ceil(H_R1_MIN_FRAC * n55 - 1e-12))
    if complete['main']:
        nS = out['counts']['SGR']['repaired']
        out['hyp']['H_R1'] = {'verdict': 'PASS' if nS >= need else 'FAIL', 'n_repaired': nS, 'need': need, 'n_cells': n55}
        pairs = [c for c in cells_main if rep['SGR'].get(c) is not None and rep['B1'].get(c) is not None]
        b = sum(1 for c in pairs if rep['SGR'][c] and not rep['B1'][c])
        cc = sum(1 for c in pairs if rep['B1'][c] and not rep['SGR'][c])
        out['hyp']['H_R2'] = {'b_SGR_only': b, 'c_B1_only': cc, 'n_pairs': len(pairs), 'p_one_sided': mcnemar_one_sided(b, cc)}
    else:
        out['hyp']['H_R1'] = {'verdict': 'NOT TESTABLE (main block incomplete)'}
        out['hyp']['H_R2'] = {'verdict': 'NOT TESTABLE (main block incomplete)', 'p_one_sided': None}
    # H_R4 on the 3-model subset (SGR from the main records of the subset tags)
    rS, _ = repaired(R.get('SGR', {}), tags_sub)
    rB4, _ = repaired(R.get('B4', {}), tags_sub)
    for arm in SUB_ARMS:
        r, _ = repaired(R.get(arm, {}), tags_sub)
        out['counts']['sub_' + arm] = {'repaired': sum(1 for c in cells_sub if r.get(c)), 'n_cells': len(cells_sub)}
    out['counts']['sub_SGR'] = {'repaired': sum(1 for c in cells_sub if rS.get(c)), 'n_cells': len(cells_sub)}
    if complete['main'] and complete['sub']:
        nS4 = sum(1 for c in cells_sub if rS.get(c)); nB4 = sum(1 for c in cells_sub if rB4.get(c))
        if nB4 == 0:
            out['hyp']['H_R4'] = {'verdict': 'NOT TESTABLE (B4 repairs 0 cells)', 'p_boot': 1.0, 'n_SGR': nS4, 'n_B4': nB4}
        else:
            ok_pt = nS4 >= H_R4_RATIO * nB4
            p = boot_p({c: bool(rS.get(c)) for c in cells_sub}, {c: bool(rB4.get(c)) for c in cells_sub}, list(cells_sub))
            out['hyp']['H_R4'] = {'n_SGR': nS4, 'n_B4': nB4, 'point_ok': ok_pt, 'p_boot': p}
    else:
        out['hyp']['H_R4'] = {'verdict': 'NOT TESTABLE (block incomplete)', 'p_boot': 1.0}
    # Holm over {H_R2, H_R4}
    pv = {'H_R2': out['hyp']['H_R2'].get('p_one_sided'), 'H_R4': out['hyp']['H_R4'].get('p_boot')}
    pv = {k: (1.0 if v is None else v) for k, v in pv.items()}
    rej = holm(pv)
    out['holm'] = {'p': pv, 'reject': rej}
    if complete['main']:
        h2 = out['hyp']['H_R2']
        h2['verdict'] = 'PASS' if (rej['H_R2'] and h2['b_SGR_only'] > h2['c_B1_only']) else 'FAIL'
    h4 = out['hyp']['H_R4']
    if 'verdict' not in h4:
        h4['verdict'] = 'PASS' if (h4['point_ok'] and rej['H_R4']) else 'FAIL'
    # pre-registered branches
    br = []
    if complete['main']:
        if out['counts']['B1']['repaired'] > out['counts']['SGR']['repaired']:
            br.append('B1 repairs more cells than SGR: SGR discarded, B1 reported as the R2 implementation')
        if out['hyp']['H_R1']['verdict'] == 'FAIL':
            br.append('H_R1 not met: report "label-free repair under mask / bit-width preservation reaches %d/%d cells"; R2 = partial verification'
                      % (out['counts']['SGR']['repaired'], n55))
    out['branches'] = br
    out['B1_better'] = bool(complete['main'] and out['counts']['B1']['repaired'] > out['counts']['SGR']['repaired'])
    return out


# ------------------------------------------------------------------ selftest (§9.1)
def fake_R(arm_sets, cells_main, tags_main, tags_sub, straddle_cell):
    """arm_sets[arm] = set of repaired cells.  R = 0.95 (repaired) / 0.3; the straddle cell (SGR only) gets split values
    [0.75, 0.85 x4] (mean 0.83, repaired) or [0.85, 0.75 x4] (mean 0.77, not repaired)."""
    R = {}
    alltags = sorted(set(tags_main) | set(tags_sub))
    for arm, cs in arm_sets.items():
        R[arm] = {}
        for t in alltags:
            c = cell_of(t)
            k = int(t.rsplit('__k', 1)[1])
            if arm == 'SGR' and c == straddle_cell:
                R[arm][t] = (0.75 if k == 1 else 0.85) if c in cs else (0.85 if k == 1 else 0.75)
            else:
                R[arm][t] = 0.95 if c in cs else 0.3
    return R


def selftest():
    it = jload(BASE + '/ITEMS_SGR.json')
    tm, ts = it['blocks']['main'], it['blocks']['sub']
    cm = sorted(set(cell_of(t) for t in tm))
    cs = sorted(set(cell_of(t) for t in ts))
    other = [c for c in cm if c not in cs]
    assert len(cm) == 55 and len(cs) == 13, (len(cm), len(cs))
    allc = set(cm)
    straddle = other[0]
    sc = {}
    sc['S1_all_repaired'] = ({a: allc for a in MAIN_ARMS + SUB_ARMS},
                             {'H_R1': 'PASS', 'H_R2': 'FAIL', 'H_R4': 'PASS', 'n_SGR': 55, 'B1_better': False})
    sc['S2_all_fail'] = ({a: set() for a in MAIN_ARMS + SUB_ARMS},
                         {'H_R1': 'FAIL', 'H_R2': 'FAIL', 'H_R4': 'NOT TESTABLE', 'n_SGR': 0, 'B1_better': False})
    b1_s3 = set(cs) | set(other[:37])                       # 50 cells incl. all 13 subset cells
    sgr_s3 = set(cs[:6]) | set(other[:24])                  # 30 cells, subset of B1, 6 subset cells
    sc['S3_B1_better'] = ({'B0': set(), 'B1': b1_s3, 'B2': set(), 'B3': set(), 'SGR': sgr_s3, 'B4': set(cs), 'A1': set(), 'A2': set(), 'A3': set()},
                          {'H_R1': 'FAIL', 'H_R2': 'FAIL', 'H_R4': 'FAIL', 'n_SGR': 30, 'B1_better': True})
    sgr_s4 = set(cs) | set(other[:32])                      # 45 cells incl. all subset cells
    b1_s4 = set(sorted(sgr_s4)[:20])                        # 20 cells, subset of SGR -> b = 25, c = 0
    sc['S4_SGR_better'] = ({'B0': set(), 'B1': b1_s4, 'B2': set(), 'B3': set(), 'SGR': sgr_s4, 'B4': set(cs), 'A1': set(), 'A2': set(), 'A3': set()},
                           {'H_R1': 'PASS', 'H_R2': 'PASS', 'H_R4': 'PASS', 'n_SGR': 45, 'B1_better': False, 'p_H_R2': 2.0 ** -25})
    res, lines = {}, []
    ok_all = True
    for name, (sets, exp) in sc.items():
        R = fake_R(sets, cm, tm, ts, straddle)
        j = judge(R, cm, tm, cs, ts, {'main': True, 'sub': True})
        got = {'H_R1': j['hyp']['H_R1']['verdict'], 'H_R2': j['hyp']['H_R2']['verdict'],
               'H_R4': j['hyp']['H_R4']['verdict'].split(' (')[0], 'n_SGR': j['counts']['SGR']['repaired'], 'B1_better': j['B1_better']}
        ok = all(got[k] == exp[k] for k in got)
        if 'p_H_R2' in exp:
            got['p_H_R2'] = j['hyp']['H_R2']['p_one_sided']
            ok = ok and abs(got['p_H_R2'] - exp['p_H_R2']) < 1e-15
        res[name] = {'expected': exp, 'got': got, 'PASS': ok, 'holm': j['holm'], 'branches': j['branches']}
        ok_all = ok_all and ok
        lines.append('SELFTEST %s %s expected=%s got=%s' % (name, 'PASS' if ok else 'FAIL', json.dumps(exp, sort_keys=True), json.dumps(got, sort_keys=True)))
        print(lines[-1], flush=True)
    # straddle-cell mean rule (S1: repaired with one split < 0.8; S2: not repaired with one split > 0.8)
    R1 = fake_R({'SGR': allc}, cm, tm, ts, straddle)
    r1, cmeans = repaired(R1['SGR'], tm)
    ok_st = r1[straddle] is True and abs(cmeans[straddle]['mean_R'] - 0.83) < 1e-12
    R2 = fake_R({'SGR': set()}, cm, tm, ts, straddle)
    r2, cmeans2 = repaired(R2['SGR'], tm)
    ok_st = ok_st and r2[straddle] is False and abs(cmeans2[straddle]['mean_R'] - 0.77) < 1e-12
    res['straddle_mean_rule'] = {'PASS': ok_st, 'cell': straddle}
    lines.append('SELFTEST straddle_mean_rule %s cell=%s' % ('PASS' if ok_st else 'FAIL', straddle))
    print(lines[-1], flush=True)
    # unit values
    p = mcnemar_one_sided(10, 2)
    ok_m = abs(p - 79.0 / 4096.0) < 1e-15 and mcnemar_one_sided(0, 0) == 1.0
    hr = holm({'a': 0.01, 'b': 0.04}); hr2 = holm({'a': 0.03, 'b': 0.04}); hr3 = holm({'a': 0.001, 'b': 0.2})
    ok_h = hr == {'a': True, 'b': True} and hr2 == {'a': False, 'b': False} and hr3 == {'a': True, 'b': False}
    res['mcnemar_unit'] = {'PASS': ok_m, 'p_10_2': p}
    res['holm_unit'] = {'PASS': ok_h}
    lines.append('SELFTEST mcnemar_unit %s p(10,2)=%.6f (79/4096=%.6f)' % ('PASS' if ok_m else 'FAIL', p, 79 / 4096.0))
    lines.append('SELFTEST holm_unit %s' % ('PASS' if ok_h else 'FAIL'))
    print(lines[-2]); print(lines[-1], flush=True)
    res['ALL_PASS'] = bool(ok_all and ok_st and ok_m and ok_h)
    dump_json(BASE + '/SELFTEST.json', res)
    with open(BASE + '/synthetic.log', 'a') as f:
        f.write('\n'.join(lines) + '\nSELFTEST ALL %s %s\n' % ('PASS' if res['ALL_PASS'] else 'FAIL', time.strftime('%F %T %Z')))
    print('SELFTEST ALL', 'PASS' if res['ALL_PASS'] else 'FAIL', flush=True)
    return res['ALL_PASS']


# ------------------------------------------------------------------ decide
def collect():
    it = jload(BASE + '/ITEMS_SGR.json')
    R, recs, missing, fails = {}, {}, {}, {}
    for b in ('main', 'sub', 'w4', 'naux'):
        for t in it['blocks'][b]:
            base = '%s/%s__%s' % (OUT, t, b)
            if os.path.exists(base + '.DONE'):
                r = jload(base + '.json')
                recs[(t, b)] = r
                for arm, v in r['arms'].items():
                    R.setdefault(arm, {})[t] = v.get('R')
            elif os.path.exists(base + '.FAIL'):
                fails.setdefault(b, []).append(t)
            else:
                missing.setdefault(b, []).append(t)
    return it, R, recs, missing, fails


def aux_tables(it, R, recs):
    out = {}
    blocks = it['blocks']
    for b, arms in (('main', MAIN_ARMS), ('sub', SUB_ARMS), ('w4', W4_ARMS), ('naux', NAUX_ARMS)):
        tags = blocks[b]
        cells = sorted(set(cell_of(t) for t in tags))
        tb = {}
        for arm in arms + (['SGR'] if b in ('sub', 'naux') else []):
            r, cm = repaired(R.get(arm, {}), tags)
            vals = [R.get(arm, {}).get(t) for t in tags]
            tb[arm] = {'cells_repaired': sum(1 for c in cells if r.get(c)), 'n_cells': len(cells),
                       'ckpt_R_ge_0.8': sum(1 for v in vals if v is not None and v >= THR), 'n_ckpt_with_value': sum(1 for v in vals if v is not None),
                       'cell_mean_R': {c: cm[c]['mean_R'] for c in cells}}
        out[b] = tb
    # re-screen flags and tuned fractions
    flags, frac = {}, {}
    for (t, b), r in recs.items():
        for arm, v in r['arms'].items():
            if 'flag_rescreen' in v:
                f = flags.setdefault(arm, [0, 0]); f[0] += int(v['flag_rescreen']); f[1] += 1
            tf = (v.get('tune') or {}).get('frac_tuned_of_inference_params')
            if tf is not None:
                frac.setdefault(arm, {}).setdefault(r['model'], []).append(tf)
    out['rescreen_flags'] = {a: {'flagged': v[0], 'n': v[1]} for a, v in flags.items()}
    out['tuned_fraction_by_model'] = {a: {m: [min(v), max(v)] for m, v in d.items()} for a, d in frac.items()}
    out['tuned_fraction_range'] = {a: [min(min(v) for v in d.values()), max(max(v) for v in d.values())] for a, d in frac.items()}
    ig = {'max_fp32_repro': max(r['integrity']['fp32_repro_absdiff'] for r in recs.values()),
          'max_B0_repro': max([r['arms']['B0']['repro_absdiff_F3_D_l50'] for r in recs.values() if 'B0' in r['arms']] or [0.0]),
          'E1_all_pass': all(r['equiv']['E1_pass'] for r in recs.values()),
          'state_restored_all': all(r['integrity']['state_restored_exact'] for r in recs.values()),
          'mask_preserved_all': all((v.get('tune') or {}).get('mask_preserved', True) and (v.get('b3') or {}).get('mask_preserved', True)
                                    for r in recs.values() for v in r['arms'].values()),
          'levels_ok_all': all((v.get('tune') or {}).get('levels_ok', True) and (v.get('b3') or {}).get('levels_ok', True)
                               for r in recs.values() for v in r['arms'].values()),
          'exact_count_all': all(v.get('exact_count') in (None, True) for r in recs.values() for v in r['arms'].values()),
          'overlap_zero_all': all(r['overlap']['n_sel_in_val'] == 0 and r['overlap']['n_sel_in_test'] == 0 for r in recs.values()),
          'gpu_s_total': sum(r.get('gpu_wall_s', 0.0) for r in recs.values())}
    out['integrity'] = ig
    return out


def write_md(res, path):
    h = res['judge']['hyp']
    L = ['# RESULT_SGR (decide, once)', '', 'time: %s' % res['time'], '',
         '| hypothesis | verdict | detail |', '|---|---|---|',
         '| H_R1 SGR repairs >= 39/55 | %s | %s |' % (h['H_R1']['verdict'], json.dumps({k: v for k, v in h['H_R1'].items() if k != 'verdict'})),
         '| H_R2 SGR > B1 (McNemar, Holm) | %s | %s |' % (h['H_R2']['verdict'], json.dumps({k: v for k, v in h['H_R2'].items() if k != 'verdict'})),
         '| H_R4 SGR >= 0.8 B4 (subset, Holm) | %s | %s |' % (h['H_R4']['verdict'], json.dumps({k: v for k, v in h['H_R4'].items() if k != 'verdict'})),
         '', 'Holm: %s' % json.dumps(res['judge']['holm']), '', 'Branches: %s' % ('; '.join(res['judge']['branches']) or 'none'), '',
         '## Cell counts (main 55)', '', '| arm | repaired |', '|---|---|']
    for a in MAIN_ARMS:
        L.append('| %s | %d/%d |' % (a, res['judge']['counts'][a]['repaired'], res['judge']['counts'][a]['n_cells']))
    L += ['', '## Auxiliary (checkpoint level, subset, w4, n)', '', '```', json.dumps({b: {a: {k: v for k, v in d.items() if k != 'cell_mean_R'} for a, d in t.items()}
                                                                                  for b, t in res['aux'].items() if b in ('main', 'sub', 'w4', 'naux')}, indent=1), '```',
          '', 'Re-screen flags (tau = 0.738, reported only): %s' % json.dumps(res['aux']['rescreen_flags']),
          '', 'Tuned-parameter fraction (by construction, H_R3 not a hypothesis): %s' % json.dumps(res['aux']['tuned_fraction_range']),
          '', 'Integrity: %s' % json.dumps(res['aux']['integrity']), '', 'Missing: %s; FAIL: %s' % (json.dumps({k: len(v) for k, v in res['missing'].items()}),
                                                                                                  json.dumps(res['fails']))]
    with open(path, 'w', encoding='utf8') as f:
        f.write('\n'.join(L) + '\n')


def main_decide():
    assert os.path.exists(BASE + '/out/ALL_DONE'), 'refuse: ALL_DONE missing'
    assert not os.path.exists(BASE + '/RESULT_SGR.json'), 'refuse: RESULT_SGR.json exists'
    it, R, recs, missing, fails = collect()
    complete = {'main': 'main' not in missing, 'sub': 'sub' not in missing}
    tm, ts = it['blocks']['main'], it['blocks']['sub']
    cm = sorted(set(cell_of(t) for t in tm)); cs = sorted(set(cell_of(t) for t in ts))
    j = judge(R, cm, tm, cs, ts, complete)
    res = {'time': time.strftime('%F %T %Z'), 'judge': j, 'aux': aux_tables(it, R, recs), 'missing': missing, 'fails': fails,
           'all_done': open(BASE + '/out/ALL_DONE').read().strip()}
    dump_json(BASE + '/RESULT_SGR.json', res)
    write_md(res, BASE + '/RESULT_SGR.md')
    print('DECIDE H_R1=%s H_R2=%s H_R4=%s' % (j['hyp']['H_R1']['verdict'], j['hyp']['H_R2']['verdict'], j['hyp']['H_R4']['verdict']))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--decide', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.decide:
        main_decide()
