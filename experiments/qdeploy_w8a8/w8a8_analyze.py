# -*- coding: utf-8 -*-
"""W8A8 ADDENDUM decision step (PREREG_QDEPLOY_ADDENDUM_W8A8.md), copied from the F3 QDEPLOY decision step (PREREG_QDEPLOY.md sections 5-6). Pure python, no torch.
  --selftest : section 7.3 reverse checks on synthetic tables (same functions) -> appends to synthetic.log
  --decide   : ONCE, after out/ALL_DONE (run_decide.sh enforces) -> RESULT_QDEPLOY.json / RESULT_QDEPLOY.md
"""
import os
import sys
import json
import glob
import math
import random
import argparse

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qdeploy as Q                                         # constants only (torch is imported lazily there)

ORIG = []
DEPL = ['D_w8a8']
SENS = []
DEN = {s: ('train' if Q.SPEC[s][0] == 'train' else 'dep') for s in Q.SETTINGS}
IOU_REPRO = 1e-3
AUC_MARGIN = 0.02


def auc(pos, neg):
    """Mann-Whitney AUC, ties 0.5 (larger score = collapse)."""
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    r, i, n = {}, 0, len(allv)
    rs = 0.0
    while i < n:
        j = i
        while j + 1 < n and allv[j + 1][0] == allv[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            if allv[k][1] == 1:
                rs += avg
        i = j + 1
    np_, nn_ = len(pos), len(neg)
    return (rs - np_ * (np_ + 1) / 2.0) / (np_ * nn_)


def _rank(v):
    idx = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and v[idx[j + 1]] == v[idx[i]]:
            j += 1
        for k in range(i, j + 1):
            r[idx[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(a, b):
    n = len(a)
    if n < 3:
        return None
    ra, rb = _rank(a), _rank(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in ra)); sb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    if sa == 0 or sb == 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / (sa * sb)


def scores(c, ref_fg):
    """collapse-oriented scores: 1-agree, c_fg, -SQNR."""
    fq, fr = c['fg_q'], ref_fg
    cfg = 0.0 if (fq == 0 and fr == 0) else 1.0 - min(fq, fr) / max(fq, fr)
    return {'agree': 1.0 - c['agree'], 'fg': cfg, 'sqnr': -c['sqnr_db']}


def ckpt_rows(recs, settings, excl_ds):
    """one row per (checkpoint, setting): R, scores, split."""
    rows = []
    for r in recs:
        for s in settings:
            if s not in DEPL + SENS or r['tag'] not in excl_ds:
                pass
            if (s in DEPL or s in SENS) and r['tag'] in excl_ds:
                continue
            c = r['cells'][s]
            g = DEN[s]
            f = r['fp32'][g]['val_dice']
            R = c['val_dice'] / f if f > 0 else None
            if R is None:
                continue
            rows.append({'tag': r['tag'], 'model': r['model'], 'ds': r['ds'], 'k': r['k'], 'setting': s, 'R': R,
                         'score': scores(c, r['fp32'][g]['test8_fg'])})
    return rows


def cell_table(rows):
    cells = {}
    for x in rows:
        cells.setdefault((x['model'], x['ds'], x['setting']), []).append(x['R'])
    return {k: {'mean_R': sum(v) / len(v), 'n_splits': len(v)} for k, v in cells.items()}


def counts(cells, thr, settings):
    sel = {k: v for k, v in cells.items() if k[2] in settings}
    col = {k for k, v in sel.items() if v['mean_R'] < thr}
    pbe = [k for k in sel if k[0] == 'pbe']
    return {'all': '%d/%d' % (len(col), len(sel)), 'n_all': len(col), 'den_all': len(sel),
            'pbe': '%d/%d' % (sum(1 for k in col if k[0] == 'pbe'), len(pbe)), 'n_pbe': sum(1 for k in col if k[0] == 'pbe'), 'den_pbe': len(pbe),
            'pbe_partial': any(v['n_splits'] < 5 for k, v in sel.items() if k[0] == 'pbe'),
            'baseline': '%d/%d' % (sum(1 for k in col if k[0] != 'pbe'), len(sel) - len(pbe)),
            'by_setting': {s: sum(1 for k in col if k[2] == s) for s in settings},
            'pbe_by_setting': {s: sum(1 for k in col if k[2] == s and k[0] == 'pbe') for s in settings},
            'collapsed_cells': sorted('%s|%s|%s' % k for k in col)}


def ckpt_counts(rows, thr):
    return '%d/%d' % (sum(1 for x in rows if x['R'] < thr), len(rows))


def branch(n_pbe, partial):
    if partial:
        return 'PARTIAL'
    return 'artifact' if n_pbe <= 3 else 'survives'


def split_auc(rows, thr, key):
    per = {}
    excluded = 0
    for k in range(1, 6):
        rs = [x for x in rows if x['k'] == k]
        pos = [x['score'][key] for x in rs if x['R'] < thr]
        neg = [x['score'][key] for x in rs if x['R'] >= thr]
        a = auc(pos, neg)
        if a is None:
            excluded += 1
        else:
            per[k] = a
    if not per:
        return {'mean': None, 'min': None, 'max': None, 'per_split': {}, 'n_split_excluded': excluded}
    v = list(per.values())
    return {'mean': sum(v) / len(v), 'min': min(v), 'max': max(v), 'per_split': per, 'n_split_excluded': excluded}


def pooled_auc_ci(rows, thr, key, n_boot=2000, seed=Q.SEED):
    pos = [x['score'][key] for x in rows if x['R'] < thr]
    neg = [x['score'][key] for x in rows if x['R'] >= thr]
    a = auc(pos, neg)
    by = {}
    for x in rows:
        by.setdefault(x['tag'], []).append(x)
    tags = sorted(by)
    rng = random.Random(seed)
    bs = []
    for _ in range(n_boot):
        p, n = [], []
        for _t in range(len(tags)):
            for x in by[tags[rng.randrange(len(tags))]]:
                (p if x['R'] < thr else n).append(x['score'][key])
        b = auc(p, n)
        if b is not None:
            bs.append(b)
    bs.sort()
    ci = (bs[int(0.025 * len(bs))], bs[min(len(bs) - 1, int(0.975 * len(bs)))]) if bs else (None, None)
    return {'auc': a, 'ci95': ci, 'n_pos': len(pos), 'n_neg': len(neg)}


def per_setting_auc(rows, thr, key, settings):
    out = {}
    for s in settings:
        rs = [x for x in rows if x['setting'] == s]
        pos = [x['score'][key] for x in rs if x['R'] < thr]
        neg = [x['score'][key] for x in rs if x['R'] >= thr]
        out[s] = ('-(%d)' % len(pos)) if len(pos) < 5 else auc(pos, neg)
    return out


def read_rule(a_agree, a_other):
    if a_agree['mean'] is None or a_other['mean'] is None:
        return 'not testable'
    d = a_agree['mean'] - a_other['mean']
    common = [k for k in a_agree['per_split'] if k in a_other['per_split']]
    allpos = len(common) == 5 and all(a_agree['per_split'][k] - a_other['per_split'][k] > 0 for k in common)
    if d <= AUC_MARGIN:
        return 'indistinguishable (diff %.3f <= %.2f)' % (d, AUC_MARGIN)
    if allpos:
        return 'agreement higher (diff %.3f, positive in 5/5 splits)' % d
    return 'numbers only (diff %.3f, not positive in all splits)' % d


def q_block(rows, settings):
    out = {}
    for lab, thr in (('L80', 0.8), ('L90', 0.9)):
        b = {}
        for key in ('agree', 'fg', 'sqnr'):
            b[key] = {'split': split_auc(rows, thr, key), 'pooled': pooled_auc_ci(rows, thr, key),
                      'per_setting': per_setting_auc(rows, thr, key, settings)}
        b['rule_agree_vs_fg'] = read_rule(b['agree']['split'], b['fg']['split'])
        b['rule_agree_vs_sqnr'] = read_rule(b['agree']['split'], b['sqnr']['split'])
        out[lab] = b
    out['spearman'] = {s: {key: spearman([x['score'][key] for x in rows if x['setting'] == s], [x['R'] for x in rows if x['setting'] == s])
                           for key in ('agree', 'fg', 'sqnr')} for s in settings}
    return out


def inclusion(items, recs_by_tag, fails, man):
    """exclusion bookkeeping: FAIL markers, manifest mismatch, old FAILED.txt rule, E1 failures."""
    excl, excl_ds, notes = {}, set(), {}
    for it in items:
        t = it['tag']
        if t not in recs_by_tag:
            excl[t] = 'FAIL marker' if t in fails else 'missing'
            continue
        r = recs_by_tag[t]
        if man.get(it['dir'] + '/checkpoint.pth') != r['ckpt_sha256'] or man.get(it['dir'] + '/results.json') != r['results_sha256']:
            excl[t] = 'manifest mismatch'
            continue
        if it.get('old_failed_txt'):
            key = '%s/%s/k%d' % (it['model'], it['ds'], it['k'])
            ref_ck, ref_rj = Q.OLD_FAILED[key]
            div = abs(r['fp32']['train']['val_iou'] - r['best_val_iou_stored'])
            ok = r['ckpt_sha256'] == ref_ck and r['results_sha256'] == ref_rj and div <= IOU_REPRO
            notes[t] = {'ckpt_sha_match_local_copy': r['ckpt_sha256'] == ref_ck, 'results_sha_match_table1_copy': r['results_sha256'] == ref_rj,
                        'val_iou_reeval': r['fp32']['train']['val_iou'], 'best_val_iou_stored': r['best_val_iou_stored'], 'abs_diff': div,
                        'included': ok}
            if not ok:
                excl[t] = 'old FAILED.txt rule not met'
                continue
        if not r['equiv']['E1_pass']:
            excl_ds.add(t)
    return excl, excl_ds, notes


def decide(items, recs, fails, man):
    by = {r['tag']: r for r in recs}
    excl, excl_ds, notes = inclusion(items, by, fails, man)
    use = [r for r in recs if r['tag'] not in excl]
    res = {'n_items': len(items), 'n_records': len(recs), 'n_used': len(use), 'excluded': excl,
           'E1_failed_excluded_from_D_S': sorted(excl_ds), 'old_failed_txt_runs': notes}
    rows = {'O': ckpt_rows(use, ORIG, excl_ds), 'D': ckpt_rows(use, DEPL, excl_ds)}
    for s in SENS:
        rows[s] = ckpt_rows(use, [s], excl_ds)
    tab = {}
    for lab, thr in (('L80', 0.8), ('L90', 0.9)):
        t = {}
        for b, sets in [('O', ORIG), ('D', DEPL)] + [(s, [s]) for s in SENS]:
            cells = cell_table(rows[b])
            t[b] = counts(cells, thr, sets)
            t[b]['ckpt_level'] = ckpt_counts(rows[b], thr)
        tab[lab] = t
    res['table_i'] = tab
    res['table_i_footnote'] = ('Manuscript "11/18" (PBE-Net): queue10 k1 reference checkpoints (sha differs from the v1 runs), split 1 only, '
                               'collapse = q IoU <= chance+0.05; not the label used here (5-split mean val-Dice retention < 0.8).')
    d = tab['L80']['D']
    res['branch_ii'] = {'N_PBE_D': d['n_pbe'], 'den': d['den_pbe'], 'partial': d['pbe_partial'],
                        'verdict': branch(d['n_pbe'], d['pbe_partial']),
                        'text': {'artifact': 'restate PBE-Net collapse as a product of the original setting; C1-C3 to appendix',
                                 'survives': 'PBE-Net vulnerability remains in the deployment graph; main Q table uses D, O becomes stress setting',
                                 'PARTIAL': 'PBE cells incomplete; report partial counts, design unchanged'}[branch(d['n_pbe'], d['pbe_partial'])]}
    res['iii'] = {'O': q_block(rows['O'], ORIG), 'D': q_block(rows['D'], DEPL)}
    # reporting extras
    res['equivalence'] = {'E1_max_fp64_by_model': {}, 'E2_rel_fp32_max_by_model': {}, 'mask_mismatch_max_by_model': {}}
    for r in recs:
        e = r['equiv']; m = r['model']
        if e['E1_max_abs_fp64'] is not None:
            res['equivalence']['E1_max_fp64_by_model'][m] = max(res['equivalence']['E1_max_fp64_by_model'].get(m, 0.0), e['E1_max_abs_fp64'])
        res['equivalence']['E2_rel_fp32_max_by_model'][m] = max(res['equivalence']['E2_rel_fp32_max_by_model'].get(m, 0.0), e['E2']['rel_fp32'])
        res['equivalence']['mask_mismatch_max_by_model'][m] = max(res['equivalence']['mask_mismatch_max_by_model'].get(m, 0.0), e['E2']['mask_mismatch_frac'])
    res['norm_left_by_model'] = {r['model']: {'train': r['norm_train'], 'dep': r['norm_dep'], 'n_folded': r['n_folded'],
                                              'params_inf_train': r['params_inf_train'], 'params_inf_dep': r['params_inf_dep']} for r in recs}
    res['fp32_dice_train_minus_dep_maxabs'] = max([abs(r['fp32']['train']['val_dice'] - r['fp32']['dep']['val_dice']) for r in recs] or [0])
    ivs = [abs(r['fp32']['train']['val_iou'] - r['best_val_iou_stored']) for r in recs if r.get('best_val_iou_stored') is not None]
    res['val_iou_reproduction_all'] = {'n': len(ivs), 'max_abs_diff': max(ivs or [0]), 'n_over_1e-3': sum(1 for v in ivs if v > IOU_REPRO)}
    res['empty_fp32_test_masks'] = sum(sum(1 for v in r['fp32']['dep']['test8_fg_per_img'] if v == 0) for r in recs)
    res['integrity'] = {'restored_all': all(all(r['state_restored_exact'].values()) for r in recs),
                        'gpu_wall_s_total': sum(r.get('gpu_wall_s', 0) for r in recs)}
    return res


def fmt(v):
    return '-' if v is None else ('%.3f' % v if isinstance(v, float) else str(v))


def write_md(res, path):
    L = ['# F3 QDEPLOY RESULT (decide run once)', '']
    for lab in ('L80', 'L90'):
        L += ['## Table (i) %s' % lab, '', '| set | all cells | PBE-Net | baselines | checkpoint-level |', '|---|---|---|---|---|']
        for b, v in res['table_i'][lab].items():
            L.append('| %s | %s | %s | %s | %s |' % (b, v['all'], v['pbe'], v['baseline'], v['ckpt_level']))
        L.append('')
    L += ['Footnote: ' + res['table_i_footnote'], '', '## Branch (ii)', '',
          'N_PBE(D) = %d/%d -> **%s**: %s' % (res['branch_ii']['N_PBE_D'], res['branch_ii']['den'], res['branch_ii']['verdict'], res['branch_ii']['text']), '',
          '## (iii) AUC (5-split mean [min, max]; pooled [95% CI])', '']
    for b in ('O', 'D'):
        for lab in ('L80', 'L90'):
            q = res['iii'][b][lab]
            L.append('- %s %s: ' % (b, lab) + '; '.join('%s %s [%s, %s] pooled %s [%s, %s]' % (
                k, fmt(q[k]['split']['mean']), fmt(q[k]['split']['min']), fmt(q[k]['split']['max']), fmt(q[k]['pooled']['auc']),
                fmt(q[k]['pooled']['ci95'][0]), fmt(q[k]['pooled']['ci95'][1])) for k in ('agree', 'fg', 'sqnr')))
            L.append('  - rule agree vs fg: %s; agree vs SQNR: %s' % (q['rule_agree_vs_fg'], q['rule_agree_vs_sqnr']))
    L += ['', '## Exclusions', '', json.dumps(res['excluded'], indent=1), 'E1 failed (D/S excluded): %s' % res['E1_failed_excluded_from_D_S'],
          '', 'Old FAILED.txt runs: ' + json.dumps(res['old_failed_txt_runs'], indent=1)]
    with open(path, 'w', encoding='utf8') as f:
        f.write('\n'.join(L) + '\n')


# ------------------------------------------------------------------ section 7.3 selftest
def selftest():
    lines, ok_all = [], True

    def chk(name, ok, msg):
        nonlocal ok_all
        ok_all = ok_all and ok
        lines.append('SYNTHETIC 7.3_%s %s: %s' % (name, 'PASS' if ok else 'FAIL', msg))
        print(lines[-1])
    for planted, want in ((2, 'artifact'), (5, 'survives')):
        rows = []
        for i, ds in enumerate(Q.DS_ORDER):
            for j, s in enumerate(DEPL):
                col = (i * 3 + j) < planted
                for k in range(1, 6):
                    rows.append({'tag': 'pbe__%s__k%d' % (ds, k), 'model': 'pbe', 'ds': ds, 'k': k, 'setting': s,
                                 'R': 0.5 if col else 0.95, 'score': {}})
        c = counts(cell_table(rows), 0.8, DEPL)
        v = branch(c['n_pbe'], c['pbe_partial'])
        chk('branch_%d_of_18' % planted, v == want and c['n_pbe'] == planted, 'N=%d -> %s (want %s)' % (c['n_pbe'], v, want))
    rng = random.Random(Q.SEED)
    rows = []
    for k in range(1, 6):
        for i in range(198):
            R = rng.random() * 1.2
            rows.append({'tag': 't%d_%d' % (k, i // 3), 'k': k, 'setting': 'x', 'R': R,
                         'score': {'agree': 1 - R ** 3 if R <= 1 else -R, 'fg': R, 'sqnr': rng.random()}})
    a1 = split_auc(rows, 0.8, 'agree')['mean']; a0 = split_auc(rows, 0.8, 'fg')['mean']; ar = split_auc(rows, 0.8, 'sqnr')['mean']
    chk('auc_monotone', a1 == 1.0 and a0 == 0.0 and abs(ar - 0.5) <= 0.1, 'score=1-R^3 -> %.3f (1), score=R -> %.3f (0), random -> %.3f (0.5+-0.1)' % (a1, a0, ar))
    hand = [{'k': 1, 'R': 0.5, 'score': {'s': 3}}, {'k': 1, 'R': 0.9, 'score': {'s': 1}}, {'k': 1, 'R': 0.95, 'score': {'s': 3}},
            {'k': 2, 'R': 0.5, 'score': {'s': 1}}, {'k': 2, 'R': 0.9, 'score': {'s': 2}},
            {'k': 3, 'R': 0.9, 'score': {'s': 1}}]
    h = split_auc(hand, 0.8, 's')
    # split1: pos {3}, neg {1,3} -> (1 + 0.5)/2 = 0.75; split2: pos {1}, neg {2} -> 0; split3: no positive -> excluded
    chk('split_mean_range', abs(h['mean'] - 0.375) < 1e-12 and h['min'] == 0.0 and h['max'] == 0.75 and h['n_split_excluded'] == 3,
        'mean=%.3f [%.2f, %.2f] excluded=%d (hand: 0.375 [0, 0.75], 3 excluded: splits 3,4,5)' % (h['mean'], h['min'], h['max'], h['n_split_excluded']))
    chk('ALL', ok_all, 'analysis selftest')
    with open(Q.BASE + '/synthetic.log', 'a') as f:
        f.write('\n'.join(lines) + '\n')
    return ok_all


def decide_w8a8(items, recs, fails, man):
    by = {r['tag']: r for r in recs}
    excl, excl_ds, notes = inclusion(items, by, fails, man)
    use = [r for r in recs if r['tag'] not in excl]
    res = {'n_items': len(items), 'n_records': len(recs), 'n_used': len(use), 'excluded': excl,
           'E1_failed_excluded_from_D_S': sorted(excl_ds), 'old_failed_txt_runs': notes}
    rows = ckpt_rows(use, DEPL, excl_ds)
    tab = {}
    for lab, thr in (('L80', 0.8), ('L90', 0.9)):
        t = counts(cell_table(rows), thr, DEPL)
        t['ckpt_level'] = ckpt_counts(rows, thr)
        npos = sum(1 for x in rows if x['R'] < thr)
        if npos == 0:
            t['auc'] = 'no positives, AUC undefined'
        else:
            b = {}
            for key in ('agree', 'fg', 'sqnr'):
                try:
                    b[key] = {'split': split_auc(rows, thr, key), 'pooled': pooled_auc_ci(rows, thr, key)}
                except Exception as ex:
                    b[key] = {'error': repr(ex)}
            t['auc'] = b
        tab[lab] = t
    res['table'] = tab
    res['integrity'] = {'restored_all': all(all(r['state_restored_exact'].values()) for r in recs),
                        'max_abs_level': max(r['cells']['D_w8a8']['integrity']['max_abs_level'] for r in recs),
                        'gpu_wall_s_total': sum(r.get('gpu_wall_s', 0) for r in recs)}
    res['predictions'] = {'P-W1 L80 <= 2/66 cells': tab['L80']['n_all'] <= 2, 'P-W2 PBE-Net 0/6 at L80': tab['L80']['n_pbe'] == 0}
    return res


def main_decide():
    if not os.path.exists(Q.BASE + '/out/ALL_DONE'):
        sys.exit('refuse: out/ALL_DONE missing')
    if os.path.exists(Q.BASE + '/RESULT_W8A8.json'):
        sys.exit('refuse: RESULT_W8A8.json exists (decide runs once)')
    items = Q.jload(Q.BASE + '/ITEMS.json')
    recs = [Q.jload(p) for p in sorted(glob.glob(Q.OUT + '/*.json'))]
    fails = set(os.path.basename(p)[:-5] for p in glob.glob(Q.OUT + '/*.FAIL'))
    res = decide_w8a8(items, recs, fails, Q.manifest())
    Q.dump_json(Q.BASE + '/RESULT_W8A8.json', res)
    L = ['# W8A8 ADDENDUM RESULT (decide run once)', '']
    for lab in ('L80', 'L90'):
        t = res['table'][lab]
        L.append('- %s: all %s, PBE-Net %s, baselines %s, checkpoint-level %s; collapsed cells: %s' % (
            lab, t['all'], t['pbe'], t['baseline'], t['ckpt_level'], ', '.join(t['collapsed_cells']) or 'none'))
        L.append('  - AUC: %s' % json.dumps(t['auc']))
    L += ['', 'Excluded: %s' % json.dumps(res['excluded']), 'E1 excluded: %s' % res['E1_failed_excluded_from_D_S'],
          'Integrity: %s' % json.dumps(res['integrity']), 'Predictions: %s' % json.dumps(res['predictions'])]
    open(Q.BASE + '/RESULT_W8A8.md', 'w').write('\n'.join(L) + '\n')
    print('DECIDE W8A8 L80 all %s PBE %s' % (res['table']['L80']['all'], res['table']['L80']['pbe']))


def main_decide_f3_unused():
    if not os.path.exists(Q.BASE + '/out/ALL_DONE'):
        sys.exit('refuse: out/ALL_DONE missing')
    if os.path.exists(Q.BASE + '/RESULT_QDEPLOY.json'):
        sys.exit('refuse: RESULT_QDEPLOY.json exists (decide runs once)')
    items = Q.jload(Q.BASE + '/ITEMS.json')
    recs = [Q.jload(p) for p in sorted(glob.glob(Q.OUT + '/*.json'))]
    fails = set(os.path.basename(p)[:-5] for p in glob.glob(Q.OUT + '/*.FAIL'))
    res = decide(items, recs, fails, Q.manifest())
    Q.dump_json(Q.BASE + '/RESULT_QDEPLOY.json', res)
    write_md(res, Q.BASE + '/RESULT_QDEPLOY.md')
    print('DECIDE branch(ii)=%s N_PBE(D)=%d/%d | O L80 PBE %s | D L80 all %s' % (
        res['branch_ii']['verdict'], res['branch_ii']['N_PBE_D'], res['branch_ii']['den'], res['table_i']['L80']['O']['pbe'], res['table_i']['L80']['D']['all']))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--decide', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.decide:
        main_decide()
