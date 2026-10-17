# -*- coding: utf-8 -*-
"""THEORY_Q decision (PREREG_THEORYQ.md section 5). Pure python, no torch. Runs ONCE after out/ALL_DONE.
Labels: F3 frozen functions (qdeploy_analyze.inclusion / ckpt_rows / cell_table) on the F3 records, unchanged.
Writes RESULT_THEORYQ.json / RESULT_THEORYQ.md; refuses if either exists or DECIDE_STARTED exists."""
import os
import sys
import json
import glob
import math

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import theoryq_core as C  # noqa: E402
Q, A = C.QD, C.QAN
OUTD = os.path.join(HERE, 'out')
RJ, RM, ST = HERE + '/RESULT_THEORYQ.json', HERE + '/RESULT_THEORYQ.md', HERE + '/DECIDE_STARTED'
THR = {'H_T1': 0.85, 'H_T3': 0.75}
FOLDED = ['pbe', 'maresunet', 'utanet', 'mk_unet', 'tinyunet', 'cmunext', 'lbunet']


def labels(use, excl_ds, setting):
    rows = A.ckpt_rows(use, [setting], excl_ds)
    cells = A.cell_table(rows)
    return ({(k[0], k[1]): v['mean_R'] < 0.8 for k, v in cells.items()},
            {r['tag']: r['R'] < 0.8 for r in rows})


def get(p, path):
    for k in path:
        p = p[k]
    return p


def main():
    for f in (RJ, RM, ST):
        if os.path.exists(f):
            sys.exit('refuse: %s exists (decide runs once)' % f)
    if not os.path.exists(OUTD + '/ALL_DONE'):
        sys.exit('refuse: out/ALL_DONE missing')
    open(ST, 'w').write('started\n')
    items = Q.jload(Q.BASE + '/ITEMS.json')
    recs = [Q.jload(p) for p in sorted(glob.glob(Q.OUT + '/*.json'))]
    fails = set(os.path.basename(p)[:-5] for p in glob.glob(Q.OUT + '/*.FAIL'))
    by = {r['tag']: r for r in recs}
    excl, excl_ds, _ = A.inclusion(items, by, fails, Q.manifest())
    use = [r for r in recs if r['tag'] not in excl]
    pred = {}
    for p in sorted(glob.glob(OUTD + '/*.json')):
        if p.endswith('.integrity.json'):
            continue
        x = Q.jload(p)
        pred[x['tag']] = x
    tq_fail = sorted(os.path.basename(p)[:-5] for p in glob.glob(OUTD + '/*.FAIL'))
    inc = [r['tag'] for r in use if r['tag'] in pred]
    res = {'n_f3_used': len(use), 'f3_excluded': excl, 'theoryq_fail': tq_fail, 'n_pred_used': len(inc)}

    def rows_of(path):
        return [{'model': pred[t]['model'], 'ds': pred[t]['ds'], 'tag': t, 'v': get(pred[t], path)} for t in inc]

    def cellstat(path, agg='mean'):
        rr = rows_of(path)
        return C.cell_mean(rr, 'v') if agg == 'mean' else C.cell_max(rr, 'v')
    lab = {}
    for s in ('S_w4g', 'D_w4', 'D_l50', 'O_w4'):
        lab[s] = labels(use, excl_ds, s)
    res['label_counts'] = {s: '%d/%d' % (sum(v[0].values()), len(v[0])) for s, v in lab.items()}
    # ---- primary
    prim = {'H_T1': (('dep', 'Z'), 'S_w4g'), 'H_T3': (('l50', 'Mstat'), 'D_l50')}
    out = {}
    for h, (path, s) in prim.items():
        sc = cellstat(path)
        b = C.cell_boot(sc, lab[s][0], null=THR[h])
        b05 = C.cell_boot(sc, lab[s][0], null=0.5)
        out[h] = {'predictor': '.'.join(path), 'setting': s, 'threshold': THR[h], 'auc': b['point'], 'ci95': b['ci95'],
                  'p_le_threshold': b['p_le_null'], 'p_le_0.5': b05['p_le_null'], 'n_pos': b['n_pos'], 'n_neg': b['n_neg'],
                  'PASS': bool(b['ci95'][0] >= THR[h])}
    hl = C.holm({h: out[h]['p_le_threshold'] for h in out})
    for h in out:
        out[h]['holm_adj_p'] = hl[h][0]; out[h]['holm_reject'] = hl[h][1]
    zpc = {t: pred[t]['dep']['Zpc'] for t in inc}
    nonfin = sum(pred[t]['dep']['n_nonfinite_tensors'] for t in inc)
    out['H_T2'] = {'n_ckpt': len(zpc), 'n_ckpt_Zpc_gt0': sum(1 for v in zpc.values() if v > 0), 'n_nonfinite_tensors': nonfin,
                   'n_ckpt_Zpc_all_targets_gt0': sum(1 for t in inc if pred[t]['Zpc_dep_all_targets'] > 0),
                   'PASS': all(v == 0 for v in zpc.values()) and nonfin == 0,
                   'D_w4_collapsed_cells': sorted('%s|%s' % k for k, v in lab['D_w4'][0].items() if v)}
    res['primary'] = out
    # ---- auxiliary cell-level AUCs
    aux = {}
    specs = {'Zprime_S_w4g': (('dep', 'Zprime'), 'S_w4g'), 'Zw_S_w4g': (('dep', 'Zw'), 'S_w4g'), 'Zglob_S_w4g': (('dep', 'Zglob'), 'S_w4g'),
             'Zelem_S_w4g': (('dep', 'Zelem'), 'S_w4g'), 'Z_all_targets_S_w4g': (('Z_dep_all_targets',), 'S_w4g'),
             'logparams_S_w4g': (('log10_params_inf_dep',), 'S_w4g'), 'ZO_O_w4': (('train', 'Z'), 'O_w4'),
             'E_D_l50': (('l50', 'E'), 'D_l50'), 'Msum_D_l50': (('l50', 'Msum'), 'D_l50'), 'logparams_D_l50': (('log10_params_inf_dep',), 'D_l50'),
             'Z_D_w4': (('dep', 'Z'), 'D_w4')}
    for k, (path, s) in specs.items():
        b = C.cell_boot(cellstat(path), lab[s][0])
        aux[k] = {'auc': b['point'], 'ci95': b['ci95'], 'p_le_0.5': b['p_le_null']}
    for k, (path, s) in {'Z_max_agg_S_w4g': (('dep', 'Z'), 'S_w4g'), 'Mstat_max_agg_D_l50': (('l50', 'Mstat'), 'D_l50')}.items():
        b = C.cell_boot(cellstat(path, 'max'), lab[s][0])
        aux[k] = {'auc': b['point'], 'ci95': b['ci95']}
    # checkpoint level (q_robust cluster bootstrap)
    for k, (path, s) in {'ckpt_Z_S_w4g': (('dep', 'Z'), 'S_w4g'), 'ckpt_Mstat_D_l50': (('l50', 'Mstat'), 'D_l50'),
                         'ckpt_ZO_O_w4': (('train', 'Z'), 'O_w4')}.items():
        rr = [dict(x, label=lab[s][1][x['tag']]) for x in rows_of(path) if x['tag'] in lab[s][1]]
        aux[k] = C.ckpt_cluster_boot(rr, 'v')
    # within-model (cells)
    wm = {}
    for k, (path, s) in {'Z_S_w4g': (('dep', 'Z'), 'S_w4g'), 'Mstat_D_l50': (('l50', 'Mstat'), 'D_l50')}.items():
        sc = cellstat(path)
        wm[k] = {}
        for mdl in sorted(set(c[0] for c in sc)):
            ks = {c: sc[c] for c in sc if c[0] == mdl}
            lb = {c: lab[s][0][c] for c in ks}
            wm[k][mdl] = C.cell_auc(ks, lb) if 0 < sum(lb.values()) < len(lb) else '-(%d/%d)' % (sum(lb.values()), len(lb))
    aux['within_model'] = wm
    # ---- Z change, training -> deployed graph, vs new collapses (O_w4 -> S_w4g)
    zd, zt = cellstat(('dep', 'Z')), cellstat(('train', 'Z'))
    dz = {c: zd[c] - zt[c] for c in zd}
    risk = [c for c in dz if not lab['O_w4'][0][c]]
    new = {c: lab['S_w4g'][0][c] for c in risk}
    a_all = C.cell_boot({c: dz[c] for c in risk}, new)
    rf = [c for c in risk if c[0] in FOLDED]
    a_f = C.cell_boot({c: dz[c] for c in rf}, {c: new[c] for c in rf})
    res['dZ'] = {'n_risk_cells': len(risk), 'n_new_collapse': sum(new.values()), 'auc_all_risk': a_all['point'], 'ci95_all_risk': a_all['ci95'],
                 'n_risk_folded': len(rf), 'n_new_collapse_folded': sum(new[c] for c in rf), 'auc_folded': a_f['point'], 'ci95_folded': a_f['ci95'],
                 'n_cells_dZ_gt0': sum(1 for c in dz if dz[c] > 0), 'n_cells_dZ_lt0': sum(1 for c in dz if dz[c] < 0),
                 'n_new_with_dZ_gt0': sum(1 for c in risk if new[c] and dz[c] > 0),
                 'n_risk_nonnew_with_dZ_gt0': sum(1 for c in risk if not new[c] and dz[c] > 0),
                 'new_cells': {'%s|%s' % c: {'dZ': dz[c], 'Z_dep': zd[c], 'Z_train': zt[c]} for c in risk if new[c]}}
    res['aux'] = aux
    # ---- excluded tensors per model, harmless, timing, near-threshold
    ex = {}
    for t in inc:
        p = pred[t]; m = p['model']
        e = ex.setdefault(m, {'dep_unexecuted': set(), 'dep_train_only': set(), 'train_unexecuted': set(), 'dep_targets': p['n_targets']['dep'],
                              'harmless_channels_max': 0, 'harmless_in_zeroed_max': 0, 'bn_off_max': 0, 'autograd_fail': 0})
        for kk in ('dep_unexecuted', 'dep_train_only', 'train_unexecuted'):
            e[kk] |= set(p['excluded'][kk])
        e['harmless_channels_max'] = max(e['harmless_channels_max'], p['n_harmless_channels_dep_kept'])
        e['harmless_in_zeroed_max'] = max(e['harmless_in_zeroed_max'], p['dep']['n_harmless_in_zeroed'])
        e['bn_off_max'] = max(e['bn_off_max'], p['n_bn_channels_switched_off'])
        e['autograd_fail'] += int(not all(p['autograd_ok'].values()))
    res['excluded_by_model'] = {m: {k: (len(v) if isinstance(v, set) else v) for k, v in e.items()} for m, e in ex.items()}
    res['excluded_names_by_model'] = {m: {k: sorted(v) for k, v in e.items() if isinstance(v, set)} for m, e in ex.items()}
    res['near_threshold_channels_total'] = sum(pred[t]['dep']['n_near_threshold_channels'] + pred[t]['train']['n_near_threshold_channels'] for t in inc)
    res['cpu_wall_s_total'] = sum(pred[t]['cpu_wall_s'] for t in inc)
    res['cell_Z_dep'] = {'%s|%s' % c: v for c, v in sorted(zd.items())}
    res['cell_Mstat'] = {'%s|%s' % c: v for c, v in sorted(cellstat(('l50', 'Mstat')).items())}
    Q.dump_json(RJ, res)
    L = ['# RESULT_THEORYQ (decide run once)', '', 'labels: %s' % res['label_counts'], '']
    for h in ('H_T1', 'H_T3'):
        o = out[h]
        L.append('- %s %s vs %s: AUC %.3f [%.3f, %.3f], threshold %.2f -> %s; p(AUC<=thr) %.4f, Holm adj %.4f (%s); p(AUC<=0.5) %.4f; pos/neg %d/%d' % (
            h, o['predictor'], o['setting'], o['auc'], o['ci95'][0], o['ci95'][1], o['threshold'], 'PASS' if o['PASS'] else 'FAIL',
            o['p_le_threshold'], o['holm_adj_p'], 'reject' if o['holm_reject'] else 'keep', o['p_le_0.5'], o['n_pos'], o['n_neg']))
    t2 = out['H_T2']
    L.append('- H_T2 Zpc = 0: %s (%d/%d ckpt with Zpc > 0, nonfinite %d)' % ('PASS' if t2['PASS'] else 'FAIL', t2['n_ckpt_Zpc_gt0'], t2['n_ckpt'], t2['n_nonfinite_tensors']))
    L += ['', '## aux', '```', json.dumps(aux, indent=1, default=str), '```', '', '## dZ', '```', json.dumps(res['dZ'], indent=1), '```',
          '', '## excluded by model', '```', json.dumps(res['excluded_by_model'], indent=1), '```']
    with open(RM, 'w', encoding='utf8') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L[:6]))


if __name__ == '__main__':
    main()
