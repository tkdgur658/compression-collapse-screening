import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)


rep('"""F3 QDEPLOY decision step', '"""W8A8 ADDENDUM decision step (PREREG_QDEPLOY_ADDENDUM_W8A8.md), copied from the F3 QDEPLOY decision step')
rep("ORIG = ['O_w4', 'O_a6', 'O_l50']\nDEPL = ['D_w4', 'D_a6', 'D_l50']\nSENS = ['S_w4g', 'S_w4c', 'S_a6c', 'S_l50c']",
    "ORIG = []\nDEPL = ['D_w8a8']\nSENS = []")
rep("""def main_decide():
    if not os.path.exists(Q.BASE + '/out/ALL_DONE'):
        sys.exit('refuse: out/ALL_DONE missing')
    if os.path.exists(Q.BASE + '/RESULT_QDEPLOY.json'):
        sys.exit('refuse: RESULT_QDEPLOY.json exists (decide runs once)')""",
    """def decide_w8a8(items, recs, fails, man):
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
    open(Q.BASE + '/RESULT_W8A8.md', 'w').write('\\n'.join(L) + '\\n')
    print('DECIDE W8A8 L80 all %s PBE %s' % (res['table']['L80']['all'], res['table']['L80']['pbe']))


def main_decide_f3_unused():
    if not os.path.exists(Q.BASE + '/out/ALL_DONE'):
        sys.exit('refuse: out/ALL_DONE missing')
    if os.path.exists(Q.BASE + '/RESULT_QDEPLOY.json'):
        sys.exit('refuse: RESULT_QDEPLOY.json exists (decide runs once)')""")
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
