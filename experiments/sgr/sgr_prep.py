# -*- coding: utf-8 -*-
"""PREREG_SGR §2: checkpoint list by rule from the frozen F3 rows (qdeploy/out/items, qdeploy_analyze functions unchanged).
CPU, pure python, no torch. Writes ITEMS_SGR.json (blocks, run order) and prints counts / names only (no R values).
  main : all splits of the D_l50 collapsed cells (5-split mean val-Dice retention < 0.8)
  sub  : main restricted to {pbe, MOST, FEWEST}; MOST / FEWEST = baseline with the most / fewest D_l50 collapsed cells
         (ties -> qdeploy BASELINES order)
  w4   : all splits of the S_w4g collapsed cells of {pbe, MOST, FEWEST}
  naux : pbe checkpoints of main (n = 10, 30 auxiliary)"""
import os as _ccs_os  # path configuration of the released copy (README, Configuration)
CCS_REPO_ROOT = _ccs_os.path.normpath(_ccs_os.path.join(_ccs_os.path.dirname(_ccs_os.path.abspath(__file__)), '../..'))
CCS_WORK_ROOT = _ccs_os.environ.get('CCS_WORK', _ccs_os.path.join(CCS_REPO_ROOT, 'experiments'))
import os, sys, json, glob, hashlib
sys.dont_write_bytecode = True
QD = _ccs_os.path.join(CCS_WORK_ROOT, 'qdeploy')
OUTD = _ccs_os.path.join(CCS_WORK_ROOT, 'sgr')
sys.path.insert(0, QD)
import qdeploy as Q
import qdeploy_analyze as A

items = Q.jload(QD + '/ITEMS.json')
recs = [Q.jload(p) for p in sorted(glob.glob(Q.OUT + '/*.json'))]
fails = set(os.path.basename(p)[:-5] for p in glob.glob(Q.OUT + '/*.FAIL'))
by = {r['tag']: r for r in recs}
excl, excl_ds, _ = A.inclusion(items, by, fails, Q.manifest())
use = [r for r in recs if r['tag'] not in excl]
it_by = {i['tag']: i for i in items}


def collapsed(setting):
    rows = A.ckpt_rows(use, [setting], excl_ds)
    c = A.counts(A.cell_table(rows), 0.8, [setting])
    cells = [tuple(x.split('|')[:2]) for x in c['collapsed_cells']]
    tags = sorted(x['tag'] for x in rows if (x['model'], x['ds']) in set(cells))
    return cells, tags, c['all']


cl50, t50, s50 = collapsed('D_l50')
cnt = {m: sum(1 for mm, _ in cl50 if mm == m) for m in Q.BASELINES}
MOST = [m for m in Q.BASELINES if cnt[m] == max(cnt.values())][0]
FEWEST = [m for m in Q.BASELINES if cnt[m] == min(cnt.values())][0]
SUBM = ['pbe', MOST, FEWEST]
cw4, tw4, sw4 = collapsed('S_w4g')
order = {m: i for i, m in enumerate(['pbe'] + Q.BASELINES)}
key = lambda t: (order[it_by[t]['model']], Q.DS_ORDER.index(it_by[t]['ds']), it_by[t]['k'])
blocks = {
    'main': sorted(t50, key=key),
    'sub': sorted([t for t in t50 if it_by[t]['model'] in SUBM], key=key),
    'w4': sorted([t for t in tw4 if it_by[t]['model'] in SUBM], key=key),
    'naux': sorted([t for t in t50 if it_by[t]['model'] == 'pbe'], key=key),
}
res = {'rule': {'collapse': '5-split mean val-Dice retention < 0.8 (qdeploy_analyze.counts, frozen)', 'MOST': MOST, 'FEWEST': FEWEST,
                'baseline_D_l50_cell_counts': cnt},
       'D_l50_cells': ['%s|%s' % c for c in cl50], 'S_w4g_cells_sub': ['%s|%s' % c for c in cw4 if c[0] in SUBM],
       'counts': {'D_l50': s50, 'S_w4g': sw4, **{b: len(v) for b, v in blocks.items()}},
       'blocks': blocks, 'excluded_F3': excl,
       'src_sha': {'ITEMS.json': Q.sha_file(QD + '/ITEMS.json'), 'qdeploy_analyze.py': Q.sha_file(QD + '/qdeploy_analyze.py')}}
os.makedirs(OUTD, exist_ok=True)
Q.dump_json(OUTD + '/ITEMS_SGR.json', res)
print(json.dumps({k: res[k] for k in ('rule', 'counts', 'S_w4g_cells_sub')}, indent=1))
print('ITEMS_SGR.json sha256', Q.sha_file(OUTD + '/ITEMS_SGR.json'))
