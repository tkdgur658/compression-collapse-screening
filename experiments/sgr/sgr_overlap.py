# -*- coding: utf-8 -*-
"""PREREG_SGR pre-freeze check (B): the 100 matching (incl. the 20 diagnosis) TRAIN images selected per checkpoint do not
overlap the val or test images of the same split, by file name AND by sha256 of the image bytes.  Reads the `names` and
`images` keys only (masks never read).  CPU, numpy only.  -> OVERLAP_dedup.json (OVERLAP.json = first run before the duplicate-exclusion rule, kept)"""
import os as _ccs_os  # path configuration of the released copy (README, Configuration)
CCS_REPO_ROOT = _ccs_os.path.normpath(_ccs_os.path.join(_ccs_os.path.dirname(_ccs_os.path.abspath(__file__)), '../..'))
CCS_WORK_ROOT = _ccs_os.environ.get('CCS_WORK', _ccs_os.path.join(CCS_REPO_ROOT, 'experiments'))
import os, sys, json, hashlib
sys.dont_write_bytecode = True
import numpy as np
BASE = _ccs_os.path.join(CCS_WORK_ROOT, 'sgr')
sys.path.insert(0, BASE)
from sgr_run import seed_ck, train_order, N_MAIN, N_DIAG, jload, QD
import qdeploy as Q

items = {i['tag']: i for i in jload(QD + '/ITEMS.json')}
it = jload(BASE + '/ITEMS_SGR.json')
tags = sorted(set(t for b in it['blocks'].values() for t in b))
res = {'n_checkpoints': len(tags), 'per_ckpt': {}}
tot = {'sel_name_in_val': 0, 'sel_name_in_test': 0, 'sel_hash_in_val': 0, 'sel_hash_in_test': 0,
       'train_name_in_valtest': 0, 'train_hash_in_valtest': 0, 'n_sel': 0, 'n_train_excluded_dup': 0}
for t in tags:
    r = jload(items[t]['dir'] + '/results.json')
    dc = r['data_cache']
    root = '%s/%s' % (Q.CACHE_ROOT, items[t]['ds'])

    def load(f):
        z = np.load('%s/%s' % (root, f))
        names = [str(x) for x in z['names']]
        im = z['images']
        hs = [hashlib.sha256(np.ascontiguousarray(im[i]).tobytes()).hexdigest() for i in range(im.shape[0])]
        return names, hs
    trn, trh = load(dc['train'])
    vn, vh = load(dc['val'])
    tn, th = load(dc['test'])
    perm, n_dup = train_order(np, seed_ck(t), trn, trh, set(vn) | set(tn), set(vh) | set(th))
    sel = perm[:N_MAIN]
    assert len(sel) == N_MAIN
    sn = [trn[i] for i in sel]; sh = [trh[i] for i in sel]
    d = {'n_train': len(trn), 'n_val': len(vn), 'n_test': len(tn), 'n_sel': len(sel),
         'sel_name_in_val': sum(1 for x in sn if x in set(vn)), 'sel_name_in_test': sum(1 for x in sn if x in set(tn)),
         'sel_hash_in_val': sum(1 for x in sh if x in set(vh)), 'sel_hash_in_test': sum(1 for x in sh if x in set(th)),
         'train_name_in_valtest': sum(1 for x in trn if x in set(vn) | set(tn)),
         'train_hash_in_valtest': sum(1 for x in trh if x in set(vh) | set(th)),
         'sel_names_sha': hashlib.sha256('|'.join(sn).encode()).hexdigest(), 'n_train_excluded_dup': n_dup}
    res['per_ckpt'][t] = d
    for k in tot:
        tot[k] += d[k]
res['totals'] = tot
res['PASS'] = all(tot[k] == 0 for k in ('sel_name_in_val', 'sel_name_in_test', 'sel_hash_in_val', 'sel_hash_in_test'))
Q.dump_json(BASE + '/OVERLAP_dedup.json', res)
print('OVERLAP %s checkpoints=%d totals=%s' % ('PASS' if res['PASS'] else 'FAIL', len(tags), json.dumps(tot)))
