# Post hoc, descriptive: output-agreement AUC at the CELL level (same unit as the THEORYQ weight predictors),
# for S_w4g and D_l50, with the same cell-cluster bootstrap as q_robust.py. Frozen F3 records and functions, read-only.
import os as _ccs_os  # path configuration of the released copy (README, Configuration)
CCS_REPO_ROOT = _ccs_os.path.normpath(_ccs_os.path.join(_ccs_os.path.dirname(_ccs_os.path.abspath(__file__)), '../..'))
CCS_WORK_ROOT = _ccs_os.environ.get('CCS_WORK', _ccs_os.path.join(CCS_REPO_ROOT, 'experiments'))
import os, sys, json, glob, random
sys.dont_write_bytecode = True
QD = _ccs_os.path.join(CCS_WORK_ROOT, 'qdeploy')
sys.path.insert(0, QD)
import qdeploy as Q
import qdeploy_analyze as A

items = Q.jload(QD + '/ITEMS.json')
recs = [Q.jload(p) for p in sorted(glob.glob(Q.OUT + '/*.json'))]
fails = set(os.path.basename(p)[:-5] for p in glob.glob(Q.OUT + '/*.FAIL'))
by = {r['tag']: r for r in recs}
excl, excl_ds, _ = A.inclusion(items, by, fails, Q.manifest())
use = [r for r in recs if r['tag'] not in excl]
out = {}
for s in ('S_w4g', 'D_l50', 'D_w4', 'D_a6'):
    rows = A.ckpt_rows(use, [s], excl_ds)
    cells = {}
    for x in rows:
        cells.setdefault((x['model'], x['ds']), []).append(x)
    keys = sorted(cells)
    lab = {k: sum(x['R'] for x in cells[k]) / len(cells[k]) < 0.8 for k in keys}
    sc = {k: sum(x['score']['agree'] for x in cells[k]) / len(cells[k]) for k in keys}   # mean (1 - agreement)

    def auc_of(ks):
        pos = [sc[k] for k in ks if lab[k]]; neg = [sc[k] for k in ks if not lab[k]]
        return A.auc(pos, neg) if pos and neg else None
    rng = random.Random(20261017); vals = []
    for _ in range(2000):
        v = auc_of([keys[rng.randrange(len(keys))] for _ in keys])
        if v is not None:
            vals.append(v)
    vals.sort()
    out[s] = {'cell_auc': auc_of(keys), 'ci95': [vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]],
              'n_pos': sum(lab.values()), 'n_neg': len(keys) - sum(lab.values())}
json.dump(out, open(_ccs_os.path.join(CCS_WORK_ROOT, 'qdeploy_tau/AGREE_CELL_AUC.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
