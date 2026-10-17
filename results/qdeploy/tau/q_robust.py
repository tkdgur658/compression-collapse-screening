# Journal review 2, item A-1 (post hoc, CPU, outside the pre-registration): robustness of fact 2 (label-free identification) in setting D.
# Uses the frozen F3 records and the frozen row/label/AUC functions unchanged (qdeploy_analyze). Writes Q_ROBUST.json only.
import os as _ccs_os  # path configuration of the released copy (README, Configuration)
CCS_REPO_ROOT = _ccs_os.path.normpath(_ccs_os.path.join(_ccs_os.path.dirname(_ccs_os.path.abspath(__file__)), '../../..'))
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
rows = A.ckpt_rows(use, A.DEPL, excl_ds)
THR = 0.8
KEYS = ('agree', 'fg', 'sqnr')
out = {'n_rows': len(rows), 'label': 'L80 (R < 0.8)', 'B': 2000, 'seed': 20261017}


def auc_rows(rr, key):
    pos = [x['score'][key] for x in rr if x['R'] < THR]
    neg = [x['score'][key] for x in rr if x['R'] >= THR]
    if not pos or not neg:
        return None
    return A.auc(pos, neg)


def cluster_boot(rr, key, B=2000, seed=20261017):
    cells = {}
    for x in rr:
        cells.setdefault((x['model'], x['ds']), []).append(x)
    keys = sorted(cells)
    rng = random.Random(seed)
    vals = []
    for _ in range(B):
        samp = []
        for _k in keys:
            samp += cells[keys[rng.randrange(len(keys))]]
        v = auc_rows(samp, key)
        if v is not None:
            vals.append(v)
    vals.sort()
    lo = vals[int(0.025 * (len(vals) - 1))]; hi = vals[int(0.975 * (len(vals) - 1))]
    return {'point': auc_rows(rr, key), 'ci95': [lo, hi], 'n_boot_valid': len(vals), 'n_cells': len(keys),
            'n_pos': sum(1 for x in rr if x['R'] < THR), 'n_neg': sum(1 for x in rr if x['R'] >= THR)}


# (a) per setting and pooled, with cell-level cluster bootstrap
out['auc'] = {}
for s in A.DEPL + ['pooled_D']:
    rr = rows if s == 'pooled_D' else [x for x in rows if x['setting'] == s]
    out['auc'][s] = {k: cluster_boot(rr, k) for k in KEYS}


# threshold helpers (score = 1 - agreement; flag iff score >= tau)
def youden(tr):
    P = sum(1 for x in tr if x['R'] < THR); N = len(tr) - P
    best = None
    for t in sorted(set(x['score']['agree'] for x in tr)):
        tp = sum(1 for x in tr if x['R'] < THR and x['score']['agree'] >= t)
        tn = sum(1 for x in tr if x['R'] >= THR and x['score']['agree'] < t)
        J = (tp / P if P else 0) + (tn / N if N else 0) - 1
        if best is None or J > best[0] + 1e-15:
            best = (J, t)
    return best[1]


def sens_spec(te, t):
    p = [x for x in te if x['R'] < THR]; n = [x for x in te if x['R'] >= THR]
    return {'n_pos': len(p), 'n_neg': len(n),
            'sens': (sum(1 for x in p if x['score']['agree'] >= t) / len(p)) if p else None,
            'spec': (sum(1 for x in n if x['score']['agree'] < t) / len(n)) if n else None}


tau0 = youden([x for x in rows if x['k'] == 1])
out['tau_split1_agreement'] = 1 - tau0
te0 = [x for x in rows if x['k'] != 1]
out['heldout_splits_2_5'] = sens_spec(te0, tau0)
# (b) LOMO / LODO: tau by Youden on split 1 of the other models (datasets); evaluated on the held-out group, splits 2-5
models = sorted(set(x['model'] for x in rows)); dss = sorted(set(x['ds'] for x in rows))
out['LOMO'] = {}
for m in models:
    t = youden([x for x in rows if x['k'] == 1 and x['model'] != m])
    out['LOMO'][m] = dict(sens_spec([x for x in rows if x['k'] != 1 and x['model'] == m], t), tau_agreement=1 - t)
out['LODO'] = {}
for d in dss:
    t = youden([x for x in rows if x['k'] == 1 and x['ds'] != d])
    out['LODO'][d] = dict(sens_spec([x for x in rows if x['k'] != 1 and x['ds'] == d], t), tau_agreement=1 - t)


def pooled(dct):
    P = sum(v['n_pos'] for v in dct.values()); N = sum(v['n_neg'] for v in dct.values())
    tp = sum((v['sens'] or 0) * v['n_pos'] for v in dct.values()); tn = sum((v['spec'] or 0) * v['n_neg'] for v in dct.values())
    return {'n_pos': P, 'n_neg': N, 'sens': tp / P if P else None, 'spec': tn / N if N else None}


out['LOMO_pooled'] = pooled(out['LOMO']); out['LODO_pooled'] = pooled(out['LODO'])
# (c) misses and false alarms of the split-1 threshold on splits 2-5
miss = [x for x in te0 if x['R'] < THR and x['score']['agree'] < tau0]
fa = [x for x in te0 if x['R'] >= THR and x['score']['agree'] >= tau0]


def comp(lst):
    c = {}
    for x in lst:
        c.setdefault(x['model'], {}).setdefault(x['setting'], 0)
        c[x['model']][x['setting']] += 1
    return c


out['missed'] = {'n': len(miss), 'by_model_setting': comp(miss), 'R_range': [min(x['R'] for x in miss), max(x['R'] for x in miss)] if miss else None}
out['false_alarms'] = {'n': len(fa), 'by_model_setting': comp(fa)}
# (d) per model flag counts, all splits, split-1 threshold
out['per_model_all_splits'] = {}
for m in models:
    rr = [x for x in rows if x['model'] == m]
    col = [x for x in rr if x['R'] < THR]; ok = [x for x in rr if x['R'] >= THR]
    out['per_model_all_splits'][m] = {'n': len(rr), 'collapsed': len(col), 'collapsed_flagged': sum(1 for x in col if x['score']['agree'] >= tau0),
                                       'intact': len(ok), 'intact_flagged': sum(1 for x in ok if x['score']['agree'] >= tau0)}
json.dump(out, open(_ccs_os.path.join(CCS_WORK_ROOT, 'qdeploy_tau/Q_ROBUST.json'), 'w'), indent=1)
print(json.dumps({k: out[k] for k in ('tau_split1_agreement', 'heldout_splits_2_5', 'LOMO_pooled', 'LODO_pooled')}, indent=0))
