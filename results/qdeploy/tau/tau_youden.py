# Post hoc (outside the pre-registration): operating threshold for the R2 screen (output agreement) in setting D.
# [internal decision] decision 2026-10-16: choose tau by maximum Youden J on split 1, report sensitivity / specificity on splits 2-5.
# Uses the frozen F3 QDEPLOY records and the frozen row/label functions unchanged (qdeploy_analyze.ckpt_rows, inclusion).
import os as _ccs_os  # path configuration of the released copy (README, Configuration)
CCS_REPO_ROOT = _ccs_os.path.normpath(_ccs_os.path.join(_ccs_os.path.dirname(_ccs_os.path.abspath(__file__)), '../../..'))
CCS_WORK_ROOT = _ccs_os.environ.get('CCS_WORK', _ccs_os.path.join(CCS_REPO_ROOT, 'experiments'))
import os, sys, json, glob
sys.dont_write_bytecode = True
QD = _ccs_os.path.join(CCS_WORK_ROOT, 'qdeploy')
sys.path.insert(0, QD)
import qdeploy as Q
import qdeploy_analyze as A

items = Q.jload(QD + '/ITEMS.json')
recs = [Q.jload(p) for p in sorted(glob.glob(Q.OUT + '/*.json'))]
fails = set(os.path.basename(p)[:-5] for p in glob.glob(Q.OUT + '/*.FAIL'))
by = {r['tag']: r for r in recs}
excl, excl_ds, notes = A.inclusion(items, by, fails, Q.manifest())
use = [r for r in recs if r['tag'] not in excl]
rows = A.ckpt_rows(use, A.DEPL, excl_ds)          # setting D: D_w4, D_a6, D_l50 per checkpoint
out = {'source': 'qdeploy/out/items (F3, frozen)', 'n_rows': len(rows), 'excluded': excl, 'E1_excluded': sorted(excl_ds)}
for lab, thr in (('L80', 0.8), ('L90', 0.9)):
    tr = [(x['score']['agree'], x['R'] < thr) for x in rows if x['k'] == 1]
    te = [(x['score']['agree'], x['R'] < thr, x['k']) for x in rows if x['k'] != 1]
    cands = sorted(set(s for s, _ in tr))
    P = sum(1 for _, y in tr if y); N = len(tr) - P
    best = None
    for t in cands:                                   # flag iff score >= t ; ties -> smallest t (earliest)
        tp = sum(1 for s, y in tr if y and s >= t); tn = sum(1 for s, y in tr if not y and s < t)
        J = tp / P + tn / N - 1
        if best is None or J > best[0] + 1e-15:
            best = (J, t, tp / P, tn / N)
    J, tau, se1, sp1 = best

    def ss(rr):
        p = [s for s, y, *_ in rr if y]; n = [s for s, y, *_ in rr if not y]
        return {'n_pos': len(p), 'n_neg': len(n), 'sens': sum(1 for s in p if s >= tau) / len(p) if p else None,
                'spec': sum(1 for s in n if s < tau) / len(n) if n else None}
    o = {'tau_score': tau, 'tau_agreement': 1 - tau, 'J_split1': J, 'sens_split1': se1, 'spec_split1': sp1,
         'n_split1': len(tr), 'pos_split1': P, 'test_splits_2_5': ss(te),
         'per_split': {str(k): ss([z for z in te if z[2] == k]) for k in (2, 3, 4, 5)}}
    out[lab] = o
json.dump(out, open(_ccs_os.path.join(CCS_WORK_ROOT, 'qdeploy_tau/TAU_RESULT.json'), 'w'), indent=1)
print(json.dumps({k: out[k] for k in ('L80', 'L90')}, indent=1))
