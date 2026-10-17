"""Single entry point: rows x datasets x splits under protocol v1.

  python main.py --suite main --rows all --datasets all --splits 1-5            # plan (default)
  python main.py --suite main --rows pbe --datasets etis --splits 1 --stage check
  python main.py --suite main --stage train --execute --slot 0 --nslot 2        # run a queue share

Queue order is split -> dataset -> row (row varies fastest), so an interrupted queue leaves
every row filled to the same split. A run is skipped only if results.json is complete and its
config hash matches; failed runs never produce a results.json with status 'complete'.
"""
import os as _ccs_os  # path configuration of the released copy (README, Configuration)
CCS_REPO_ROOT = _ccs_os.path.normpath(_ccs_os.path.join(_ccs_os.path.dirname(_ccs_os.path.abspath(__file__)), '../../../..'))
CCS_WORK_ROOT = _ccs_os.environ.get('CCS_WORK', _ccs_os.path.join(CCS_REPO_ROOT, 'experiments'))
CCS_DATA_ROOT = _ccs_os.environ.get('CCS_DATA', _ccs_os.path.join(CCS_REPO_ROOT, 'datasets'))
import argparse
import copy
import json
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from common import data as D  # noqa: E402

DEF_DATA = CCS_DATA_ROOT
DEF_CACHE = os.path.join(ROOT, 'cache')
DEF_RUNS = os.path.join(ROOT, 'runs')


def parse_splits(s):
    out = []
    for part in s.split(','):
        if '-' in part:
            a, b = part.split('-'); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def plan(args):
    rows = json.load(open(os.path.join(ROOT, 'configs', 'rows.json')))[args.suite]
    names = list(rows) if args.rows == 'all' else args.rows.split(',')
    dsets = list(D.DATASETS) if args.datasets == 'all' else args.datasets.split(',')
    for n in names:
        if n not in rows:
            raise SystemExit(f'row {n} not in suite {args.suite}')
    for d in dsets:
        if d not in D.DATASETS:
            raise SystemExit(f'unknown dataset {d}')
    jobs = []
    for k in parse_splits(args.splits):
        for d in dsets:
            for n in names:
                jobs.append((n, rows[n], d, k))
    return [j for i, j in enumerate(jobs) if i % args.nslot == args.slot]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--suite', default='main'); ap.add_argument('--rows', default='all')
    ap.add_argument('--datasets', default='all'); ap.add_argument('--splits', default='1-5')
    ap.add_argument('--stage', choices=['plan', 'check', 'train'], default='plan')
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--slot', type=int, default=0); ap.add_argument('--nslot', type=int, default=1)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--epochs', type=int, help='override (check/smoke only; recorded in config)')
    ap.add_argument('--data-root', default=DEF_DATA); ap.add_argument('--cache-root', default=DEF_CACHE)
    ap.add_argument('--run-root', default=DEF_RUNS)
    a = ap.parse_args()
    jobs = plan(a)
    print(f'[plan] suite={a.suite} jobs={len(jobs)} (slot {a.slot}/{a.nslot})')
    if a.stage == 'plan' or not a.execute:
        for n, r, d, k in jobs:
            print(f'  {n:28s} {d:7s} k={k}  model={r["model"]} cfg={r["cfg"]}')
        if a.stage != 'plan':
            print('[plan] add --execute to run')
        return

    from common import trainer
    from models import registry
    proto = {}
    if a.stage == 'check':
        proto['epochs'] = a.epochs or 1
    elif a.epochs:
        proto['epochs'] = a.epochs
    sub = 'checks' if a.stage == 'check' else a.suite
    for n, r, d, k in jobs:
        ad = registry.get(r['model'])
        cfg = copy.deepcopy(ad.DEFAULT_CFG); cfg.update(r['cfg'])
        out = os.path.join(a.run_root, sub, n, d, f'k{k}')
        print(f'[run] {n} {d} k={k} -> {out}', flush=True)
        try:
            trainer.run(ad, cfg, d, k, out, a.data_root, a.cache_root, a.device, proto)
        except Exception:
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, 'FAILED.txt'), 'w') as f:
                f.write(traceback.format_exc())
            print(f'[fail] {n} {d} k={k}\n{traceback.format_exc()}', flush=True)


if __name__ == '__main__':
    main()
