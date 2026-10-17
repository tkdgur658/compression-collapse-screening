"""pre  DS K ROW CODE DATA CACHE        : original caches for (DS,K) exist and equal the full v1 reference run's data_cache.
   post DS K ROW CODE DATA CACHE DIGEST : results.json of the ablation run is consistent (test must be SEALED).
Run with cwd = CODE (snapshot). Prints no validation or test metric."""
import hashlib
import json
import os
import sys

mode, ds, k, row, CODE, DATA, CACHE = sys.argv[1:8]
k = int(k)
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import data as D                       # noqa: E402
from common.provenance import config_hash          # noqa: E402
from models import registry                        # noqa: E402
import abl_common as A                             # noqa: E402

errs = []
ref = A.ref_results('pbe', ds, k)
names = [ref['data_cache'][s] for s in ('train', 'val', 'test')]


def cache_path(split):  # same key rule as common.data.build_cache, without building anything
    sd = os.path.join(DATA, D.DATASETS[ds][0], f'{split}_{k}')
    h = hashlib.sha256()
    for ip, mp in D.list_pairs(sd):
        for p in (ip, mp):
            h.update(f'{os.path.relpath(p, DATA)}|{os.stat(p).st_size}'.encode())
    return os.path.join(CACHE, 'v1', ds, f'{split}_{k}_{h.hexdigest()[:16]}.npz')


for s, n in zip(('train', 'val', 'test'), names):
    p = cache_path(s)
    if os.path.basename(p) != n:
        errs.append(f'cache name {os.path.basename(p)} != reference {n}')
    if not os.path.exists(p):
        errs.append(f'missing cache {p}')
ad = registry.get('pbe')
exp_mcfg = dict(ad.DEFAULT_CFG)
exp_mcfg.update(A.VARIANTS[row])
if ref['config']['model_cfg'] != ad.DEFAULT_CFG:
    errs.append('reference model_cfg != DEFAULT_CFG')
if mode == 'pre':
    print('[pre]', row, ds, k, 'OK' if not errs else errs, flush=True)
    sys.exit(1 if errs else 0)

digest = sys.argv[8]
out = A.run_dir(row, ds, k)
try:
    r = json.load(open(os.path.join(out, 'results.json')))
except Exception as e:
    errs.append(f'cannot read results.json: {e}')
    r = None
if r is not None:
    c = r['config']
    if r.get('status') != 'complete':
        errs.append('status != complete')
    if c['model'] != 'pbe':
        errs.append('model != pbe')
    if c['model_cfg'] != exp_mcfg:
        errs.append(f'model_cfg {c["model_cfg"]} != {exp_mcfg}')
    if c['protocol'] != ref['config']['protocol']:
        errs.append('protocol != full v1 reference')
    if (c['dataset'], c['split'], c['seed']) != (ds, k, k):
        errs.append('dataset/split/seed')
    if config_hash(c) != r['config_hash']:
        errs.append('config_hash not reproducible')
    if r['code']['source_sha256'] != digest:
        errs.append(f'code digest {r["code"]["source_sha256"]} != snapshot {digest}')
    if [r['data_cache'][s] for s in ('train', 'val', 'test')] != names:
        errs.append(f'data_cache {r["data_cache"]}')
    if len(r['history']) != c['protocol']['epochs']:
        errs.append(f'history length {len(r["history"])}')
    if r['test'] != 'SEALED':
        errs.append('test not sealed')
    if not os.path.exists(os.path.join(out, 'checkpoint.pth')):
        errs.append('no checkpoint')
rep = {'row': row, 'dataset': ds, 'split': k, 'ok': not errs, 'errors': errs,
       'params_inference': r and r['params_inference'], 'params_total': r and r['params_total'],
       'train_minutes': r and r['train_minutes'], 'config_hash': r and r['config_hash']}
json.dump(rep, open(os.path.join(out, 'verify_report.json'), 'w'), indent=1)
print('[post]', json.dumps(rep), flush=True)
sys.exit(1 if errs else 0)
