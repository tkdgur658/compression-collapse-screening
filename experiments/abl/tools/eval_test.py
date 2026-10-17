"""Evaluate the sealed official test set ONCE for one ablation run (called only by run_decide.sh after ALL_DONE).
usage (cwd = CODE): eval_test.py ROW DS K CACHE   -> writes <run>/test_eval.json; refuses if it already exists.
Same path as common.trainer: load checkpoint.pth (best val IoU), common.metrics.evaluate, batch 1, threshold 0.5."""
import hashlib
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

row, ds, k, CACHE = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import data as D                 # noqa: E402
from common.metrics import evaluate          # noqa: E402
from models import registry                  # noqa: E402
import abl_common as A                       # noqa: E402

out = A.run_dir(row, ds, k)
dst = os.path.join(out, 'test_eval.json')
if os.path.exists(dst):
    raise SystemExit(f'refuse: {dst} exists (test is evaluated once)')
r = json.load(open(os.path.join(out, 'results.json')))
assert r['status'] == 'complete' and r['test'] == 'SEALED'
ad = registry.get(r['config']['model'])
model = ad.build(r['config']['model_cfg']).to('cuda:0')
ck = os.path.join(out, 'checkpoint.pth')
model.load_state_dict(torch.load(ck, map_location='cuda:0'), strict=True)
te_path = os.path.join(CACHE, 'v1', ds, r['data_cache']['test'])
te = DataLoader(D.SegCache(te_path, False), batch_size=1, shuffle=False, num_workers=2)
test, per = evaluate(model, te, ad, 'cuda:0')
res = {'row': row, 'dataset': ds, 'split': k, 'test': test, 'test_per_image_iou': per,
       'test_names': [str(n) for n in D.SegCache(te_path, False).names],
       'checkpoint_sha256': hashlib.sha256(open(ck, 'rb').read()).hexdigest(),
       'evaluated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
with open(dst + '.tmp', 'w') as f:
    json.dump(res, f, indent=1)
os.replace(dst + '.tmp', dst)
print('[eval_test]', row, ds, k, 'written', flush=True)
