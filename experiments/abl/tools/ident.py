"""Default-path identity probe. Run once with cwd = original code, once with cwd = snapshot; compare with ident_cmp.py.
usage: ident.py OUT.pt [gpu]   (CPU part: build state_dict, fixed-batch forward/loss/grads, checkpoint strict-load logits;
gpu part: val evaluation of the cvc k1 v1 checkpoint, saved but never printed)"""
import os as _ccs_os  # path configuration of the released copy (README, Configuration)
CCS_REPO_ROOT = _ccs_os.path.normpath(_ccs_os.path.join(_ccs_os.path.dirname(_ccs_os.path.abspath(__file__)), '../../..'))
CCS_WORK_ROOT = _ccs_os.environ.get('CCS_WORK', _ccs_os.path.join(CCS_REPO_ROOT, 'experiments'))
CCS_ARCHIVE_ROOT = _ccs_os.environ.get('CCS_ARCHIVE', _ccs_os.path.join(CCS_REPO_ROOT, 'archive'))
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(2)
from common import data as D                   # noqa: E402
from common.losses import seg_loss            # noqa: E402
from common.metrics import evaluate           # noqa: E402
from models import registry                   # noqa: E402
import abl_common as A                        # noqa: E402

CACHE = _ccs_os.path.join(CCS_ARCHIVE_ROOT, 'cache')
ref = A.ref_results('pbe', 'cvc', 1)
ad = registry.get('pbe')
cfg = dict(ad.DEFAULT_CFG)
res = {}
torch.manual_seed(1)
m = ad.build(cfg)
res['state_dict'] = {k: v.clone() for k, v in m.state_dict().items()}
tr = D.SegCache(os.path.join(CACHE, 'v1', 'cvc', ref['data_cache']['train']), False)
img = torch.stack([tr[i][0] for i in range(4)])
msk = torch.stack([tr[i][1] for i in range(4)])
m.train()
torch.manual_seed(123)
o = ad.forward_train(m, img)
loss = seg_loss(o['out'], o.get('side'), msk) + ad.extra_loss(o, msk, cfg)
loss.backward()
res['out'] = o['out'].detach().clone()
res['loss'] = loss.detach().clone()
res['grads'] = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
ck = torch.load(os.path.join(A.ref_dir('pbe', 'cvc', 1), 'checkpoint.pth'), map_location='cpu')
m2 = ad.build(cfg)
m2.load_state_dict(ck, strict=True)
m2.eval()
va = D.SegCache(os.path.join(CACHE, 'v1', 'cvc', ref['data_cache']['val']), False)
with torch.no_grad():
    res['ckpt_logits'] = ad.infer(m2, torch.stack([va[i][0] for i in range(4)]))
if len(sys.argv) > 2 and sys.argv[2] == 'gpu':
    from torch.utils.data import DataLoader
    m2 = m2.to('cuda:0')
    v, _ = evaluate(m2, DataLoader(va, batch_size=1, shuffle=False, num_workers=2), ad, 'cuda:0')
    res['val'] = v
torch.save(res, sys.argv[1])
print('[ident] saved', sys.argv[1], 'keys', len(res['state_dict']), flush=True)
