"""Smoke + timing for full and the 4 variants (cwd = snapshot, CUDA_VISIBLE_DEVICES=1).
Checks: module swapped, param counts, finite loss and grad norm for 55 AdamW steps on preloaded isic18 k1 train batches
(batch 16, train augmentation); timing = mean ms/step of steps 6-55 (pure compute, batches preloaded on GPU).
Prints no validation or test metric. Writes BASE/smoke.json."""
import os as _ccs_os  # path configuration of the released copy (README, Configuration)
CCS_REPO_ROOT = _ccs_os.path.normpath(_ccs_os.path.join(_ccs_os.path.dirname(_ccs_os.path.abspath(__file__)), '../../..'))
CCS_WORK_ROOT = _ccs_os.environ.get('CCS_WORK', _ccs_os.path.join(CCS_REPO_ROOT, 'experiments'))
CCS_ARCHIVE_ROOT = _ccs_os.environ.get('CCS_ARCHIVE', _ccs_os.path.join(CCS_REPO_ROOT, 'archive'))
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(2)
from common import data as D                   # noqa: E402
from common.losses import seg_loss            # noqa: E402
from models import registry                   # noqa: E402
from models.pbe.src import net as N           # noqa: E402
import abl_common as A                        # noqa: E402

CACHE = _ccs_os.path.join(CCS_ARCHIVE_ROOT, 'cache')
ad = registry.get('pbe')
ref = A.ref_results('pbe', 'isic18', 1)
ds = D.SegCache(os.path.join(CACHE, 'v1', 'isic18', ref['data_cache']['train']), True)
g = torch.Generator()
g.manual_seed(1)
torch.manual_seed(1)
dl = DataLoader(ds, batch_size=16, shuffle=True, drop_last=True, num_workers=0, generator=g)
batches = []
for i, (x, y) in enumerate(dl):
    batches.append((x.cuda(), y.cuda()))
    if len(batches) == 8:
        break
rows = {'full': {}, **A.VARIANTS}
out = {}
for row, over in rows.items():
    cfg = dict(ad.DEFAULT_CFG)
    cfg.update(over)
    torch.manual_seed(1)
    m = ad.build(cfg).cuda()
    mods = list(m.modules())
    info = {'cfg_override': over,
            'n_PRCM': sum(isinstance(x, N.PRCM) for x in mods), 'n_SEGate': sum(isinstance(x, N.SEGate) for x in mods),
            'block_s1': type(m.base.encoder1[0][2]).__name__, 'block_s2': type(m.base.encoder2[0][2]).__name__,
            'skip': m.skip, 'lambda_s': cfg['lambda_s'],
            'params_total': sum(p.numel() for p in m.parameters()),
            'params_inference': sum(p.numel() for n, p in m.named_parameters() if not ad.is_train_only(n)),
            'gate_params': sum(p.numel() for x in mods if isinstance(x, (N.PRCM, N.SEGate)) for p in x.parameters())}
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2, betas=(0.9, 0.999))
    finite, times, extra_zero = True, [], True
    m.train()
    for step in range(55):
        x, y = batches[step % len(batches)]
        torch.cuda.synchronize()
        t0 = time.time()
        o = ad.forward_train(m, x)
        lm = seg_loss(o['out'], o.get('side'), y)
        le = ad.extra_loss(o, y, cfg)
        loss = lm + le
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step == 0:
            info['proj_grad_none'] = all(getattr(m, f'proj{i}').weight.grad is None for i in (1, 2, 3))
            info['shead_grad_none'] = all(getattr(m, f'shead{i}').weight.grad is None for i in (1, 2, 3))
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step()
        torch.cuda.synchronize()
        if step >= 5:
            times.append(time.time() - t0)
        finite &= bool(torch.isfinite(loss).item() and torch.isfinite(gn).item())
        extra_zero &= float(le) == 0.0
    info.update({'finite_loss_and_gradnorm_55_steps': finite, 'extra_loss_always_zero': extra_zero,
                 'ms_per_step': 1000 * sum(times) / len(times), 'peak_mem_GB': torch.cuda.max_memory_allocated() / 1e9})
    torch.cuda.reset_peak_memory_stats()
    out[row] = info
    print('[smoke]', row, json.dumps(info), flush=True)
    del m, opt
    torch.cuda.empty_cache()
for row in out:
    out[row]['ratio_vs_full'] = out[row]['ms_per_step'] / out['full']['ms_per_step']
ok = (out['full']['n_PRCM'] == 9 and out['full']['block_s1'] == 'EBPB' and out['full']['skip'] == 'predictive'
      and out['pbe_idskip']['skip'] == 'plain' and out['pbe_idskip']['proj_grad_none']
      and out['pbe_lam0']['extra_loss_always_zero'] and out['pbe_lam0']['shead_grad_none'] and not out['full']['extra_loss_always_zero']
      and out['pbe_mbv2']['block_s1'] == 'InvResBlock' and out['pbe_mbv2']['block_s2'] == 'InvResBlock'
      and out['pbe_se']['n_SEGate'] == 9 and out['pbe_se']['n_PRCM'] == 0
      and all(out[r]['finite_loss_and_gradnorm_55_steps'] for r in out))
out['ALL_OK'] = ok
json.dump(out, open(os.path.join(A.BASE, 'smoke.json'), 'w'), indent=1)
print('[smoke] ratios', {r: round(out[r]['ratio_vs_full'], 3) for r in rows}, 'ALL_OK', ok, flush=True)
sys.exit(0 if ok else 1)
