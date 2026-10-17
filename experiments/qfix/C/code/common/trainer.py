"""Common trainer: one (model, dataset, split k) run under protocol v1.

The model package supplies an adapter with build / forward_train / extra_loss / infer.
Everything else (data, optimiser, schedule, checkpoint rule, metric, storage) is shared."""
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import data as D
from .losses import seg_loss
from .metrics import evaluate
from .provenance import config_hash, deployed_commit, environment, source_digest

# Code identity is fixed when the process starts, so a later deploy cannot relabel running jobs.
CODE = {'commit': deployed_commit(), 'source_sha256': source_digest()}

PROTOCOL = {
    'protocol': 'v1',
    'epochs': 100, 'batch_size': 16, 'lr': 1e-3, 'weight_decay': 1e-2, 'betas': [0.9, 0.999],
    'scheduler': 'cosine', 'eta_min': 1e-6, 'grad_clip': 5.0, 'num_workers': 4,
    'checkpoint_rule': 'max_val_mean_iou', 'threshold': 0.5, 'input_size': D.SIZE,
}


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _worker_init(wid):
    s = torch.initial_seed() % 2 ** 32
    random.seed(s); np.random.seed(s)


def run(adapter, model_cfg, dataset, k, out_dir, data_root, cache_root, device='cuda:0', protocol=None,
        skip_test=False):
    # skip_test (QFIX stage C, default False = protocol v1 unchanged): the official test split stays sealed in the
    # training job; no test loader is built and no test metric is computed (evaluated once later by the stage-C evaluator).
    proto = dict(PROTOCOL, **(protocol or {}))
    cfg = {'model': adapter.NAME, 'model_cfg': model_cfg, 'dataset': dataset, 'split': k,
           'seed': k, 'protocol': proto}
    chash = config_hash(cfg)
    res_path = os.path.join(out_dir, 'results.json')
    if os.path.exists(res_path):
        old = json.load(open(res_path))
        if old.get('config_hash') == chash and old.get('status') == 'complete':
            print(f'[skip] complete run with same config at {out_dir}', flush=True)
            return old
    os.makedirs(out_dir, exist_ok=True)
    seed_everything(k)

    caches = {s: D.build_cache(data_root, dataset, s, k, cache_root) for s in ('train', 'val', 'test')}
    g = torch.Generator(); g.manual_seed(k)
    tr = DataLoader(D.SegCache(caches['train'], True), batch_size=proto['batch_size'], shuffle=True,
                    drop_last=True, num_workers=proto['num_workers'], worker_init_fn=_worker_init,
                    generator=g, pin_memory=True, persistent_workers=proto['num_workers'] > 0)
    va = DataLoader(D.SegCache(caches['val'], False), batch_size=1, shuffle=False, num_workers=2)
    te = None if skip_test else DataLoader(D.SegCache(caches['test'], False), batch_size=1, shuffle=False, num_workers=2)

    model = adapter.build(model_cfg).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_inf = sum(p.numel() for n, p in model.named_parameters() if not adapter.is_train_only(n))
    opt = torch.optim.AdamW(model.parameters(), lr=proto['lr'], weight_decay=proto['weight_decay'],
                            betas=tuple(proto['betas']))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=proto['epochs'], eta_min=proto['eta_min'])
    print(f'[setup] {adapter.NAME} {dataset} k={k} params_total={n_total:,} params_inference={n_inf:,}', flush=True)

    hist, best, best_ep, t0 = [], -1.0, 0, time.time()
    ckpt = os.path.join(out_dir, 'checkpoint.pth')
    for ep in range(1, proto['epochs'] + 1):
        model.train(); tl = te_ = nb = 0
        for img, msk in tr:
            img, msk = img.to(device, non_blocking=True), msk.to(device, non_blocking=True)
            o = adapter.forward_train(model, img)
            lmain = seg_loss(o['out'], o.get('side'), msk)
            lext = adapter.extra_loss(o, msk, model_cfg)
            loss = lmain + lext
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite loss at epoch {ep}')
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), proto['grad_clip']); opt.step()
            tl += lmain.item(); te_ += float(lext); nb += 1
        sched.step()
        v, _ = evaluate(model, va, adapter, device)
        hist.append({'epoch': ep, 'loss_main': tl / nb, 'loss_extra': te_ / nb, **{f'val_{a}': b for a, b in v.items()}})
        if v['iou'] > best:
            best, best_ep = v['iou'], ep
            torch.save(model.state_dict(), ckpt)
        if ep <= 2 or ep % 10 == 0 or ep == proto['epochs']:
            print(f'[ep {ep:03d}] main={tl/nb:.4f} extra={te_/nb:.4f} val_iou={v["iou"]:.4f}', flush=True)
    train_min = (time.time() - t0) / 60
    model.load_state_dict(torch.load(ckpt, map_location=device))
    if skip_test:
        test, per_image = None, None
        print(f'[sealed] test not evaluated (best ep {best_ep}, {train_min:.1f} min)', flush=True)
    else:
        test, per_image = evaluate(model, te, adapter, device)
        print(f'[test] iou={test["iou"]:.4f} dice={test["dice"]:.4f} (best ep {best_ep}, {train_min:.1f} min)', flush=True)
    out = {'status': 'complete', 'config': cfg, 'config_hash': chash,
           'code': CODE,
           'env': environment(), 'data_cache': {s: os.path.basename(c) for s, c in caches.items()},
           'params_total': n_total, 'params_inference': n_inf,
           'best_epoch': best_ep, 'best_val_iou': best, 'test': test,
           'test_per_image_iou': per_image, 'test_names': [] if skip_test else D.SegCache(caches['test'], False).names,
           'train_minutes': train_min, 'history': hist}
    out['test_names'] = [str(n) for n in out['test_names']]
    if skip_test:
        out['test_sealed'] = True
    with open(res_path + '.tmp', 'w') as f:
        json.dump(out, f, indent=1)
    os.replace(res_path + '.tmp', res_path)
    return out
