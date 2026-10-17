# -*- coding: utf-8 -*-
"""QFIX stage C1: PBE-Net compression-aware training (CAT), B_fix_design.md section 3 / 6.1.

Adapter object passed to the (unchanged except `skip_test`) protocol-v1 trainer `common.trainer.run`.
Each training batch draws one mode from {fp32, w4_pt, a6, l50} (uniform, own RNG Random(7919+k)); epochs 1-10 are
fp32 only.  Compressed batches apply the RECORD operators (imported unchanged from the stage-A copy `rec/qfix_a.py`,
sha256 b36df82f...) wrapped in a straight-through estimator, with every BatchNorm in running-statistics mode
(no running-stat update; affine parameters still trained).  fp32 batches call the original `pbe` forward_train.

usage (cwd = BASE/code):  python -B ../qfix_c_cat.py --ds D --k K
"""
import os as _ccs_os  # path configuration of the released copy (README, Configuration)
CCS_REPO_ROOT = _ccs_os.path.normpath(_ccs_os.path.join(_ccs_os.path.dirname(_ccs_os.path.abspath(__file__)), '../../..'))
CCS_WORK_ROOT = _ccs_os.environ.get('CCS_WORK', _ccs_os.path.join(CCS_REPO_ROOT, 'experiments'))
CCS_ARCHIVE_ROOT = _ccs_os.environ.get('CCS_ARCHIVE', _ccs_os.path.join(CCS_REPO_ROOT, 'archive'))
CCS_DATA_ROOT = _ccs_os.environ.get('CCS_DATA', _ccs_os.path.join(CCS_REPO_ROOT, 'datasets'))
import os
import sys
import json
import random
import hashlib
import argparse
import importlib.util

sys.dont_write_bytecode = True
BASE = _ccs_os.path.join(CCS_WORK_ROOT, 'qfix/C')
CODE = BASE + '/code'
REC = BASE + '/rec/qfix_a.py'
REC_SHA = 'b36df82ffd238ccd2fadee1b02ab7df016e1bcc6500fc2c865c6adbc52fa94c6'
DATA = CCS_DATA_ROOT
CACHE = _ccs_os.path.join(CCS_ARCHIVE_ROOT, 'cache')
V1_ROOTS = [_ccs_os.path.join(CCS_ARCHIVE_ROOT, 'runs/main/pbe'), _ccs_os.path.join(CCS_WORK_ROOT, 'pbe_v1_fill/runs/pbe')]
MODES = ['fp32', 'w4_pt', 'a6', 'l50']
CAT = {"modes": ["fp32", "w4_pt", "a6", "l50"], "probs": [0.25, 0.25, 0.25, 0.25], "warmup_epochs": 10,
       "bn_compressed": "running", "ste": "identity", "l50_mask_every": "each_l50_batch", "mode_seed": "7919+k",
       "op_source": "qfix_a.py b36df82f"}
MODE_SEED_BASE = 7919

if CODE not in sys.path:
    sys.path.insert(0, CODE)
import torch  # noqa: E402
import models.pbe.adapter as P  # noqa: E402


def _load_rec():
    with open(REC, 'rb') as f:
        assert hashlib.sha256(f.read()).hexdigest() == REC_SHA, 'record operator file changed'
    spec = importlib.util.spec_from_file_location('qfix_a_rec', REC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


R = _load_rec()        # R.rtn_sym_tensor, R.act_hook_factory, R.prune_layer = record operators (unchanged)

NAME = 'pbe_cat'
DEFAULT_CFG = dict(P.DEFAULT_CFG, cat=CAT)
build = P.build
extra_loss = P.extra_loss
is_train_only = P.is_train_only
STATE = {}


def reset_state(k):
    STATE.clear()
    STATE.update(rng=random.Random(MODE_SEED_BASE + k), epoch=1, saw_eval=False, sets=None, force_mode=None,
                 mode_counts={m: 0 for m in MODES}, epoch_modes={})


def infer(model, img):
    STATE['saw_eval'] = True
    return model(img)


def _build_sets(model):
    leafcls = (torch.nn.modules.conv._ConvNd, torch.nn.Linear)
    leaf = [(n, m) for n, m in model.named_modules() if isinstance(m, leafcls)]
    targets = [n for n, q in model.named_parameters() if q.is_floating_point() and q.dim() >= 2]   # = Engine.targets
    called = set()
    hs = [m.register_forward_hook(lambda m_, i_, o_: called.add(id(m_))) for _, m in leaf]
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            p = next(model.parameters())
            model(torch.zeros(1, 3, 256, 256, device=p.device, dtype=p.dtype))
    finally:
        for h in hs:
            h.remove()
        model.train(was)
    l50 = [n + '.weight' for n, m in leaf if id(m) in called
           and getattr(m, 'weight', None) is not None and m.weight.dim() >= 2]                       # = Engine.executed
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    return {'leaf': leaf, 'targets': targets, 'l50': l50, 'bns': bns}


def _ste_hook(rec_hook):
    def h(mod_, inp, out):
        if not torch.is_tensor(out):
            return None
        q = rec_hook(mod_, inp, out.detach())
        return q + (out - out.detach())          # STE: forward value == record q exactly, gradient identity
    return h


def compressed_forward(model, img, mode, sets):
    """Record operator + STE; BatchNorm in running-statistics mode for the duration of the forward."""
    for m in sets['bns']:
        m.eval()
    handles = []
    try:
        params = dict(model.named_parameters())
        if mode == 'w4_pt':
            sub = {}
            for n in sets['targets']:
                w = params[n]
                q = R.rtn_sym_tensor(torch, w.detach(), 4)
                sub[n] = q + (w - w.detach())     # STE (= w + (q - w).detach() with exact forward value q)
            out, side, heads = torch.func.functional_call(model, sub, (img,), {'return_aux': True})
        elif mode == 'l50':
            sub = {}
            for n in sets['l50']:
                w = params[n]
                q = R.prune_layer(torch, w.detach(), 0.5)
                sub[n] = q + (w - w.detach())     # STE (= w + (q - w).detach() with exact forward value q)
            out, side, heads = torch.func.functional_call(model, sub, (img,), {'return_aux': True})
        elif mode == 'a6':
            for _, m in sets['leaf']:
                handles.append(m.register_forward_hook(_ste_hook(R.act_hook_factory(torch, 6))))
            out, side, heads = model(img, return_aux=True)
        else:
            raise ValueError(mode)
    finally:
        for h in handles:
            h.remove()
        if model.training:
            for m in sets['bns']:
                m.train()
    return {'out': out, 'side': side, 'heads': heads}


def forward_train(model, img):
    if STATE['sets'] is None:
        STATE['sets'] = _build_sets(model)
    if STATE['saw_eval']:
        STATE['epoch'] += 1
        STATE['saw_eval'] = False
    ep = STATE['epoch']
    if STATE['force_mode'] is not None:                       # smoke / timing only
        mode = STATE['force_mode']
    elif ep <= CAT['warmup_epochs']:
        mode = 'fp32'
    else:
        mode = STATE['rng'].choice(CAT['modes'])
    STATE['mode_counts'][mode] += 1
    em = STATE['epoch_modes'].setdefault(ep, {m: 0 for m in MODES})
    em[mode] += 1
    if mode == 'fp32':
        return P.forward_train(model, img)
    return compressed_forward(model, img, mode, STATE['sets'])


# ------------------------------------------------------------------ runner
def cache_path(D, ds, k, split):  # same key rule as common.data.build_cache, without building anything
    sd = os.path.join(DATA, D.DATASETS[ds][0], f'{split}_{k}')
    h = hashlib.sha256()
    for ip, mp in D.list_pairs(sd):
        for p in (ip, mp):
            h.update(f'{os.path.relpath(p, DATA)}|{os.stat(p).st_size}'.encode())
    return os.path.join(CACHE, 'v1', ds, f'{split}_{k}_{h.hexdigest()[:16]}.npz')


def v1_run(ds, k):
    for root in V1_ROOTS:
        rj = '%s/%s/k%d/results.json' % (root, ds, k)
        if os.path.exists(rj) and not os.path.exists('%s/%s/k%d/FAILED.txt' % (root, ds, k)):
            r = json.load(open(rj))
            if r.get('status') == 'complete':
                return r
    raise FileNotFoundError('no complete v1 run for %s k%d' % (ds, k))


def precheck(ds, k):
    from common import data as D
    v1 = v1_run(ds, k)
    errs = []
    for s in ('train', 'val', 'test'):
        p = cache_path(D, ds, k, s)
        if not os.path.exists(p):
            errs.append('missing cache ' + p)
        if os.path.basename(p) != v1['data_cache'][s]:
            errs.append('cache %s != v1 %s' % (os.path.basename(p), v1['data_cache'][s]))
    return errs, v1


def verify(out_dir, ds, k, v1):
    errs = []
    r = json.load(open(os.path.join(out_dir, 'results.json')))
    c = r['config']
    if r.get('status') != 'complete':
        errs.append('status')
    if c['model'] != NAME or c['model_cfg'] != DEFAULT_CFG or c['model_cfg'].get('cat') != CAT:
        errs.append('model/model_cfg')
    if (c['dataset'], c['split'], c['seed']) != (ds, k, k):
        errs.append('dataset/split/seed')
    if c['protocol'] != v1['config']['protocol']:
        errs.append('protocol != v1')
    if len(r['history']) != c['protocol']['epochs']:
        errs.append('history %d' % len(r['history']))
    if r['data_cache'] != v1['data_cache']:
        errs.append('data_cache != v1')
    if r['params_total'] != v1['params_total'] or r['params_inference'] != v1['params_inference']:
        errs.append('params != v1')
    if r.get('test') is not None or not r.get('test_sealed'):
        errs.append('test not sealed')
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True)
    ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = a.out or '%s/runs/%s/k%d' % (BASE, a.ds, a.k)
    errs, v1 = precheck(a.ds, a.k)
    if errs:
        print('[precheck] FAIL', errs, flush=True)
        sys.exit(3)
    print('[precheck] OK caches = v1', v1['data_cache'], flush=True)
    from common import trainer
    reset_state(a.k)
    adapter = sys.modules[__name__]
    trainer.run(adapter, DEFAULT_CFG, a.ds, a.k, out, DATA, CACHE, device='cuda:0', skip_test=True)
    sets = STATE['sets']
    log = {'mode_counts': STATE['mode_counts'], 'epoch_modes': STATE['epoch_modes'], 'epochs_seen': STATE['epoch'],
           'n_targets_w4': len(sets['targets']), 'n_leaf_a6': len(sets['leaf']), 'n_l50': len(sets['l50']),
           'n_bn': len(sets['bns']), 'cat': CAT, 'rec_sha256': REC_SHA,
           'self_sha256': hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()}
    with open(os.path.join(out, 'cat_log.json'), 'w') as f:
        json.dump(log, f, indent=1)
    errs = verify(out, a.ds, a.k, v1)
    print('[verify]', 'OK' if not errs else errs, 'mode_counts', STATE['mode_counts'], flush=True)
    sys.exit(4 if errs else 0)


if __name__ == '__main__':
    main()
