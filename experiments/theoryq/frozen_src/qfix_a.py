# -*- coding: utf-8 -*-
"""QFIX stage A: component-wise collapse attribution for PBE-Net v1 (PREREG_QFIX.md section 1; plan A_PLAN.md).

One GPU, batch 16, pure FP32 (TF32 off, cudnn deterministic).  Quantisation / pruning operators are copies of the record
code (9101 precision_probe.py rtn_sym_tensor + activation hook, 9601 prune_probe.py prunable_layers/prune_layer), restricted
to ONE component group at a time; everything else stays FP32.  The a6 hook uses a per-SAMPLE amax so that batching is
numerically the record's batch-1 per-forward dynamic per-tensor scale.

Modes:
  --run       : all 30 checkpoints, one JSON per run in --out, DONE/FAIL markers, restart-safe (skips DONE)
  --synthetic : planted-outlier reverse check on one checkpoint (A_PLAN.md section 5)
  --analyze   : pure-python aggregation of the per-run JSONs -> A_attribution.json (no torch)
"""
import os as _ccs_os  # path configuration of the released copy (README, Configuration)
CCS_REPO_ROOT = _ccs_os.path.normpath(_ccs_os.path.join(_ccs_os.path.dirname(_ccs_os.path.abspath(__file__)), '../../..'))
CCS_WORK_ROOT = _ccs_os.environ.get('CCS_WORK', _ccs_os.path.join(CCS_REPO_ROOT, 'experiments'))
CCS_ARCHIVE_ROOT = _ccs_os.environ.get('CCS_ARCHIVE', _ccs_os.path.join(CCS_REPO_ROOT, 'archive'))
import os
import sys
import io
import json
import glob
import time
import math
import copy
import hashlib
import argparse
import traceback

sys.dont_write_bytecode = True

WORK = _ccs_os.path.join(CCS_WORK_ROOT, 'qfix/A')
CODE_ROOT = WORK + '/code'
CACHE_ROOT = _ccs_os.path.join(CCS_ARCHIVE_ROOT, 'cache/v1')
RUN_ROOTS = [_ccs_os.path.join(CCS_ARCHIVE_ROOT, 'runs/main/pbe'), _ccs_os.path.join(CCS_WORK_ROOT, 'pbe_v1_fill/runs/pbe')]
DATASETS = ['etis', 'PH2', 'cvc', 'kvasir', 'isic17', 'isic18']
SETTINGS = ['w4_pt', 'a6', 'l50']
COMPONENTS = ['stem', 'feat', 'skip1', 'skip2', 'skip3', 'bhead', 'final',
              'enc2', 'enc3', 'enc4', 'enc5', 'bott', 'dec1', 'dec2', 'dec3', 'dec4']
PREREG_GROUPS = ['stem', 'feat', 'skip1', 'skip2', 'skip3', 'bhead', 'final']
N_AGREE = 8
BATCH = 16
COLLAPSE_RET = 0.8
SHARE_MIN = 0.70
TOPK = 2
CHANCE_MARGIN = 0.05
SYN_DS, SYN_K, SYN_COMP, SYN_MOD, SYN_BN, SYN_CH, SYN_SCALE = 'kvasir', 1, 'enc3', 'base.encoder3.0.1.conv_kxk', 'base.encoder3.0.1.bn_kxk', 0, 1000.0


def comp_of(name):
    """Module / parameter name -> component group (complete partition of PBENet; raises on anything unmapped)."""
    def under(p):
        return name == p or name.startswith(p + '.')
    if under('base.encoder1.0.2') or under('base.encoder2.0.2'):
        return 'feat'                               # eBPB feature blocks (stages 1-2)
    if under('base.encoder1'):
        return 'stem'                               # input stem: 1x1 conv 3->c0, RepConv, PRCM of stage 1
    for i in (2, 3, 4, 5):
        if under('base.encoder%d' % i):
            return 'enc%d' % i
    if under('base.bottleneck'):
        return 'bott'
    for i in (1, 2, 3, 4):
        if under('base.decoder%d' % i):
            return 'dec%d' % i
    for i in (1, 2, 3):
        if under('proj%d' % i):
            return 'skip%d' % i                     # predictive skip W_i
    if under('shead1') or under('shead2') or under('shead3') or name.startswith('base.gt_conv'):
        return 'bhead'                              # boundary heads (train-only shead_i) + deep-supervision heads
    if under('base.final'):
        return 'final'                              # final 1x1 head
    raise KeyError(name)


# ------------------------------------------------------------------ helpers
def sanitize(o):
    if isinstance(o, float):
        return o if (o == o and o not in (float('inf'), float('-inf'))) else None
    if isinstance(o, dict):
        return {k: sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    return o


def dump_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf8') as f:
        json.dump(sanitize(obj), f, indent=1, sort_keys=True, allow_nan=False)
    os.replace(tmp, path)


def list_runs():
    runs = []
    for root in RUN_ROOTS:
        for d in DATASETS:
            for kd in sorted(glob.glob('%s/%s/k*' % (root, d))):
                rj = kd + '/results.json'
                if not os.path.exists(rj) or os.path.exists(kd + '/FAILED.txt'):
                    continue
                with open(rj, encoding='utf8') as f:
                    r = json.load(f)
                if r.get('status') != 'complete':
                    continue
                runs.append({'ds': d, 'k': int(kd.rsplit('k', 1)[1]), 'dir': kd, 'data_cache': r['data_cache'],
                             'model_cfg': r['config']['model_cfg'], 'commit': r['code']['commit']})
    keys = [(x['ds'], x['k']) for x in runs]
    assert len(keys) == len(set(keys)) == 30, (len(keys), len(set(keys)))
    return runs


# ------------------------------------------------------------------ record operators (copies)
def rtn_sym_tensor(torch, w, bits):
    """9101 precision_probe.py make_ops.rtn_sym_tensor: per-tensor symmetric RTN, scale = max|w|/(2^(b-1)-1)."""
    qm = float(2 ** (bits - 1) - 1)
    amax = w.detach().abs().amax()
    scale = torch.where(torch.isfinite(amax) & (amax > 0), amax / qm, torch.ones_like(amax))
    return torch.clamp(torch.round(w / scale), -qm, qm) * scale


def act_hook_factory(torch, bits):
    """9101 precision_probe.py activation hook: dynamic symmetric per-tensor fake-quant of each Conv/Linear OUTPUT.
    Record ran batch 1 (amax of that forward); here amax is taken per sample = identical maths at batch > 1."""
    qm = float(2 ** (bits - 1) - 1)

    def hook(mod_, inp, out):
        if not torch.is_tensor(out):
            return None
        y = out
        amax = y.detach().abs().flatten(1).amax(1).view(-1, *([1] * (y.dim() - 1)))
        sc = torch.where(torch.isfinite(amax) & (amax > 0), amax / qm, torch.ones_like(amax))
        return torch.clamp(torch.round(y / sc), -qm, qm) * sc
    return hook


def prune_layer(torch, w, s):
    """9601 prune_probe.py prune_layer: layer-wise unstructured magnitude pruning, zero |w| <= k-th smallest, k=floor(s n)."""
    n = w.numel()
    k = int(s * n)
    if k < 1:
        return w.clone()
    thr = torch.kthvalue(w.detach().abs().reshape(-1).cpu(), k).values.to(w.device)
    return torch.where(w.abs() <= thr, torch.zeros_like(w), w)


# ------------------------------------------------------------------ model / data
def setup_torch():
    import torch
    torch.set_num_threads(1)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, CODE_ROOT)
    from models import registry
    return torch, registry


def load_npz(torch, path, need_masks):
    import numpy as np
    with open(path, 'rb') as f:
        raw = f.read()
    z = np.load(io.BytesIO(raw))
    imgs = torch.from_numpy(z['images'])
    msks = torch.from_numpy(z['masks']) if need_masks else None
    return imgs, msks, hashlib.sha256(raw).hexdigest()


class Engine(object):
    def __init__(self, torch, registry, run, dev):
        self.torch, self.dev, self.run = torch, dev, run
        self.ad = registry.get('pbe')
        assert run['model_cfg'] == self.ad.DEFAULT_CFG, run['model_cfg']
        p = run['dir'] + '/checkpoint.pth'
        with open(p, 'rb') as f:
            craw = f.read()
        self.ckpt_sha = hashlib.sha256(craw).hexdigest()
        state = torch.load(io.BytesIO(craw), map_location='cpu')
        if isinstance(state, dict) and 'model' in state and not any(k.startswith('base.') for k in state):
            state = state['model']
        model = self.ad.build(copy.deepcopy(self.ad.DEFAULT_CFG))
        model.load_state_dict(state, strict=True)
        self.model = model.to(dev).eval()
        self.params = list(self.model.named_parameters())
        self.orig = {n: q.detach().clone() for n, q in self.params}
        self.buf_orig = {n: b.detach().clone() for n, b in self.model.named_buffers()}
        self.mods = dict(self.model.named_modules())
        cls_leaf = (torch.nn.modules.conv._ConvNd, torch.nn.Linear)
        self.leaf = [(n, m) for n, m in self.model.named_modules() if isinstance(m, cls_leaf)]
        self.targets = [(n, q) for n, q in self.params if q.is_floating_point() and q.dim() >= 2]
        for n, _ in self.targets:
            comp_of(n[:-len('.weight')] if n.endswith('.weight') else n)
        for n, _ in self.leaf:
            comp_of(n)
        # integrity: every parameter (any dim) maps to exactly one component
        self.param_comp = {n: comp_of(n.rsplit('.', 1)[0]) if n.rsplit('.', 1)[0] in self.mods else comp_of(n)
                           for n, _ in self.params}
        dc = run['data_cache']
        vimgs, vmsks, self.val_sha = load_npz(torch, '%s/%s/%s' % (CACHE_ROOT, run['ds'], dc['val']), True)
        timgs, _, self.test_sha = load_npz(torch, '%s/%s/%s' % (CACHE_ROOT, run['ds'], dc['test']), False)
        self.vimgs = (vimgs.float() / 255.0).to(dev)
        self.vmsks = vmsks.float().to(dev)
        self.timgs = (timgs[:N_AGREE].float() / 255.0).to(dev)
        assert tuple(self.vimgs.shape[1:]) == (3, 256, 256) and tuple(self.vmsks.shape[1:]) == (1, 256, 256)
        assert self.timgs.shape[0] == N_AGREE
        fg = vmsks.flatten(1).sum(1).double()
        pix = 256.0 * 256.0
        self.chance_iou = max(float((fg == 0).double().mean()), float((fg / pix).mean()))
        # executed leaf modules (record 9601: forward hooks on the first image)
        called = set()
        hs = [m.register_forward_hook(lambda m_, i_, o_: called.add(id(m_))) for _, m in self.leaf]
        try:
            with torch.no_grad():
                self.ad.infer(self.model, self.vimgs[0:1])
        finally:
            for h in hs:
                h.remove()
        self.executed = called

    def restore(self):
        with self.torch.no_grad():
            for n, q in self.params:
                q.copy_(self.orig[n])
            for n, b in self.model.named_buffers():
                b.copy_(self.buf_orig[n])

    def forward_masks(self, imgs):
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, imgs.shape[0], BATCH):
                lg = self.ad.infer(self.model, imgs[i:i + BATCH]).float()
                out.append(torch.sigmoid(lg) > 0.5)
        return torch.cat(out, 0)

    def val_metrics(self, pred):
        p = pred.double()
        t = self.vmsks.double()
        tp = (p * t).flatten(1).sum(1)
        fp = (p * (1 - t)).flatten(1).sum(1)
        fn = ((1 - p) * t).flatten(1).sum(1)
        u = tp + fp + fn
        iou = self.torch.where(u > 0, tp / u.clamp(min=1.0), self.torch.ones_like(u))
        ds = 2 * tp + fp + fn
        dice = self.torch.where(ds > 0, 2 * tp / ds.clamp(min=1.0), self.torch.ones_like(ds))
        return float(dice.mean()), float(iou.mean()), float(p.flatten(1).mean(1).mean())

    @staticmethod
    def pair_dice(torch, a, b):
        a = a.double().flatten(1); b = b.double().flatten(1)
        s = a.sum(1) + b.sum(1)
        d = torch.where(s > 0, 2 * (a * b).sum(1) / s.clamp(min=1.0), torch.ones_like(s))
        return [float(v) for v in d]

    def apply(self, setting, comps):
        """Quantise / prune the given component set; returns (hook handles, integrity dict)."""
        torch = self.torch
        comps = set(comps)
        handles, integ = [], {}
        with torch.no_grad():
            if setting == 'w4_pt':
                tg = [(n, q) for n, q in self.targets if self.param_comp[n] in comps]
                for n, q in tg:
                    q.copy_(rtn_sym_tensor(torch, self.orig[n], 4))
                integ = {'n_tensors': len(tg), 'max_levels': max([int(torch.unique(q.detach()).numel()) for _, q in tg] or [0])}
                assert integ['max_levels'] <= 15
            elif setting == 'l50':
                layers = [(n, m) for n, m in self.leaf if comp_of(n) in comps and id(m) in self.executed
                          and getattr(m, 'weight', None) is not None and m.weight.dim() >= 2]
                zs = []
                for n, m in layers:
                    m.weight.copy_(prune_layer(torch, self.orig[n + '.weight'], 0.5))
                    zs.append(float((m.weight == 0).double().mean()))
                integ = {'n_layers': len(layers), 'min_zero_frac': min(zs) if zs else None}
            elif setting == 'a6':
                mods = [(n, m) for n, m in self.leaf if comp_of(n) in comps]
                for n, m in mods:
                    handles.append(m.register_forward_hook(act_hook_factory(torch, 6)))
                integ = {'n_hooked': len(mods), 'n_hooked_executed': sum(1 for _, m in mods if id(m) in self.executed)}
            else:
                raise ValueError(setting)
        return handles, integ

    def eval_cell(self, setting, comps, ref_test):
        handles, integ = self.apply(setting, comps)
        try:
            pv = self.forward_masks(self.vimgs)
            pt = self.forward_masks(self.timgs)
        finally:
            for h in handles:
                h.remove()
            self.restore()
        dice, iou, fg = self.val_metrics(pv)
        ag = self.pair_dice(self.torch, ref_test, pt)
        return {'val_dice': dice, 'val_iou': iou, 'val_pred_fg': fg, 'agree': sum(ag) / len(ag), 'agree_per_img': ag,
                'integrity': integ}

    def diagnostics(self):
        """FP32 statistics per component on the N_AGREE test images (A_PLAN.md section 4)."""
        torch = self.torch
        chmax = {}

        def mk(name):
            def h(m_, i_, o_):
                y = o_.detach().abs()
                v = y.amax(dim=[0] + list(range(2, y.dim()))) if y.dim() > 2 else y.amax(dim=0)
                chmax[name] = torch.maximum(chmax[name], v) if name in chmax else v
            return h
        hs = [m.register_forward_hook(mk(n)) for n, m in self.leaf if id(m) in self.executed]
        try:
            self.forward_masks(self.timgs)
        finally:
            for h in hs:
                h.remove()
        out = {}
        for c in COMPONENTS:
            mods = [n for n in chmax if comp_of(n) == c]
            ws = [(n, q) for n, q in self.targets if self.param_comp[n] == c]
            d = {'n_exec_modules': len(mods), 'n_weight_tensors': len(ws)}
            if mods:
                act_max, ratio, dead_q, dead_0, nch = 0.0, 0.0, 0, 0, 0
                for n in mods:
                    v = chmax[n].double()
                    am = float(v.max())
                    act_max = max(act_max, am)
                    med = float(v.median())
                    ratio = max(ratio, am / max(med, am * 1e-6) if am > 0 else 1.0)   # capped at 1e6
                    dead_q += int((v < am / (2 * 31.0)).sum())
                    dead_0 += int((v == 0).sum())
                    nch += v.numel()
                d.update({'act_max': act_max, 'ch_ratio': ratio, 'dead_act_a6': dead_q / nch, 'dead_act_zero': dead_0 / nch})
            if ws:
                wr, wo, dq, nc = 0.0, 0.0, 0, 0
                for n, q in ws:
                    w = self.orig[n].double()
                    am = float(w.abs().max())
                    rms = float(w.pow(2).mean().sqrt())
                    wr = max(wr, am)
                    wo = max(wo, am / rms if rms > 0 else 1.0)
                    cm = w.abs().flatten(1).amax(1)
                    dq += int((cm < am / (2 * 7.0)).sum())
                    nc += cm.numel()
                d.update({'w_range': wr, 'w_outlier': wo, 'dead_w_w4': dq / nc})
            out[c] = d
        return out

    def sweep(self, settings=SETTINGS):
        torch = self.torch
        t0 = time.time()
        self.restore()
        pv = self.forward_masks(self.vimgs)
        ref_test = self.forward_masks(self.timgs)
        d0, i0, f0 = self.val_metrics(pv)
        rec = {'fp32': {'val_dice': d0, 'val_iou': i0, 'val_pred_fg': f0, 'test8_fg': [float(v) for v in ref_test.double().flatten(1).mean(1)]},
               'cells': {}}
        for s in settings:
            rec['cells'][s] = {'all': self.eval_cell(s, COMPONENTS, ref_test)}
            for c in COMPONENTS:
                rec['cells'][s][c] = self.eval_cell(s, [c], ref_test)
        ok = all(torch.equal(q.data, self.orig[n]) for n, q in self.params)
        rec['state_restored_exact'] = bool(ok)
        rec['n_val'] = int(self.vimgs.shape[0])
        rec['chance_iou'] = self.chance_iou
        rec['wall_s'] = time.time() - t0
        return rec


def main_run(args):
    torch, registry = setup_torch()
    dev = torch.device('cuda:0')
    runs = list_runs()
    os.makedirs(args.out, exist_ok=True)
    t_all = time.time()
    for run in runs:
        tag = '%s_k%d' % (run['ds'], run['k'])
        if os.path.exists('%s/%s.DONE' % (args.out, tag)):
            continue
        try:
            t0 = time.time()
            eng = Engine(torch, registry, run, dev)
            rec = eng.sweep()
            rec['diag'] = eng.diagnostics()
            rec.update({'ds': run['ds'], 'k': run['k'], 'dir': run['dir'], 'ckpt_sha256': eng.ckpt_sha,
                        'val_cache_sha256': eng.val_sha, 'test_cache_sha256': eng.test_sha, 'data_cache': run['data_cache'],
                        'commit': run['commit'], 'torch': torch.__version__, 'gpu_wall_s': time.time() - t0})
            dump_json('%s/%s.json' % (args.out, tag), rec)
            with open('%s/%s.DONE' % (args.out, tag), 'w') as f:
                f.write('%.1f\n' % rec['gpu_wall_s'])
            a = rec['cells']
            print('[%s] %.1fs fp32=%.4f ret_all w4=%.3f a6=%.3f l50=%.3f restored=%s' % (
                tag, rec['gpu_wall_s'], rec['fp32']['val_dice'], a['w4_pt']['all']['val_dice'] / rec['fp32']['val_dice'],
                a['a6']['all']['val_dice'] / rec['fp32']['val_dice'], a['l50']['all']['val_dice'] / rec['fp32']['val_dice'],
                rec['state_restored_exact']), flush=True)
            del eng
            torch.cuda.empty_cache()
        except Exception:
            tb = traceback.format_exc()
            with open('%s/%s.FAIL' % (args.out, tag), 'w') as f:
                f.write(tb)
            print('[%s] FAIL %s' % (tag, tb.splitlines()[-1]), flush=True)
    print('RUN_END total_s=%.1f' % (time.time() - t_all), flush=True)


def main_synthetic(args):
    """Planted outlier (function-preserving in FP32): output channel SYN_CH of SYN_MOD scaled by SYN_SCALE, the following
    eval-mode BN running_mean x S, running_var x S^2.  Expected: FP32 val Dice unchanged; top-1 attributed component under
    w4_pt and a6 = SYN_COMP."""
    torch, registry = setup_torch()
    dev = torch.device('cuda:0')
    run = [r for r in list_runs() if r['ds'] == SYN_DS and r['k'] == SYN_K][0]
    eng = Engine(torch, registry, run, dev)
    base = eng.sweep(settings=['w4_pt', 'a6'])
    with torch.no_grad():
        m = eng.mods[SYN_MOD]; bn = eng.mods[SYN_BN]
        m.weight[SYN_CH].mul_(SYN_SCALE)
        bn.running_mean[SYN_CH].mul_(SYN_SCALE)
        bn.running_var[SYN_CH].mul_(SYN_SCALE ** 2)
    eng.orig = {n: q.detach().clone() for n, q in eng.params}          # planted model becomes the reference
    eng.buf_orig = {n: b.detach().clone() for n, b in eng.model.named_buffers()}
    pl = eng.sweep(settings=['w4_pt', 'a6'])
    res = {'ds': SYN_DS, 'k': SYN_K, 'planted_component': SYN_COMP, 'module': SYN_MOD, 'channel': SYN_CH, 'scale': SYN_SCALE,
           'fp32_dice_clean': base['fp32']['val_dice'], 'fp32_dice_planted': pl['fp32']['val_dice'], 'per_setting': {}}
    ok = abs(base['fp32']['val_dice'] - pl['fp32']['val_dice']) < 1e-3
    for s in ['w4_pt', 'a6']:
        f = pl['fp32']['val_dice']
        losses = {c: max(0.0, 1 - pl['cells'][s][c]['val_dice'] / f) for c in COMPONENTS}
        top = sorted(COMPONENTS, key=lambda c: -losses[c])
        tot = sum(losses.values())
        res['per_setting'][s] = {'top1': top[0], 'top1_loss': losses[top[0]], 'second': top[1], 'second_loss': losses[top[1]],
                                 'share_top1': losses[top[0]] / tot if tot > 0 else None,
                                 'ret_all_planted': pl['cells'][s]['all']['val_dice'] / f,
                                 'clean_loss_of_planted_comp': max(0.0, 1 - base['cells'][s][SYN_COMP]['val_dice'] / base['fp32']['val_dice']),
                                 'losses': losses}
        ok = ok and top[0] == SYN_COMP
    res['PASS'] = bool(ok)
    dump_json(args.out + '/synthetic.json', res)
    print('SYNTHETIC %s: fp32 dice clean=%.4f planted=%.4f | %s' % (
        'PASS' if ok else 'FAIL', res['fp32_dice_clean'], res['fp32_dice_planted'],
        ' | '.join('%s top1=%s loss=%.3f (2nd %s %.3f; clean loss of %s %.3f)' % (
            s, v['top1'], v['top1_loss'], v['second'], v['second_loss'], SYN_COMP, v['clean_loss_of_planted_comp'])
            for s, v in res['per_setting'].items())), flush=True)


# ------------------------------------------------------------------ analysis (pure python)
def _rank(v):
    idx = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and v[idx[j + 1]] == v[idx[i]]:
            j += 1
        for k in range(i, j + 1):
            r[idx[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None and x == x and y == y]
    n = len(pairs)
    if n < 3:
        return None, n
    ra = _rank([p[0] for p in pairs]); rb = _rank([p[1] for p in pairs])
    ma = sum(ra) / n; mb = sum(rb) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in ra)); sb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    if sa == 0 or sb == 0:
        return None, n
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / (sa * sb), n


def analyze(recs):
    cells = []
    for r in recs:
        f = r['fp32']['val_dice']
        for s in SETTINGS:
            cc = r['cells'][s]
            ret_all = cc['all']['val_dice'] / f
            ret = {c: cc[c]['val_dice'] / f for c in COMPONENTS}
            loss = {c: max(0.0, 1.0 - ret[c]) for c in COMPONENTS}
            L = sum(loss.values())
            top = sorted(COMPONENTS, key=lambda c: -loss[c])
            ci = r['fp32']['val_iou']; qi = cc['all']['val_iou']
            rec_collapse = (False if ci <= r['chance_iou'] + CHANCE_MARGIN else bool(qi <= r['chance_iou'] + CHANCE_MARGIN))
            cells.append({'ds': r['ds'], 'k': r['k'], 'setting': s, 'fp32_dice': f, 'ret_all': ret_all,
                          'agree_all': cc['all']['agree'], 'collapsed': ret_all < COLLAPSE_RET, 'record_chance_collapse': rec_collapse,
                          'ret': ret, 'loss': loss, 'agree': {c: cc[c]['agree'] for c in COMPONENTS}, 'sum_single_loss': L,
                          'loss_all': 1.0 - ret_all, 'additivity': (L / (1.0 - ret_all)) if ret_all < 1.0 else None,
                          'cell_top2': top[:2], 'cell_top2_share': (loss[top[0]] + loss[top[1]]) / L if L > 0 else None})
    col = [c for c in cells if c['collapsed']]
    out = {'n_runs': len(recs), 'n_cells': len(cells), 'n_collapsed': len(col),
           'n_collapsed_by_setting': {s: sum(1 for c in col if c['setting'] == s) for s in SETTINGS}}

    def pooled(cs):
        Lg = {g: sum(c['loss'][g] for c in cs) for g in COMPONENTS}
        tot = sum(Lg.values())
        top = sorted(COMPONENTS, key=lambda g: -Lg[g])[:TOPK]
        return {'pooled_loss': Lg, 'total': tot, 'top': top,
                'share_top': (sum(Lg[g] for g in top) / tot) if tot > 0 else 0.0,
                'share_top1': (Lg[top[0]] / tot) if tot > 0 else 0.0,
                'mean_cell_share_with_pooled_top': (sum(sum(c['loss'][g] for g in top) / c['sum_single_loss'] for c in cs if c['sum_single_loss'] > 0)
                                                    / max(1, sum(1 for c in cs if c['sum_single_loss'] > 0))),
                'median_cell_own_top2_share': (sorted(c['cell_top2_share'] for c in cs if c['cell_top2_share'] is not None)[
                    len([c for c in cs if c['cell_top2_share'] is not None]) // 2] if any(c['cell_top2_share'] is not None for c in cs) else None),
                'median_additivity': (sorted(c['additivity'] for c in cs if c['additivity'] is not None)[
                    len([c for c in cs if c['additivity'] is not None]) // 2] if any(c['additivity'] is not None for c in cs) else None),
                'n_cells': len(cs)}
    if col:
        prim = pooled(col)
        out['H_A'] = {'rule': 'pooled over all collapsed cells (ret_all<0.8): share of sum_g sum_cells max(0,1-R_g) carried by the 2 components with the largest pooled loss >= 0.70',
                      'primary': prim, 'verdict': 'PASS' if prim['share_top'] >= SHARE_MIN else 'FAIL'}
        out['H_A']['by_setting'] = {s: pooled([c for c in col if c['setting'] == s]) for s in SETTINGS if any(c['setting'] == s for c in col)}
    else:
        out['H_A'] = {'verdict': 'NOT TESTABLE (no collapsed cell)'}
    out['all_cells_by_setting'] = {s: pooled([c for c in cells if c['setting'] == s]) for s in SETTINGS}
    # diagnostics vs per-component loss (executed components only)
    keys = ['act_max', 'ch_ratio', 'dead_act_a6', 'dead_act_zero', 'w_range', 'w_outlier', 'dead_w_w4']
    rr = {r['ds'] + '_k%d' % r['k']: r for r in recs}
    sp = {}
    for scope, cs in (('all_cells', cells), ('collapsed_cells', col)):
        sp[scope] = {}
        for s in SETTINGS:
            sp[scope][s] = {}
            for kk in keys:
                xs, ys = [], []
                for c in cs:
                    if c['setting'] != s:
                        continue
                    dg = rr[c['ds'] + '_k%d' % c['k']]['diag']
                    for g in COMPONENTS:
                        if dg[g].get('n_exec_modules', 0) == 0 or kk not in dg[g]:
                            continue
                        xs.append(dg[g][kk]); ys.append(c['loss'][g])
                rho, n = spearman(xs, ys)
                sp[scope][s][kk] = {'rho': rho, 'n': n}
    out['diag_spearman'] = sp
    # agreement (label-free) vs retention over every single-component and all-component cell
    for s in SETTINGS:
        xs, ys = [], []
        for c in cells:
            if c['setting'] != s:
                continue
            xs.append(c['agree_all']); ys.append(c['ret_all'])
            for g in COMPONENTS:
                xs.append(c['agree'][g]); ys.append(c['ret'][g])
        out.setdefault('agree_vs_ret_spearman', {})[s] = dict(zip(('rho', 'n'), spearman(xs, ys)))
    out['cells'] = cells
    return out


def main_analyze(args):
    recs = []
    for p in sorted(glob.glob(args.out + '/*_k*.json')):
        with open(p, encoding='utf8') as f:
            recs.append(json.load(f))
    a = analyze(recs)
    syn = args.out + '/synthetic.json'
    if os.path.exists(syn):
        with open(syn, encoding='utf8') as f:
            a['synthetic'] = json.load(f)
    a['diag_by_run'] = {r['ds'] + '_k%d' % r['k']: r['diag'] for r in recs}
    a['integrity'] = {'state_restored_all': all(r['state_restored_exact'] for r in recs),
                      'ckpt_sha256': {r['ds'] + '_k%d' % r['k']: r['ckpt_sha256'] for r in recs},
                      'gpu_wall_s_total': sum(r['gpu_wall_s'] for r in recs),
                      'integrity_example': {s: {g: recs[0]['cells'][s][g]['integrity'] for g in ['all'] + COMPONENTS} for s in SETTINGS} if recs else None}
    dump_json(args.analysis, a)
    h = a['H_A']
    print('H_A', h['verdict'], json.dumps({k: v for k, v in h.get('primary', {}).items() if k != 'pooled_loss'}, default=str))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--out', default=WORK + '/out')
    ap.add_argument('--analysis', default=WORK + '/A_attribution.json')
    a = ap.parse_args()
    if a.synthetic:
        main_synthetic(a)
    if a.run:
        main_run(a)
    if a.analyze:
        main_analyze(a)
