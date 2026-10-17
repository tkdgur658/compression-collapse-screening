# -*- coding: utf-8 -*-
"""F3 QDEPLOY engine (PREREG_QDEPLOY.md): Q re-measured on the deployment graph (RepConv fused, BN folded).

Modes (cwd anywhere; CUDA_VISIBLE_DEVICES must expose exactly one GPU):
  --prep        : CPU, read-only. ITEMS.json (330 items, run order) + MANIFEST_330.sha256 (checkpoint.pth + results.json)
  --foldpairs   : GPU. per baseline model, etis k1, BN stats randomised (seed 7919), each traced Conv->BN pair folded alone
                  in float64; accepted iff max|dlogit| <= 1e-9 max|logit|  -> FOLD_PAIRS.json
  --synthetic   : GPU. reverse checks section 7.1, 7.2, 7.4 -> synthetic.log / synthetic.json (no Dice / R printed)
  --run [--only TAG,...] : per-item JSON + DONE/FAIL markers in out/items (restart-safe, skips DONE/FAIL),
                  stop rules (6 GPU-h compute cap, 2nd OOM, PBE sha / E1 failure), waits while a foreign process uses the GPU.
                  Values are sealed: the log shows wall time and integrity counters only.
Record operators are imported unchanged from qfix/C/rec/qfix_a.py; PBE fusion = f2_fuse.fuse_pbe, generic BN folding
helpers = f2b_fold_all.find_pairs / fold_into / set_child (both copied byte-identical from manuscript_cleanup/F2_efficiency).
"""
import os as _ccs_os  # path configuration of the released copy (README, Configuration)
CCS_REPO_ROOT = _ccs_os.path.normpath(_ccs_os.path.join(_ccs_os.path.dirname(_ccs_os.path.abspath(__file__)), '../..'))
CCS_WORK_ROOT = _ccs_os.environ.get('CCS_WORK', _ccs_os.path.join(CCS_REPO_ROOT, 'experiments'))
CCS_ARCHIVE_ROOT = _ccs_os.environ.get('CCS_ARCHIVE', _ccs_os.path.join(CCS_REPO_ROOT, 'archive'))
import os
import sys
import io
import json
import glob
import time
import copy
import math
import hashlib
import argparse
import traceback
import subprocess

sys.dont_write_bytecode = True

BASE = _ccs_os.path.join(CCS_WORK_ROOT, 'qdeploy')
CODE_ROOT = _ccs_os.path.join(CCS_WORK_ROOT, 'qfix/A/code')          # e0e2f07 deployed code (models byte-identical to archive code)
REC = _ccs_os.path.join(CCS_WORK_ROOT, 'qfix/C/rec')
ARCH = CCS_ARCHIVE_ROOT
CACHE_ROOT = ARCH + '/cache/v1'
BASE_RUNS = ARCH + '/runs/main'
PBE_ROOTS = [ARCH + '/runs/main/pbe', _ccs_os.path.join(CCS_WORK_ROOT, 'pbe_v1_fill/runs/pbe')]
BASELINES = ['maresunet', 'utanet', 'mk_unet', 'mhorunet', 'tinyunet', 'ucmnet', 'lbunet', 'cmunext', 'egeunet', 'malunet']
DS_ORDER = ['etis', 'PH2', 'cvc', 'kvasir', 'isic17', 'isic18']
OUT = BASE + '/out/items'
# 6 baseline runs with an old FAILED.txt: reference sha256 of the independent local copies
# (<workspace>/fieldswitch/ckpt/<p>/checkpoint.pth, <workspace>/data/main/<p>/results.json), hashed 2026-10-16 before any run
OLD_FAILED = {
    'egeunet/kvasir/k2': ('f78ac745d47d7d564efe24ba89a5fffee47784fc348e913971282e6c53f2fddd', '85bc646de3d82aad8a8c593e5d27825fa07e22c9f040c111a16c54b550186c00'),
    'egeunet/kvasir/k3': ('3a16a13915c5959681075ba61721b6908de7b3fe1918464a507213afbab3b186', '6c4c746f2cbd2c0f608b664e198ae450b46439ed47f1b5eef1861d36bb9f0af3'),
    'malunet/isic18/k4': ('934383518823103414a2d69da40bfea6f67a7ebdd87e89bee78cb4accede4565', '0fd20df30761d248ae4f650d9114f61369cbe4c3067a34c14791d6452f6dde9c'),
    'mhorunet/isic18/k3': ('c738c1f25cc6fb2e8d18932ac0805797bb38b34910398433309abceb9c00dbc9', '7a0044dab808b825a8777826be3e859124e97cfc134bcd25e8a18caf39232929'),
    'tinyunet/kvasir/k4': ('740e40a1a8a7251d0db06937b7ebac192dcdf304ed828bc2c0b5344e121c015d', 'a7c3ee55687eed6357c34a7f9fb1fb1c4412b38d0341584b3a010bccbe4344cc'),
    'utanet/isic17/k3': ('9fcbab83f846c4717a7c8b59037d1ffbb8f9d213e83cf5251cf58463650fb367', 'b26be32e30dc30e72a4e96a814762c03871fc9c8a075faeb5cb4402c1d89dc3f'),
}
N_AGREE = 8
N_EQ = 8                  # 8 test + 8 val images for the equivalence check
BATCH = 16
E1_TOL = 1e-4
GPU_CAP_S = 6.0 * 3600.0
SETTINGS = ['O_w4', 'O_a6', 'O_l50', 'D_w4', 'D_a6', 'D_l50', 'S_w4g', 'S_w4c', 'S_a6c', 'S_l50c']
# setting -> (graph, operator)
SPEC = {'O_w4': ('train', 'w4_pt'), 'O_a6': ('train', 'a6'), 'O_l50': ('train', 'l50'),
        'D_w4': ('dep', 'w4_pc'), 'D_a6': ('dep', 'a6'), 'D_l50': ('dep', 'l50'),
        'S_w4g': ('dep', 'w4_pt'), 'S_w4c': ('train', 'w4_pc'), 'S_a6c': ('dep', 'a6c'), 'S_l50c': ('dep', 'l50c')}
SEED = 7919


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


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


def jload(p):
    with open(p, encoding='utf8') as f:
        return json.load(f)


# ------------------------------------------------------------------ items
def list_items():
    items = []
    for ds in DS_ORDER:                                   # PBE v1 30: same rule as qfix_a.list_runs
        for root in PBE_ROOTS:
            for kd in sorted(glob.glob('%s/%s/k*' % (root, ds))):
                rj = kd + '/results.json'
                if not os.path.exists(rj) or os.path.exists(kd + '/FAILED.txt'):
                    continue
                r = jload(rj)
                if r.get('status') != 'complete':
                    continue
                items.append({'model': 'pbe', 'ds': ds, 'k': int(kd.rsplit('k', 1)[1]), 'dir': kd})
    assert len(items) == 30 and len(set((x['ds'], x['k']) for x in items)) == 30, len(items)
    for ds in DS_ORDER:
        for m in BASELINES:
            for k in range(1, 6):
                kd = '%s/%s/%s/k%d' % (BASE_RUNS, m, ds, k)
                r = jload(kd + '/results.json')
                c = r['config']
                assert r['status'] == 'complete' and c['model'] == m and c['dataset'] == ds and c['split'] == k, kd
                assert c['protocol']['protocol'] == 'v1', kd
                items.append({'model': m, 'ds': ds, 'k': k, 'dir': kd, 'old_failed_txt': os.path.exists(kd + '/FAILED.txt')})
    for it in items:
        it['tag'] = '%s__%s__k%d' % (it['model'], it['ds'], it['k'])
    assert len(items) == 330 and len(set(i['tag'] for i in items)) == 330
    return items


def main_prep():
    items = list_items()
    lines = []
    for it in items:
        for f in ('checkpoint.pth', 'results.json'):
            lines.append('%s  %s/%s' % (sha_file(it['dir'] + '/' + f), it['dir'], f))
    with open(BASE + '/MANIFEST_330.sha256', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    dump_json(BASE + '/ITEMS.json', items)
    print('PREP items=%d manifest_lines=%d old_failed=%d' % (len(items), len(lines), sum(1 for i in items if i.get('old_failed_txt'))))


def manifest():
    m = {}
    with open(BASE + '/MANIFEST_330.sha256') as f:
        for ln in f:
            h, p = ln.strip().split('  ', 1)
            m[p] = h
    return m


# ------------------------------------------------------------------ torch side
def setup_torch():
    import torch
    torch.set_num_threads(1)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, CODE_ROOT)
    sys.path.insert(0, BASE)
    sys.path.insert(0, REC)
    from models import registry
    import qfix_a
    import f2_fuse
    import f2b_fold_all
    torch.set_num_threads(1)                      # f2_* set 2 at import
    return torch, registry, qfix_a, f2_fuse, f2b_fold_all


def load_npz(torch, path, need_masks):
    import numpy as np
    with open(path, 'rb') as f:
        raw = f.read()
    z = np.load(io.BytesIO(raw))
    imgs = torch.from_numpy(z['images'])
    msks = torch.from_numpy(z['masks']) if need_masks else None
    return imgs, msks, hashlib.sha256(raw).hexdigest()


def load_state(torch, path):
    with open(path, 'rb') as f:
        raw = f.read()
    st = torch.load(io.BytesIO(raw), map_location='cpu')
    if isinstance(st, dict) and 'model' in st and isinstance(st['model'], dict) and len(st) < 5:
        st = st['model']
    return st, hashlib.sha256(raw).hexdigest()


def build(ad, state):
    m = ad.build(copy.deepcopy(ad.DEFAULT_CFG))
    m.load_state_dict(state, strict=True)
    return m.eval()


def fold_named(torch, f2b, ad, model, pairs, x1):
    """Generic BN folding of the pre-registered pairs (FOLD_PAIRS.json). Each pair must be a traced candidate."""
    m = copy.deepcopy(model).eval()
    cand, names = f2b.find_pairs(ad, m, x1)
    cn = set((names[l], names[b]) for l, b in cand)
    mods = dict(m.named_modules())
    n = 0
    for ln, bn in pairs:
        assert (ln, bn) in cn, ('pair not traced', ln, bn)
        assert f2b.fold_into(mods[ln], mods[bn]), ('fold refused', ln, bn)
        f2b.set_child(m, bn, torch.nn.Identity())
        n += 1
    return m.eval(), n


def norm_counts(torch, m):
    nn = torch.nn
    return {'bn': sum(isinstance(x, nn.modules.batchnorm._BatchNorm) for x in m.modules()),
            'gn': sum(isinstance(x, nn.GroupNorm) for x in m.modules()),
            'ln': sum(isinstance(x, nn.LayerNorm) for x in m.modules())}


def make_dep(torch, f2_fuse, f2b, ad, name, model, pairs, x1, dev):
    """Deployment graph from a training-graph model living on dev (dtype preserved)."""
    if name == 'pbe':
        cpu = copy.deepcopy(model).cpu()
        d = f2_fuse.fuse_pbe(cpu).to(dev).eval()
        return d, 'fuse_pbe'
    d, n = fold_named(torch, f2b, ad, model, pairs, x1)
    return d, n


class Graph(object):
    """One graph (training or deployment) with record-style apply/restore."""

    def __init__(self, torch, qa, ad, model, x1):
        self.torch, self.qa, self.ad, self.model = torch, qa, ad, model
        self.params = list(model.named_parameters())
        self.orig = {n: q.detach().clone() for n, q in self.params}
        self.buf_orig = {n: b.detach().clone() for n, b in model.named_buffers()}
        self.mods = dict(model.named_modules())
        leaf_cls = (torch.nn.modules.conv._ConvNd, torch.nn.Linear)
        self.leaf = [(n, m) for n, m in model.named_modules() if isinstance(m, leaf_cls)]
        self.targets = [(n, q) for n, q in self.params if q.is_floating_point() and q.dim() >= 2]
        self.axis = {}
        for n, q in self.targets:
            owner = self.mods.get(n.rsplit('.', 1)[0]) if '.' in n else None
            tr = isinstance(owner, torch.nn.modules.conv._ConvTransposeNd) and n.endswith('.weight')
            self.axis[n] = 1 if tr else 0
        called = set()
        hs = [m.register_forward_hook(lambda m_, i_, o_: called.add(id(m_))) for _, m in self.leaf]
        try:
            with torch.no_grad():
                ad.infer(model, x1)
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

    def exact(self):
        return all(self.torch.equal(q.data, self.orig[n]) for n, q in self.params)

    def apply(self, op):
        torch, qa = self.torch, self.qa
        hs, integ = [], {}
        with torch.no_grad():
            if op == 'w4_pt':
                for n, q in self.targets:
                    q.copy_(qa.rtn_sym_tensor(torch, self.orig[n], 4))
                integ = {'n_tensors': len(self.targets),
                         'max_levels': max([int(torch.unique(q.detach()).numel()) for _, q in self.targets] or [0])}
                assert integ['max_levels'] <= 15
            elif op == 'w4_pc':
                n1, mx, res = 0, 0, 0.0
                for n, q in self.targets:
                    w, lv = rtn_sym_channel(torch, self.orig[n], 4, self.axis[n])
                    q.copy_(w)
                    n1 += int(self.orig[n].shape[self.axis[n]] == 1)
                    mx = max(mx, int(lv.abs().max()))
                integ = {'n_tensors': len(self.targets), 'n_tensors_axis_size1': n1, 'max_abs_level': mx,
                         'n_axis1_tensors': sum(1 for n, _ in self.targets if self.axis[n] == 1)}
                assert mx <= 7
            elif op in ('l50', 'l50c'):
                layers = [(n, m) for n, m in self.leaf if id(m) in self.executed
                          and getattr(m, 'weight', None) is not None and m.weight.dim() >= 2]
                zs = []
                for n, m in layers:
                    if op == 'l50':
                        m.weight.copy_(qa.prune_layer(torch, self.orig[n + '.weight'], 0.5))
                    else:
                        ax = 1 if isinstance(m, torch.nn.modules.conv._ConvTransposeNd) else 0
                        m.weight.copy_(prune_channel(torch, self.orig[n + '.weight'], 0.5, ax))
                    zs.append(float((m.weight == 0).double().mean()))
                integ = {'n_layers': len(layers), 'min_zero_frac': min(zs) if zs else None}
            elif op in ('a6', 'a6c'):
                for n, m in self.leaf:
                    hk = qa.act_hook_factory(torch, 6) if op == 'a6' else act_hook_channel(torch, 6, isinstance(m, torch.nn.Linear))
                    hs.append(m.register_forward_hook(hk))
                integ = {'n_hooked': len(self.leaf), 'n_hooked_executed': sum(1 for _, m in self.leaf if id(m) in self.executed)}
            else:
                raise ValueError(op)
        return hs, integ


def rtn_sym_channel(torch, w, bits, axis):
    """Per-output-channel symmetric RTN (same rounding / clamp / zero guard as rtn_sym_tensor). Returns (dequant, levels)."""
    qm = float(2 ** (bits - 1) - 1)
    wt = w.detach().transpose(0, axis) if axis != 0 else w.detach()
    amax = wt.abs().reshape(wt.shape[0], -1).amax(1)
    scale = torch.where(torch.isfinite(amax) & (amax > 0), amax / qm, torch.ones_like(amax))
    sh = [-1] + [1] * (wt.dim() - 1)
    lv = torch.clamp(torch.round(wt / scale.view(sh)), -qm, qm)
    q = lv * scale.view(sh)
    if axis != 0:
        q = q.transpose(0, axis); lv = lv.transpose(0, axis)
    return q.contiguous(), lv


def prune_channel(torch, w, s, axis):
    """Per-output-channel magnitude pruning: per channel k_c = floor(s n_c), zero |w| <= k_c-th smallest (record tie rule)."""
    wt = w.detach().transpose(0, axis) if axis != 0 else w.detach()
    a = wt.abs().reshape(wt.shape[0], -1)
    k = int(s * a.shape[1])
    if k < 1:
        return w.clone()
    thr = torch.kthvalue(a.cpu(), k, dim=1).values.to(w.device)
    keep = (a > thr.view(-1, 1)).reshape(wt.shape)
    if axis != 0:
        keep = keep.transpose(0, axis)
    return torch.where(keep, w, torch.zeros_like(w))


def act_hook_channel(torch, bits, is_linear):
    """Dynamic per-(sample, output channel) symmetric fake-quant: channel = dim 1 (conv) or last dim (Linear)."""
    qm = float(2 ** (bits - 1) - 1)

    def hook(mod_, inp, out):
        if not torch.is_tensor(out):
            return None
        y = out
        if y.dim() <= 2:
            amax = y.detach().abs()
        elif is_linear:
            amax = y.detach().abs().flatten(1, -2).amax(1, keepdim=True)
            amax = amax.view(y.shape[0], *([1] * (y.dim() - 2)), y.shape[-1])
        else:
            amax = y.detach().abs().flatten(2).amax(2).view(y.shape[0], y.shape[1], *([1] * (y.dim() - 2)))
        sc = torch.where(torch.isfinite(amax) & (amax > 0), amax / qm, torch.ones_like(amax))
        return torch.clamp(torch.round(y / sc), -qm, qm) * sc
    return hook


class Item(object):
    def __init__(self, torch, registry, qa, f2_fuse, f2b, it, pairs, dev, batch=BATCH):
        self.torch, self.dev, self.it, self.batch = torch, dev, it, batch
        self.qa, self.f2_fuse, self.f2b = qa, f2_fuse, f2b
        self.ad = registry.get(it['model'])
        r = jload(it['dir'] + '/results.json')
        assert r['config']['model_cfg'] == self.ad.DEFAULT_CFG, r['config']['model_cfg']
        self.results = r
        self.results_sha = sha_file(it['dir'] + '/results.json')
        self.state, self.ckpt_sha = load_state(torch, it['dir'] + '/checkpoint.pth')
        dc = r['data_cache']
        vi, vm, self.val_sha = load_npz(torch, '%s/%s/%s' % (CACHE_ROOT, it['ds'], dc['val']), True)
        ti, _, self.test_sha = load_npz(torch, '%s/%s/%s' % (CACHE_ROOT, it['ds'], dc['test']), False)
        self.vimgs = (vi.float() / 255.0).to(dev)
        self.vmsks = vm.float().to(dev)
        self.timgs = (ti[:N_AGREE].float() / 255.0).to(dev)
        assert tuple(self.vimgs.shape[1:]) == (3, 256, 256) and tuple(self.vmsks.shape[1:]) == (1, 256, 256)
        assert self.timgs.shape[0] == N_AGREE
        self.eq = torch.cat([self.timgs[:N_EQ], self.vimgs[:N_EQ]], 0)
        self.pairs = pairs
        m = build(self.ad, self.state).to(dev)
        x1 = self.vimgs[0:1]
        self.ncount_train = norm_counts(torch, m)
        dep, self.n_folded = make_dep(torch, f2_fuse, f2b, self.ad, it['model'], m, pairs, x1, dev)
        self.ncount_dep = norm_counts(torch, dep)
        self.params_train = sum(p.numel() for n, p in m.named_parameters() if not self.ad.is_train_only(n))
        self.params_dep = sum(p.numel() for n, p in dep.named_parameters() if not self.ad.is_train_only(n))
        self.g = {'train': Graph(torch, qa, self.ad, m, x1), 'dep': Graph(torch, qa, self.ad, dep, x1)}

    def logits(self, model, imgs, bs):
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, imgs.shape[0], bs):
                out.append(self.ad.infer(model, imgs[i:i + bs]).float())
        return torch.cat(out, 0)

    def val_metrics(self, model):
        torch = self.torch
        dice, iou, fg = [], [], []
        with torch.no_grad():
            for i in range(0, self.vimgs.shape[0], self.batch):
                p = (torch.sigmoid(self.ad.infer(model, self.vimgs[i:i + self.batch]).float()) > 0.5).double()
                t = self.vmsks[i:i + self.batch].double()
                tp = (p * t).flatten(1).sum(1); fp = (p * (1 - t)).flatten(1).sum(1); fn = ((1 - p) * t).flatten(1).sum(1)
                u = tp + fp + fn
                iou.append(torch.where(u > 0, tp / u.clamp(min=1.0), torch.ones_like(u)))
                d = 2 * tp + fp + fn
                dice.append(torch.where(d > 0, 2 * tp / d.clamp(min=1.0), torch.ones_like(d)))
                fg.append(p.flatten(1).mean(1))
        return float(torch.cat(dice).mean()), float(torch.cat(iou).mean()), float(torch.cat(fg).mean())

    @staticmethod
    def pair_dice(torch, a, b):
        a = a.double().flatten(1); b = b.double().flatten(1)
        s = a.sum(1) + b.sum(1)
        d = torch.where(s > 0, 2 * (a * b).sum(1) / s.clamp(min=1.0), torch.ones_like(s))
        return [float(v) for v in d]

    def test_stats(self, lg, ref_lg):
        torch = self.torch
        fin = torch.isfinite(lg).flatten(1).all(1)
        mq = (torch.sigmoid(lg) > 0.5) & fin.view(-1, 1, 1, 1)
        mr = torch.sigmoid(ref_lg) > 0.5
        ag = self.pair_dice(torch, mr, mq)
        if not bool(torch.isfinite(lg).all()):
            sq = -100.0
        else:
            num = float((ref_lg.double() ** 2).sum()); den = float(((ref_lg.double() - lg.double()) ** 2).sum())
            sq = 100.0 if den == 0 else (10.0 * math.log10(num / den) if num > 0 else -100.0)
        return {'agree': sum(ag) / len(ag), 'agree_per_img': ag, 'fg_q': float(mq.double().mean()),
                'sqnr_db': sq, 'n_nonfinite_img': int((~fin).sum())}

    def equivalence(self):
        torch = self.torch
        a = self.logits(self.g['train'].model, self.eq, 4); b = self.logits(self.g['dep'].model, self.eq, 4)
        d32 = float((a - b).abs().max()); sc = float(a.abs().max())
        e2 = {'max_abs_fp32': d32, 'logit_absmax': sc, 'rel_fp32': d32 / max(sc, 1e-12),
              'mask_mismatch_frac': float(((a > 0) != (b > 0)).double().mean())}
        err = None
        try:
            t64 = copy.deepcopy(self.g['train'].model).double().eval()
            x1 = self.vimgs[0:1].double()
            d64m, _ = make_dep(torch, self.f2_fuse, self.f2b, self.ad, self.it['model'], t64, self.pairs, x1, self.dev)
            d64 = 0.0
            with torch.no_grad():
                for i in range(0, self.eq.shape[0], 2):
                    x = self.eq[i:i + 2].double()
                    d64 = max(d64, float((self.ad.infer(t64, x) - self.ad.infer(d64m, x)).abs().max()))
            del t64, d64m
        except Exception as ex:
            if 'out of memory' in str(ex).lower():
                raise
            d64, err = None, traceback.format_exc().strip().splitlines()[-1]
        torch.cuda.empty_cache()
        return {'E1_max_abs_fp64': d64, 'E1_pass': bool(d64 is not None and d64 <= E1_TOL), 'E1_error': err, 'E2': e2}

    def run(self):
        torch = self.torch
        rec = {'fp32': {}, 'cells': {}}
        ref = {}
        for gname, g in self.g.items():
            g.restore()
            d0, i0, f0 = self.val_metrics(g.model)
            ref[gname] = self.logits(g.model, self.timgs, self.batch)
            rec['fp32'][gname] = {'val_dice': d0, 'val_iou': i0, 'val_pred_fg': f0,
                                  'test8_fg': float((torch.sigmoid(ref[gname]) > 0.5).double().mean()),
                                  'test8_fg_per_img': [float(v) for v in (torch.sigmoid(ref[gname]) > 0.5).double().flatten(1).mean(1)]}
        rec['equiv'] = self.equivalence()
        for s in SETTINGS:
            gname, op = SPEC[s]
            g = self.g[gname]
            hs, integ = g.apply(op)
            try:
                d, i, f = self.val_metrics(g.model)
                lg = self.logits(g.model, self.timgs, self.batch)
            finally:
                for h in hs:
                    h.remove()
                g.restore()
            c = {'val_dice': d, 'val_iou': i, 'val_pred_fg': f, 'integrity': integ}
            c.update(self.test_stats(lg, ref[gname]))
            rec['cells'][s] = c
        rec['state_restored_exact'] = {k: bool(g.exact()) for k, g in self.g.items()}
        rec.update({'model': self.it['model'], 'ds': self.it['ds'], 'k': self.it['k'], 'dir': self.it['dir'], 'tag': self.it['tag'],
                    'ckpt_sha256': self.ckpt_sha, 'results_sha256': self.results_sha, 'val_cache_sha256': self.val_sha,
                    'test_cache_sha256': self.test_sha, 'best_val_iou_stored': self.results.get('best_val_iou'),
                    'n_val': int(self.vimgs.shape[0]), 'norm_train': self.ncount_train, 'norm_dep': self.ncount_dep,
                    'n_folded': self.n_folded, 'params_inf_train': self.params_train, 'params_inf_dep': self.params_dep,
                    'old_failed_txt': bool(self.it.get('old_failed_txt')), 'commit': self.results['code']['commit'],
                    'torch': torch.__version__, 'batch': self.batch})
        return rec


# ------------------------------------------------------------------ fold pairs
def randomize_bn(torch, m, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm) and mod.running_mean is not None:
                C = mod.running_mean.numel()
                dt, dv = mod.running_mean.dtype, mod.running_mean.device
                mod.running_mean.copy_((torch.randn(C, generator=g, dtype=torch.float64) * 0.1).to(dv, dt))
                mod.running_var.copy_((torch.rand(C, generator=g, dtype=torch.float64) * 1.5 + 0.5).to(dv, dt))
                if mod.weight is not None:
                    mod.weight.copy_((torch.rand(C, generator=g, dtype=torch.float64) * 0.4 + 0.8).to(dv, dt))
                if mod.bias is not None:
                    mod.bias.copy_((torch.randn(C, generator=g, dtype=torch.float64) * 0.1).to(dv, dt))


def main_foldpairs():
    torch, registry, qa, f2_fuse, f2b = setup_torch()
    dev = torch.device('cuda:0')
    items = {i['tag']: i for i in jload(BASE + '/ITEMS.json')}
    out = {}
    for name in BASELINES:
        t0 = time.time()
        it = items['%s__etis__k1' % name]
        ad = registry.get(name)
        st, _ = load_state(torch, it['dir'] + '/checkpoint.pth')
        m = build(ad, st).double().to(dev)
        randomize_bn(torch, m, SEED)
        r = jload(it['dir'] + '/results.json')
        vi, _, _ = load_npz(torch, '%s/etis/%s' % (CACHE_ROOT, r['data_cache']['val']), False)
        x = (vi[:2].double() / 255.0).to(dev)
        cand, names = f2b.find_pairs(ad, m, x[:1])
        with torch.no_grad():
            ref = ad.infer(m, x)
        sc = float(ref.abs().max())
        acc, rej = [], []
        for l, b in cand:
            mm = copy.deepcopy(m)
            mods = dict(mm.named_modules())
            ok = f2b.fold_into(mods[names[l]], mods[names[b]])
            if ok:
                f2b.set_child(mm, names[b], torch.nn.Identity())
                with torch.no_grad():
                    d = float((ad.infer(mm, x) - ref).abs().max())
                ok = d <= 1e-9 * sc
            else:
                d = None
            (acc if ok else rej).append([names[l], names[b], d])
            del mm
        nb = norm_counts(torch, m)
        out[name] = {'accepted': [[a, b] for a, b, _ in acc], 'accepted_maxdiff': max([d for _, _, d in acc] or [0.0]),
                     'rejected': rej, 'n_candidates': len(cand), 'norm_counts': nb, 'logit_absmax_randomised': sc,
                     'ckpt': it['dir'], 'wall_s': time.time() - t0}
        print('FOLDPAIRS %s cand=%d accepted=%d rejected=%d bn=%d gn=%d ln=%d %.1fs' % (
            name, len(cand), len(acc), len(rej), nb['bn'], nb['gn'], nb['ln'], time.time() - t0), flush=True)
        del m
        torch.cuda.empty_cache()
    dump_json(BASE + '/FOLD_PAIRS.json', out)


def load_pairs():
    fp = jload(BASE + '/FOLD_PAIRS.json')
    return {k: [tuple(p) for p in v['accepted']] for k, v in fp.items()}


# ------------------------------------------------------------------ synthetic (section 7.1, 7.2, 7.4)
def main_synthetic():
    torch, registry, qa, f2_fuse, f2b = setup_torch()
    dev = torch.device('cuda:0')
    nn = torch.nn
    res, lines = {}, []

    def rec(key, ok, msg):
        res[key] = {'PASS': bool(ok), 'msg': msg}
        lines.append('SYNTHETIC %s %s: %s' % (key, 'PASS' if ok else 'FAIL', msg))
        print(lines[-1], flush=True)
    torch.manual_seed(SEED)
    # 7.1 synthetic Conv+BN fold (fp64, random stats)
    conv = nn.Conv2d(8, 16, 3, padding=1, bias=True).double().to(dev); bn = nn.BatchNorm2d(16).double().to(dev).eval()
    randomize_bn(torch, nn.Sequential(bn), SEED)
    x = torch.randn(2, 8, 32, 32, dtype=torch.float64, device=dev)
    with torch.no_grad():
        y0 = bn(conv(x)); c2 = copy.deepcopy(conv); f2b.fold_into(c2, bn); e = float((c2(x) - y0).abs().max())
    rec('7.1a_convbn_fold_fp64', e < 1e-12, 'max|d|=%.2e' % e)
    # 7.1 PBE fusion: correct / missing bn_identity / var^2 fold  (PBE etis k1 copy, fp64)
    items = {i['tag']: i for i in jload(BASE + '/ITEMS.json')}
    it = items['pbe__etis__k1']
    ad = registry.get('pbe')
    st, _ = load_state(torch, it['dir'] + '/checkpoint.pth')
    r = jload(it['dir'] + '/results.json')
    vi, _, _ = load_npz(torch, '%s/etis/%s' % (CACHE_ROOT, r['data_cache']['val']), False)
    x = (vi[:4].double() / 255.0).to(dev)
    m64 = build(ad, st).double().to(dev)
    with torch.no_grad():
        ref = ad.infer(m64, x)

    def e1(dm):
        with torch.no_grad():
            return float((ad.infer(dm.to(dev), x) - ref).abs().max())
    ok_d = e1(f2_fuse.fuse_pbe(copy.deepcopy(m64).cpu()))
    rec('7.1b_pbe_fusion_correct', ok_d < 1e-10, 'E1 fp64 max|d|=%.2e (<1e-10, gate %.0e)' % (ok_d, E1_TOL))
    orig_fold, orig_rep = f2_fuse.fold, f2_fuse.fuse_rep

    def rep_noid(rc):                                  # deliberately wrong: drop the bn_identity branch
        k = rc.conv_kxk.kernel_size[0]; C = rc.conv_kxk.out_channels
        w3, b3 = f2_fuse.fold(rc.conv_kxk, rc.bn_kxk); w1, b1 = f2_fuse.fold(rc.conv_1x1, rc.bn_1x1)
        w1 = nn.functional.pad(w1, [k // 2] * 4)
        c = nn.Conv2d(C, C, k, 1, k // 2, groups=C, bias=True, dtype=rc.conv_kxk.weight.dtype)
        c.weight.data.copy_(w3 + w1); c.bias.data.copy_(b3 + b1)
        return nn.Sequential(c, nn.ReLU(inplace=True))

    def fold_var2(conv_, bn_):                         # deliberately wrong: running_var^2
        std = (bn_.running_var ** 2 + bn_.eps).sqrt(); t = bn_.weight / std
        w = conv_.weight * t.reshape(-1, 1, 1, 1); b = bn_.bias - bn_.running_mean * t
        if conv_.bias is not None:
            b = b + conv_.bias * t
        return w.detach(), b.detach()
    try:
        f2_fuse.fuse_rep = rep_noid
        d_noid = e1(f2_fuse.fuse_pbe(copy.deepcopy(m64).cpu()))
    finally:
        f2_fuse.fuse_rep = orig_rep
    rec('7.1c_pbe_missing_identity', d_noid > E1_TOL, 'E1 fp64 max|d|=%.2e (must FAIL gate)' % d_noid)
    try:
        f2_fuse.fold = fold_var2
        d_v2 = e1(f2_fuse.fuse_pbe(copy.deepcopy(m64).cpu()))
    finally:
        f2_fuse.fold = orig_fold
    rec('7.1d_pbe_var2_fold', d_v2 > E1_TOL, 'E1 fp64 max|d|=%.2e (must FAIL gate)' % d_v2)
    # generic fold with var^2 on a baseline (tinyunet etis k1)
    it2 = items['tinyunet__etis__k1']; ad2 = registry.get('tinyunet')
    st2, _ = load_state(torch, it2['dir'] + '/checkpoint.pth')
    b64 = build(ad2, st2).double().to(dev)
    with torch.no_grad():
        ref2 = ad2.infer(b64, x)
    pairs = load_pairs()['tinyunet']
    dgood, _ = fold_named(torch, f2b, ad2, b64, pairs, x[:1])
    orig_fi = f2b.fold_into

    def fi_var2(l, b):
        with torch.no_grad():
            b.running_var.copy_(b.running_var ** 2)
        return orig_fi(l, b)
    try:
        f2b.fold_into = fi_var2
        dbad, _ = fold_named(torch, f2b, ad2, b64, pairs, x[:1])
    finally:
        f2b.fold_into = orig_fi
    with torch.no_grad():
        eg = float((ad2.infer(dgood, x) - ref2).abs().max()); eb = float((ad2.infer(dbad, x) - ref2).abs().max())
    rec('7.1e_generic_fold_tinyunet', eg < 1e-10 and eb > E1_TOL, 'correct %.2e (<1e-10), var^2 %.2e (must FAIL gate)' % (eg, eb))
    del m64, b64, dgood, dbad
    # 7.2 quantisers
    w = torch.randn(8, 16, 3, 3, device=dev); w[0] *= 1000.0
    qt = qa.rtn_sym_tensor(torch, w, 4)
    zf = float((qt[1:] == 0).double().mean())
    qc, lv = rtn_sym_channel(torch, w, 4, 0)
    rel = max(float((qc[c] - w[c]).abs().max() / w[c].abs().max()) for c in range(8))
    nlev = max(int(torch.unique(lv[c]).numel()) for c in range(8))
    rec('7.2a_w4_outlier', zf >= 0.9 and rel <= 1 / 14 + 1e-6 and nlev <= 15,
        'per-tensor zero frac ch1-7=%.3f (>=0.9); per-channel max rel err=%.4f (<=1/14); levels<=%d' % (zf, rel, nlev))
    wt = torch.randn(16, 8, 3, 3, device=dev); wt[:, 0] *= 1000.0             # ConvTranspose layout (in, out, k, k)
    qct, lvt = rtn_sym_channel(torch, wt, 4, 1)
    relt = max(float((qct[:, c] - wt[:, c]).abs().max() / wt[:, c].abs().max()) for c in range(8))
    nlevt = max(int(torch.unique(lvt[:, c]).numel()) for c in range(8))
    rec('7.2b_w4_convtranspose_axis1', relt <= 1 / 14 + 1e-6 and nlevt <= 15, 'max rel err=%.4f, levels<=%d' % (relt, nlevt))
    y = torch.randn(3, 8, 16, 16, device=dev); y[:, 0] *= 100.0
    ya = qa.act_hook_factory(torch, 6)(None, None, y); yc = act_hook_channel(torch, 6, False)(None, None, y)
    nla = max(int(torch.unique(ya[s]).numel()) for s in range(3)); nlc = max(int(torch.unique(yc[s, c]).numel()) for s in range(3) for c in range(8))
    ec = max(float((yc[s, c] - y[s, c]).abs().max() / y[s, c].abs().max()) for s in range(3) for c in range(8))
    yl = torch.randn(2, 10, 8, device=dev); yl[..., 0] *= 100.0
    ylc = act_hook_channel(torch, 6, True)(None, None, yl)
    el = max(float((ylc[s, :, c] - yl[s, :, c]).abs().max() / yl[s, :, c].abs().max()) for s in range(2) for c in range(8))
    rec('7.2c_a6_pt_pc', nla <= 63 and nlc <= 63 and ec <= 1 / 62 + 1e-6 and el <= 1 / 62 + 1e-6,
        'levels pt<=%d pc<=%d; pc max rel err conv=%.4f linear=%.4f (<=1/62)' % (nla, nlc, ec, el))
    wp = torch.randn(8, 16, 3, 3, device=dev)
    zl = float((qa.prune_layer(torch, wp, 0.5) == 0).double().mean())
    pc = prune_channel(torch, wp, 0.5, 0)
    zc = [float((pc[c] == 0).double().mean()) for c in range(8)]
    pct = prune_channel(torch, wt, 0.5, 1)
    zct = [float((pct[:, c] == 0).double().mean()) for c in range(8)]
    n = wp.numel(); nc = wp[0].numel()
    rec('7.2d_l50_layer_channel', abs(zl - int(0.5 * n) / n) < 1e-9 and all(abs(z - int(0.5 * nc) / nc) < 1e-9 for z in zc + zct),
        'layer zero=%.4f (def %.4f); channel zero min/max=%.4f/%.4f (def %.4f)' % (zl, int(0.5 * n) / n, min(zc + zct), max(zc + zct), int(0.5 * nc) / nc))
    # 7.4 planted (function-preserving) outlier on PBE kvasir k1 copy
    it = items['pbe__kvasir__k1']
    pairs_all = load_pairs()
    clean = Item(torch, registry, qa, f2_fuse, f2b, it, [], dev)
    rc = clean.run()
    pl = Item(torch, registry, qa, f2_fuse, f2b, it, [], dev)
    mt = pl.g['train'].model
    with torch.no_grad():
        mods = dict(mt.named_modules())
        mods['base.encoder3.0.1.conv_kxk'].weight[0].mul_(1000.0)
        mods['base.encoder3.0.1.bn_kxk'].running_mean[0].mul_(1000.0)
        mods['base.encoder3.0.1.bn_kxk'].running_var[0].mul_(1000.0 ** 2)
    x1 = pl.vimgs[0:1]
    dep, _ = make_dep(torch, f2_fuse, f2b, pl.ad, 'pbe', mt, [], x1, dev)
    pl.g = {'train': Graph(torch, qa, pl.ad, mt, x1), 'dep': Graph(torch, qa, pl.ad, dep, x1)}
    rp = pl.run()
    wq = qa.rtn_sym_tensor(torch, dict(mt.named_parameters())['base.encoder3.0.1.conv_kxk.weight'].detach(), 4)
    zf = float((wq[1:] == 0).double().mean())

    def R(r, s, gname):
        return r['cells'][s]['val_dice'] / r['fp32'][gname]['val_dice']
    dfp = abs(rc['fp32']['train']['val_dice'] - rp['fp32']['train']['val_dice'])
    dD = abs(R(rc, 'D_w4', 'dep') - R(rp, 'D_w4', 'dep')); dS = abs(R(rc, 'S_w4c', 'train') - R(rp, 'S_w4c', 'train'))
    ok = dfp < 1e-3 and rp['equiv']['E1_pass'] and dD <= 0.02 and dS <= 0.02 and zf >= 0.9
    rec('7.4_planted_pbe_kvasir_k1', ok, '|dDice fp32|=%.1e, E1=%.1e, |dR D_w4|=%.4f, |dR S_w4c|=%.4f (<=0.02), per-tensor zeroes %.3f of planted tensor ch1.. (>=0.9)' % (
        dfp, rp['equiv']['E1_max_abs_fp64'], dD, dS, zf))
    res['ALL_PASS'] = all(v['PASS'] for k, v in res.items() if isinstance(v, dict))
    dump_json(BASE + '/synthetic.json', res)
    with open(BASE + '/synthetic.log', 'a') as f:
        f.write('\n'.join(lines) + '\nSYNTHETIC ALL %s %s\n' % ('PASS' if res['ALL_PASS'] else 'FAIL', time.strftime('%F %T %Z')))
    print('SYNTHETIC ALL', 'PASS' if res['ALL_PASS'] else 'FAIL', flush=True)


# ------------------------------------------------------------------ run
def gpu_uuid_of_visible():
    phys = os.environ.get('QD_PHYS_GPU', '0')
    o = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], capture_output=True, text=True).stdout
    for ln in o.strip().splitlines():
        i, u = [s.strip() for s in ln.split(',')]
        if i == phys:
            return u
    raise RuntimeError('gpu not found')


def foreign_pids(uuid):
    o = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], capture_output=True, text=True).stdout
    me = os.getpid()
    out = []
    for ln in o.strip().splitlines():
        if not ln.strip():
            continue
        u, p = [s.strip() for s in ln.split(',')]
        if u == uuid and p.isdigit() and int(p) != me:
            try:
                with open('/proc/%s/cmdline' % p, 'rb') as f:
                    if b'qdeploy.py' in f.read() and int(p) == me:
                        continue
            except Exception:
                pass
            out.append(int(p))
    return out


def stop(reason, msg):
    with open(BASE + '/out/STOPPED_' + reason, 'w') as f:
        f.write('%s %s\n' % (time.strftime('%F %T %Z'), msg))
    print('STOP', reason, msg, flush=True)
    sys.exit(5)


def main_run(only):
    torch, registry, qa, f2_fuse, f2b = setup_torch()
    dev = torch.device('cuda:0')
    os.makedirs(OUT, exist_ok=True)
    items = jload(BASE + '/ITEMS.json')
    man = manifest()
    pairs = load_pairs()
    uuid = gpu_uuid_of_visible()
    if only:
        items = [i for i in items if i['tag'] in only]
    used = 0.0
    for p in glob.glob(OUT + '/*.DONE'):
        try:
            used += float(open(p).read().split()[0])
        except Exception:
            pass
    oom = len(glob.glob(OUT + '/*.OOM'))
    t_all = time.time()
    for it in items:
        tag = it['tag']
        if os.path.exists('%s/%s.DONE' % (OUT, tag)) or os.path.exists('%s/%s.FAIL' % (OUT, tag)):
            continue
        while True:
            fp = foreign_pids(uuid)
            if not fp:
                break
            print('[%s] WAIT foreign pid(s) on GPU: %s' % (time.strftime('%T'), fp), flush=True)
            time.sleep(60)
        if used >= GPU_CAP_S:
            stop('CAP', 'compute %.0fs >= cap %.0fs before %s' % (used, GPU_CAP_S, tag))
        bs = 4 if os.path.exists('%s/%s.OOM' % (OUT, tag)) else BATCH
        t0 = time.time()
        try:
            for f in ('checkpoint.pth', 'results.json'):
                p = it['dir'] + '/' + f
                if man.get(p) != sha_file(p):
                    if it['model'] == 'pbe':
                        stop('PBE_SHA', p)
                    raise RuntimeError('MANIFEST_MISMATCH ' + p)
            eng = Item(torch, registry, qa, f2_fuse, f2b, it, pairs.get(it['model'], []), dev, batch=bs)
            rec = eng.run()
            rec['gpu_wall_s'] = time.time() - t0
            dump_json('%s/%s.json' % (OUT, tag), rec)
            with open('%s/%s.DONE' % (OUT, tag), 'w') as f:
                f.write('%.1f\n' % rec['gpu_wall_s'])
            used += rec['gpu_wall_s']
            ints = {s: rec['cells'][s]['integrity'] for s in ('D_w4', 'D_l50')}
            print('[%s] %s %.1fs bs=%d E1=%s(%.1e) folded=%s bn %d->%d restored=%s D_w4 n=%d D_l50 minzero=%.3f used=%.0fs' % (
                time.strftime('%T'), tag, rec['gpu_wall_s'], bs, rec['equiv']['E1_pass'], rec['equiv']['E1_max_abs_fp64'] if rec['equiv']['E1_max_abs_fp64'] is not None else -1.0, rec['n_folded'],
                rec['norm_train']['bn'], rec['norm_dep']['bn'], all(rec['state_restored_exact'].values()),
                ints['D_w4']['n_tensors'], ints['D_l50']['min_zero_frac'] or -1, used), flush=True)
            if it['model'] == 'pbe' and not rec['equiv']['E1_pass']:
                stop('PBE_E1', '%s E1 %s' % (tag, rec['equiv']['E1_max_abs_fp64']))
            del eng
        except SystemExit:
            raise
        except Exception as ex:
            tb = traceback.format_exc()
            used += time.time() - t0
            is_oom = 'out of memory' in str(ex).lower()
            try:
                del eng
            except Exception:
                pass
            torch.cuda.empty_cache()
            if is_oom:
                oom += 1
                with open('%s/%s.OOM' % (OUT, tag), 'w') as f:
                    f.write(tb)
                print('[%s] %s OOM (#%d)' % (time.strftime('%T'), tag, oom), flush=True)
                if oom >= 2:
                    stop('OOM', 'second OOM at %s' % tag)
                # retry once at batch 4 (restart-safe: .OOM marker selects batch 4)
                items.insert(items.index(it) + 1, it)
                continue
            with open('%s/%s.FAIL' % (OUT, tag), 'w') as f:
                f.write(tb)
            print('[%s] %s FAIL %s' % (time.strftime('%T'), tag, tb.strip().splitlines()[-1]), flush=True)
        torch.cuda.empty_cache()
    print('RUN_END wall=%.1fs used=%.1fs' % (time.time() - t_all, used), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--prep', action='store_true')
    ap.add_argument('--foldpairs', action='store_true')
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    if a.prep:
        main_prep()
    if a.foldpairs:
        main_foldpairs()
    if a.synthetic:
        main_synthetic()
    if a.run:
        main_run(set(x for x in a.only.split(',') if x))
