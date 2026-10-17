# -*- coding: utf-8 -*-
"""PREREG_SGR engine.
  --unit                 : CPU unit checks (PREREG_SGR §9.2) on synthetic tensors / a toy net -> unit.json
  --smoke TAG            : CPU, one checkpoint, every block, 2 steps per tuned arm, NO Dice / agreement (finite forward on
                           2 train images only) -> smoke/<tag>__<block>.json; prints integrity counters only
  --run --worker gpu0|gpu1 : GPU queue worker (CUDA_VISIBLE_DEVICES must expose exactly one GPU; SGR_PHYS_GPU = physical
                           index).  Queue = ITEMS_SGR.json blocks in order main -> sub -> w4 -> naux; unit = (tag, block);
                           out/items/<tag>__<block>.json + .DONE / .FAIL / .OOM / .CLAIM; restart-safe; stop rules -> out/STOPPED_*.
                           gpu1 waits for <WORK>/abl/ALL_DONE before claiming anything.
Values are sealed: logs show wall time and integrity counters only.
Data: repair sees TRAIN images only (B4 also TRAIN masks); val = retention only; test = first 8 images (images key) for the
F3 agreement re-screen only."""
import os as _ccs_os  # path configuration of the released copy (README, Configuration)
CCS_REPO_ROOT = _ccs_os.path.normpath(_ccs_os.path.join(_ccs_os.path.dirname(_ccs_os.path.abspath(__file__)), '../..'))
CCS_WORK_ROOT = _ccs_os.environ.get('CCS_WORK', _ccs_os.path.join(CCS_REPO_ROOT, 'experiments'))
import os
import sys
import io
import json
import glob
import time
import copy
import math
import socket
import hashlib
import argparse
import traceback
import subprocess

sys.dont_write_bytecode = True

BASE = _ccs_os.path.join(CCS_WORK_ROOT, 'sgr')
QD = _ccs_os.path.join(CCS_WORK_ROOT, 'qdeploy')
REC = _ccs_os.path.join(CCS_WORK_ROOT, 'qfix/C/rec')
CODE_ROOT = _ccs_os.path.join(CCS_WORK_ROOT, 'qfix/A/code')
TAU_JSON = _ccs_os.path.join(CCS_WORK_ROOT, 'qdeploy_tau/TAU_RESULT.json')
ABL_DONE = _ccs_os.path.join(CCS_WORK_ROOT, 'abl/ALL_DONE')
OUT = BASE + '/out/items'
OUTD = BASE + '/out'
CAP_S = 20.0 * 3600.0
REPRO_TOL = 1e-6
MAX_CONSEC_FAIL = 3
SEED0 = 20261017
N_MAIN, N_DIAG = 100, 20
N_AUX = (10, 30)
BLOCK_ORDER = ['main', 'sub', 'w4', 'naux']
ARMS = {'main': ['B0', 'B1', 'B2', 'B3', 'SGR'], 'sub': ['B4', 'A1', 'A2', 'A3'],
        'w4': ['B0w', 'B1w', 'B3w', 'B5', 'SGRw'], 'naux': ['SGR_n10', 'SGR_n30', 'B1_n10', 'B1_n30']}
for p in (REC, QD, CODE_ROOT, BASE):
    if p not in sys.path:
        sys.path.insert(0, p)
import qdeploy as Q
import sgr_lib as S


def jload(p):
    with open(p, encoding='utf8') as f:
        return json.load(f)


def seed_ck(tag):
    return (int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16) ^ SEED0) % (2 ** 31)


def img_hashes(np, imgs):
    return [hashlib.sha256(np.ascontiguousarray(imgs[i]).tobytes()).hexdigest() for i in range(imgs.shape[0])]


def train_order(np, seed, tr_names, tr_hash, vt_names, vt_hash):
    """selection order of TRAIN images (PREREG_SGR §3): the pool excludes every train image whose file name or image-byte
    sha256 also occurs in the val or test split of the same run (content duplicates exist in etis / isic17); the pool is
    permuted with RandomState(seed); n images = first n of this order (diagnosis = first min(n, 20))."""
    pool = [i for i in range(len(tr_names)) if tr_names[i] not in vt_names and tr_hash[i] not in vt_hash]
    perm = np.random.RandomState(seed).permutation(len(pool))
    return np.array([pool[j] for j in perm], dtype=np.int64), len(tr_names) - len(pool)


def tau_score():
    return float(jload(TAU_JSON)['L80']['tau_score'])


def setup(cpu):
    import torch
    if not cpu:
        return Q.setup_torch()                    # asserts exactly one GPU, deterministic, TF32 off, 1 thread
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
    assert not torch.cuda.is_available(), 'CPU mode needs CUDA_VISIBLE_DEVICES=""'
    for p in (CODE_ROOT, QD, REC):
        if p not in sys.path:
            sys.path.insert(0, p)
    from models import registry
    import qfix_a
    import f2_fuse
    import f2b_fold_all
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
    return torch, registry, qfix_a, f2_fuse, f2b_fold_all


class Ops(object):
    def __init__(self, torch, qa):
        self.pair_dice = Q.Item.pair_dice
        self.prune_layer = qa.prune_layer
        self.q4 = lambda w: qa.rtn_sym_tensor(torch, w, 4)
        self.rtn_sym_channel = Q.rtn_sym_channel


class Job(object):
    def __init__(self, env, it, block, dev, train_bs=S.TRAIN_BS, smoke=False):
        torch, registry, qa, f2_fuse, f2b = env
        self.torch, self.qa, self.dev, self.it, self.block, self.smoke = torch, qa, dev, it, block, smoke
        self.ops = Ops(torch, qa)
        self.train_bs = train_bs
        self.steps = 2 if smoke else S.STEPS
        self.tag = it['tag']
        pairs = Q.load_pairs()
        self.eng = Q.Item(torch, registry, qa, f2_fuse, f2b, it, pairs.get(it['model'], []), dev, batch=Q.BATCH)
        self.ad = self.eng.ad
        self.g = self.eng.g['dep']
        self.dep = self.g.model
        self.F3 = jload('%s/%s.json' % (Q.OUT, self.tag))
        self.seed = seed_ck(self.tag)
        g = self.g
        self.layers = [n for n, m in g.leaf if id(m) in g.executed and getattr(m, 'weight', None) is not None and m.weight.dim() >= 2]
        self.numel = {n: g.mods[n].weight.numel() for n in self.layers}
        self.N = sum(self.numel.values())
        self.targets = [n for n, _ in g.targets if not self.ad.is_train_only(n)]   # w4 tensors of the inference graph
        # forward execution order of the l50 layers (first call)
        order = []
        hs = [g.mods[n].register_forward_hook((lambda nn_: (lambda m_, i_, o_: order.append(nn_) if nn_ not in order else None))(n)) for n in self.layers]
        try:
            with torch.no_grad():
                self.ad.infer(self.dep, self.eng.vimgs[0:1])
        finally:
            for h in hs:
                h.remove()
        self.exec_order = order
        # train images (images key; masks only for B4), fixed permutation per checkpoint
        r = self.eng.results
        tf = r['data_cache']['train']
        assert tf.startswith('train_')
        self.train_path = '%s/%s/%s' % (Q.CACHE_ROOT, it['ds'], tf)
        import numpy as np
        self.np = np
        z = np.load(self.train_path)
        names = [str(x) for x in z['names']]
        tr_imgs = z['images']
        zv = np.load('%s/%s/%s' % (Q.CACHE_ROOT, it['ds'], r['data_cache']['val']))
        zt = np.load('%s/%s/%s' % (Q.CACHE_ROOT, it['ds'], r['data_cache']['test']))
        vt_names = set(str(x) for x in zv['names']) | set(str(x) for x in zt['names'])
        vt_hash = set(img_hashes(np, zv['images'])) | set(img_hashes(np, zt['images']))
        self.perm, n_dup = train_order(np, self.seed, names, img_hashes(np, tr_imgs), vt_names, vt_hash)
        idx = self.perm[:N_MAIN]
        self.idx = idx
        self.xm = (torch.from_numpy(tr_imgs[idx]).float() / 255.0).to(dev)
        self.sel_names = [names[i] for i in idx]
        self.ym = None
        if block == 'sub':
            y = torch.from_numpy(z['masks'][idx]).float()
            assert float(y.max()) <= 1.0 and float(y.min()) >= 0.0
            self.ym = y.to(dev)
        sel_h = set(img_hashes(np, tr_imgs[idx]))
        self.overlap = {'n_sel_in_val': sum(1 for x in self.sel_names if x in set(str(v) for v in zv['names'])),
                        'n_sel_in_test': sum(1 for x in self.sel_names if x in set(str(v) for v in zt['names'])),
                        'n_sel_hash_in_valtest': len(sel_h & vt_hash), 'n_train_excluded_dup': n_dup, 'n_pool': int(len(self.perm))}
        assert self.overlap['n_sel_in_val'] == 0 and self.overlap['n_sel_in_test'] == 0 and self.overlap['n_sel_hash_in_valtest'] == 0, self.overlap
        assert len(idx) == N_MAIN
        g.restore()
        if smoke:                                        # smoke: code paths only, 8 images
            self.xm = self.xm[:8]
            if self.ym is not None:
                self.ym = self.ym[:8]
        self.zfp = S.logits(torch, self.ad, self.dep, self.xm, Q.BATCH)
        self.tau = tau_score()
        self._dcache = {}

    # ---------------------------------------------------------- helpers
    def sync(self):
        if self.dev.type == 'cuda':
            self.torch.cuda.synchronize()

    def orig(self, n):
        return self.g.orig[n + '.weight']

    def record_l50(self):
        return {n: self.ops.prune_layer(self.torch, self.orig(n), 0.5) for n in self.layers}

    def exact(self, kmap):
        return {n: S.exact_prune(self.torch, self.orig(n), kmap[n])[0] for n in self.layers}

    def build(self, weights=None, params=None):
        self.g.restore()
        m = copy.deepcopy(self.dep).eval()
        mods = dict(m.named_modules())
        pr = dict(m.named_parameters())
        with self.torch.no_grad():
            for n, w in (weights or {}).items():
                mods[n].weight.copy_(w)
            for n, w in (params or {}).items():
                pr[n].copy_(w)
        return m

    def zeros(self, m):
        mods = dict(m.named_modules())
        return sum(int((mods[n].weight == 0).sum()) for n in self.layers)

    def evaluate(self, m):
        torch = self.torch
        if self.smoke:
            with torch.no_grad():
                lg = self.ad.infer(m, self.xm[:2]).float()
            return {'finite_smoke': bool(torch.isfinite(lg).all())}
        d, i, f = self.eng.val_metrics(m)
        lg = self.eng.logits(m, self.eng.timgs, Q.BATCH)
        ts = self.eng.test_stats(lg, self.ref_t)
        return {'val_dice': d, 'val_iou': i, 'val_pred_fg': f, 'R': d / self.d0 if self.d0 > 0 else None,
                'agree': ts['agree'], 'agree_per_img': ts['agree_per_img'], 'flag_rescreen': bool(1.0 - ts['agree'] >= self.tau),
                'fg_q': ts['fg_q'], 'sqnr_db': ts['sqnr_db'], 'n_nonfinite_img': ts['n_nonfinite_img']}

    # ---------------------------------------------------------- diagnosis
    def diag_l50(self, n_img):
        if n_img in self._dcache:
            return self._dcache[n_img]
        r = self._diag_l50(n_img)
        self._dcache[n_img] = r
        return r

    def _diag_l50(self, n_img):
        torch = self.torch
        xs, ref = self.xm[:n_img], self.zfp[:n_img]
        pruned = self.record_l50()
        m = self.build(pruned)
        mods = dict(m.named_modules())
        D_all = S.disagreement(torch, self.ops, S.logits(torch, self.ad, m, xs), ref)
        gs, gh = {}, {}
        for n in self.layers:
            with torch.no_grad():
                mods[n].weight.copy_(self.orig(n))
            dh, ds = S.disagreement(torch, self.ops, S.logits(torch, self.ad, m, xs), ref)
            with torch.no_grad():
                mods[n].weight.copy_(pruned[n])
            gs[n] = D_all[1] - ds
            gh[n] = D_all[0] - dh
        del m
        return gs, gh, S.select_lstar(gs, gh)

    def diag_w4(self, n_img):
        torch = self.torch
        xs, ref = self.xm[:n_img], self.zfp[:n_img]
        qp = {t: self.ops.q4(self.g.orig[t]) for t in self.targets}
        m = self.build(params=qp)
        pr = dict(m.named_parameters())
        D_all = S.disagreement(torch, self.ops, S.logits(torch, self.ad, m, xs), ref)
        gs, gh = {}, {}
        for t in self.targets:
            with torch.no_grad():
                pr[t].copy_(self.g.orig[t])
            dh, ds = S.disagreement(torch, self.ops, S.logits(torch, self.ad, m, xs), ref)
            with torch.no_grad():
                pr[t].copy_(qp[t])
            gs[t] = D_all[1] - ds
            gh[t] = D_all[0] - dh
        del m
        return gs, gh, S.select_lstar(gs, gh), qp

    # ---------------------------------------------------------- tuning specs
    def spec_l50(self, m, masks, wl):
        mods = dict(m.named_modules())
        w = [(mods[n], 'weight', 'mask', (masks[n], S.ch_axis(self.torch, mods[n]), n in wl)) for n in self.layers]
        b = [mods[n] for n in self.layers if getattr(mods[n], 'bias', None) is not None]
        return {'w': w, 'bias': b}

    def spec_w4(self, m, wl):
        mods = dict(m.named_modules())
        w = []
        for t in wl:
            on, at = t.rsplit('.', 1)
            w.append((mods[on], at, 'ste', self.ops.q4))
        b = [mods[n] for n in self.layers if getattr(mods[n], 'bias', None) is not None]
        return {'w': w, 'bias': b}

    def tuned_l50(self, weights, wl, n, loss='match', seed_off=0):
        m = self.build(weights)
        masks = {k: weights[k].ne(0) for k in self.layers}
        z0 = sum(int((~masks[k]).sum()) for k in self.layers)
        st = S.tune(self.torch, self.ad, m, self.spec_l50(m, masks, set(wl)), self.xm[:n],
                    self.zfp[:n] if loss == 'match' else self.ym[:n], self.steps, self.seed + seed_off, self.train_bs, loss)
        st['mask_preserved'] = self.zeros(m) == z0
        st['global_sparsity'] = self.zeros(m) / self.N
        st['frac_tuned_of_inference_params'] = st['n_tuned_total'] / self.eng.params_dep
        return m, st

    def levels_ok(self, m):
        pr = dict(m.named_parameters())
        return max(int(self.torch.unique(pr[t].detach()).numel()) for t in self.targets) <= 15

    # ---------------------------------------------------------- arms
    def run(self):
        torch = self.torch
        g = self.g
        t_job = time.time()
        rec = {'tag': self.tag, 'block': self.block, 'model': self.it['model'], 'ds': self.it['ds'], 'k': self.it['k'],
               'seed': self.seed, 'n_train_sel': int(len(self.idx)), 'sel_names_sha': hashlib.sha256('|'.join(self.sel_names).encode()).hexdigest(),
               'overlap': self.overlap, 'ckpt_sha256': self.eng.ckpt_sha, 'results_sha256': self.eng.results_sha,
               'n_layers': len(self.layers), 'n_targets_w4': len(self.targets), 'N_pruned_params': self.N,
               'params_inf_dep': self.eng.params_dep, 'train_bs': self.train_bs, 'steps': self.steps, 'smoke': self.smoke,
               'torch': torch.__version__, 'arms': {}, 'integrity': {}}
        g.restore()
        if not self.smoke:
            rec['equiv'] = self.eng.equivalence()
            self.d0, i0, f0 = self.eng.val_metrics(self.dep)
            self.ref_t = self.eng.logits(self.dep, self.eng.timgs, Q.BATCH)
            rec['fp32'] = {'val_dice': self.d0, 'val_iou': i0, 'val_pred_fg': f0}
            rec['integrity']['fp32_repro_absdiff'] = abs(self.d0 - self.F3['fp32']['dep']['val_dice'])
        else:
            self.d0 = 1.0
        A = rec['arms']
        for arm in ARMS[self.block]:
            t0 = time.time()
            a = getattr(self, 'arm_' + arm)()
            self.sync()
            a['wall_s'] = time.time() - t0
            A[arm] = a
            if self.dev.type == 'cuda':
                torch.cuda.empty_cache()
        g.restore()
        rec['integrity']['state_restored_exact'] = bool(g.exact())
        rec['wall_s'] = time.time() - t_job
        return rec

    def _fin(self, m, extra):
        ev = self.evaluate(m)
        ev.update(extra)
        del m
        return ev

    def arm_B0(self):
        hs, integ = self.g.apply('l50')
        try:
            ev = self.evaluate(self.g.model)
            z = self.zeros(self.g.model)
        finally:
            for h in hs:
                h.remove()
            self.g.restore()
        ev.update({'integrity': integ, 'global_sparsity': z / self.N})
        if not self.smoke:
            ev['repro_absdiff_F3_D_l50'] = abs(ev['val_dice'] - self.F3['cells']['D_l50']['val_dice'])
        return ev

    def arm_B0w(self):
        hs, integ = self.g.apply('w4_pt')
        try:
            ev = self.evaluate(self.g.model)
        finally:
            for h in hs:
                h.remove()
            self.g.restore()
        ev['integrity'] = integ
        if not self.smoke:
            ev['repro_absdiff_F3_S_w4g'] = abs(ev['val_dice'] - self.F3['cells']['S_w4g']['val_dice'])
        return ev

    def arm_B5(self):
        hs, integ = self.g.apply('w4_pc')
        try:
            ev = self.evaluate(self.g.model)
        finally:
            for h in hs:
                h.remove()
            self.g.restore()
        ev['integrity'] = integ
        if not self.smoke:
            ev['repro_absdiff_F3_D_w4'] = abs(ev['val_dice'] - self.F3['cells']['D_w4']['val_dice'])
        return ev

    def arm_B1(self, n=N_MAIN):
        m, st = self.tuned_l50(self.record_l50(), self.layers, n)
        return self._fin(m, {'tune': st})

    def arm_B4(self):
        m, st = self.tuned_l50(self.record_l50(), self.layers, N_MAIN, loss='label')
        return self._fin(m, {'tune': st})

    def arm_B2(self):
        kb = S.global_magnitude_counts(self.torch, {n: self.orig(n) for n in self.layers}, 0.5)
        m = self.build(self.exact(kb))
        z = self.zeros(m)
        return self._fin(m, {'global_sparsity': z / self.N, 'exact_count': z == int(0.5 * self.N),
                             'layer_s_range': [min(kb[n] / self.numel[n] for n in self.layers), max(kb[n] / self.numel[n] for n in self.layers)]})

    def arm_B3(self):
        m = self.build(self.record_l50())
        z0 = self.zeros(m)
        self.g.restore()
        st = S.b3_correct(self.torch, self.ad, m, self.dep, self.exec_order, self.xm, shift_only=False, bs=Q.BATCH)
        st['mask_preserved'] = self.zeros(m) == z0
        return self._fin(m, {'b3': st})

    def arm_B3w(self):
        qp = {t: self.ops.q4(self.g.orig[t]) for t in self.targets}
        m = self.build(params=qp)
        self.g.restore()
        st = S.b3_correct(self.torch, self.ad, m, self.dep, self.exec_order, self.xm, shift_only=True, bs=Q.BATCH)
        st['levels_ok'] = self.levels_ok(m)
        return self._fin(m, {'b3': st})

    def arm_B1w(self):
        m = self.build(params={t: self.g.orig[t] for t in self.targets})       # latent = fp32 weights, quantised by STE
        st = S.tune(self.torch, self.ad, m, self.spec_w4(m, self.targets), self.xm, self.zfp, self.steps, self.seed, self.train_bs, 'match')
        st['levels_ok'] = self.levels_ok(m)
        st['frac_tuned_of_inference_params'] = st['n_tuned_total'] / self.eng.params_dep
        return self._fin(m, {'tune': st})

    def _sgr_core(self, n, lstar_override=None, realloc=True, tune=True, seed_off=0):
        nd = min(n, N_DIAG)
        gs, gh, L = self.diag_l50(nd)
        out = {'diag_n': nd, 'Lstar_k': len(L), 'Lstar': L, 'Lstar_params': sum(self.numel[x] for x in L),
               'gain_soft': gs, 'gain_hard': gh}
        sc = {x: gs[x] for x in L}
        if lstar_override is not None:
            L, sc = lstar_override(L, gs)
            out['random_layers'] = L
        if realloc:
            kmap = S.realloc_sparsity(self.numel, sc, 0.5)
            w = self.exact(kmap)
            out['layer_s'] = {x: kmap[x] / self.numel[x] for x in self.layers}
        else:
            w = self.record_l50()
        if tune:
            m, st = self.tuned_l50(w, L, n, seed_off=seed_off)
            out['tune'] = st
        else:
            m = self.build(w)
        z = self.zeros(m)
        out['global_sparsity'] = z / self.N
        out['exact_count'] = (z == int(0.5 * self.N)) if realloc else None
        return self._fin(m, out)

    def arm_SGR(self):
        return self._sgr_core(N_MAIN)

    def arm_A1(self):
        return self._sgr_core(N_MAIN, realloc=False)

    def arm_A2(self):
        return self._sgr_core(N_MAIN, tune=False)

    def arm_A3(self):
        rng = self.np.random.RandomState((self.seed + 1) % (2 ** 31))

        def override(L, gs):
            pick = [self.layers[i] for i in sorted(rng.choice(len(self.layers), len(L), replace=False))]
            vals = [gs[x] for x in L]
            rng.shuffle(vals)
            return pick, dict(zip(pick, vals))
        return self._sgr_core(N_MAIN, lstar_override=override)

    def arm_SGR_n10(self):
        return self._sgr_core(10)

    def arm_SGR_n30(self):
        return self._sgr_core(30)

    def arm_B1_n10(self):
        return self.arm_B1(10)

    def arm_B1_n30(self):
        return self.arm_B1(30)

    def arm_SGRw(self):
        gs, gh, L, qp = self.diag_w4(N_DIAG)
        start = dict(qp)
        for t in L:
            start[t] = self.g.orig[t]                          # latent fp32 for the tuned tensors (STE)
        m = self.build(params=start)
        st = S.tune(self.torch, self.ad, m, self.spec_w4(m, L), self.xm, self.zfp, self.steps, self.seed, self.train_bs, 'match')
        st['levels_ok'] = self.levels_ok(m)
        st['frac_tuned_of_inference_params'] = st['n_tuned_total'] / self.eng.params_dep
        return self._fin(m, {'diag_n': N_DIAG, 'Lstar_k': len(L), 'Lstar': L, 'gain_soft': gs, 'gain_hard': gh, 'tune': st})


# ================================================================== unit checks (CPU)
def main_unit():
    torch, registry, qa, f2_fuse, f2b = setup(cpu=True)
    nn = torch.nn
    torch.manual_seed(SEED0)
    res = {}

    def chk(k, ok, msg):
        res[k] = {'PASS': bool(ok), 'msg': msg}
        print('UNIT %s %s: %s' % (k, 'PASS' if ok else 'FAIL', msg), flush=True)

    class AD(object):
        @staticmethod
        def infer(m, x):
            return m(x)
    net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1, bias=True), nn.ReLU(), nn.Conv2d(8, 8, 3, padding=1, groups=8, bias=False),
                        nn.ReLU(), nn.Conv2d(8, 1, 1, bias=True)).eval()
    names = ['0', '2', '4']
    mods = dict(net.named_modules())
    numel = {n: mods[n].weight.numel() for n in names}
    N = sum(numel.values())
    # 1 realloc exact global sparsity
    k = S.realloc_sparsity(numel, {'0': 1.0, '2': 0.3}, 0.5)
    ok = sum(k.values()) == int(0.5 * N) and all(0 <= k[n] <= int(0.9 * numel[n]) for n in names)
    chk('realloc_exact', ok, 'sum k=%d target=%d per-layer s=%s' % (sum(k.values()), int(0.5 * N), {n: round(k[n] / numel[n], 3) for n in names}))
    kb = S.global_magnitude_counts(torch, {n: mods[n].weight for n in names}, 0.5)
    chk('global_mag_exact', sum(kb.values()) == int(0.5 * N), 'sum=%d' % sum(kb.values()))
    # 2 mask preserved after 3 steps
    x = torch.rand(6, 3, 16, 16)
    with torch.no_grad():
        zt = net(x)
    m = copy.deepcopy(net)
    mm = dict(m.named_modules())
    masks = {}
    with torch.no_grad():
        for n in names:
            w, kp = S.exact_prune(torch, mm[n].weight, numel[n] // 2)
            mm[n].weight.copy_(w); masks[n] = kp
    z0 = sum(int((~masks[n]).sum()) for n in names)
    spec = {'w': [(mm[n], 'weight', 'mask', (masks[n], 0, True)) for n in names], 'bias': [mm[n] for n in names if mm[n].bias is not None]}
    st = S.tune(torch, AD, m, spec, x, zt, 3, 1, 4, 'match')
    z1 = sum(int((mm[n].weight == 0).sum()) for n in names)
    chk('mask_preserved', z1 == z0 and st['steps_done'] == 3, 'zeros %d -> %d, steps %d' % (z0, z1, st['steps_done']))
    # 3 B3 statistics correction (no clip active expected)
    mc = copy.deepcopy(net)
    mcm = dict(mc.named_modules())
    with torch.no_grad():
        for n in names:
            mcm[n].weight.copy_(qa.prune_layer(torch, mcm[n].weight, 0.5))
    s3 = S.b3_correct(torch, AD, mc, net, names, x, shift_only=False, bs=4)
    fs = S.collect_stats(torch, AD, net, dict(net.named_modules()), x, names, bs=4)
    cs = S.collect_stats(torch, AD, mc, dict(mc.named_modules()), x, names, bs=4)
    err = 0.0
    for n in names:
        if mcm[n].bias is not None:
            err = max(err, float(((cs[n]['mu'] - fs[n]['mu']).abs() / fs[n]['sd'].clamp(min=1e-8)).max()),
                      float(((cs[n]['sd'] - fs[n]['sd']).abs() / fs[n]['sd'].clamp(min=1e-8)).max()))
        else:
            err = max(err, float(((cs[n]['rms'] - fs[n]['rms']).abs() / fs[n]['rms'].clamp(min=1e-8)).max()))
    chk('b3_stats_match', err < 1e-4 and s3['n_gamma_clipped'] == 0, 'max rel err %.2e, counters %s' % (err, s3))
    # 4 STE w4 levels
    m4 = copy.deepcopy(net)
    m4m = dict(m4.named_modules())
    q4 = lambda w: qa.rtn_sym_tensor(torch, w, 4)
    spec4 = {'w': [(m4m[n], 'weight', 'ste', q4) for n in names], 'bias': [m4m[n] for n in names if m4m[n].bias is not None]}
    S.tune(torch, AD, m4, spec4, x, zt, 3, 1, 4, 'match')
    lv = max(int(torch.unique(m4m[n].weight).numel()) for n in names)
    chk('ste_w4_levels', lv <= 15, 'max levels %d' % lv)
    res['ALL_PASS'] = all(v['PASS'] for v in res.values() if isinstance(v, dict))
    Q.dump_json(BASE + '/unit.json', res)
    print('UNIT ALL', 'PASS' if res['ALL_PASS'] else 'FAIL', flush=True)


# ================================================================== smoke (CPU)
def main_smoke(tag, blocks):
    env = setup(cpu=True)
    torch = env[0]
    items = {i['tag']: i for i in jload(QD + '/ITEMS.json')}
    it = items[tag]
    sd = BASE + '/' + os.environ.get('SGR_SMOKE_DIR', 'smoke')
    os.makedirs(sd, exist_ok=True)
    for block in blocks:
        t0 = time.time()
        job = Job(env, it, block, torch.device('cpu'), smoke=True)
        rec = job.run()
        Q.dump_json('%s/%s__%s.json' % (sd, tag, block), rec)
        summ = {}
        for a, v in rec['arms'].items():
            s = {'finite': v.get('finite_smoke')}
            for kk in ('global_sparsity', 'exact_count', 'Lstar_k'):
                if kk in v:
                    s[kk] = v[kk]
            if 'tune' in v:
                s.update({kk: v['tune'][kk] for kk in ('mask_preserved', 'levels_ok', 'steps_done', 'n_tuned_total') if kk in v['tune']})
            if 'b3' in v:
                s.update({'b3_' + kk: vv for kk, vv in v['b3'].items()})
            if 'integrity' in v:
                s['integrity'] = v['integrity']
            summ[a] = s
        print('SMOKE %s %s %.1fs overlap=%s restored=%s %s' % (tag, block, time.time() - t0, rec['overlap'], rec['integrity']['state_restored_exact'],
                                                         json.dumps(summ, sort_keys=True)), flush=True)
        del job


# ================================================================== GPU queue worker
def gpu_uuid(idx):
    o = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], capture_output=True, text=True).stdout
    for ln in o.strip().splitlines():
        i, u = [s.strip() for s in ln.split(',')]
        if i == str(idx):
            return u
    raise RuntimeError('gpu %s not found' % idx)


def foreign_pids(uuid):
    o = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], capture_output=True, text=True).stdout
    me = os.getpid()
    return [int(p) for u, p in (tuple(s.strip() for s in ln.split(',')) for ln in o.strip().splitlines() if ln.strip())
            if u == uuid and p.isdigit() and int(p) != me]


def pid_alive_sgr(pid):
    try:
        with open('/proc/%d/cmdline' % pid, 'rb') as f:
            return b'sgr_run.py' in f.read()
    except Exception:
        return False


def keys():
    it = jload(BASE + '/ITEMS_SGR.json')
    return [(t, b) for b in BLOCK_ORDER for t in it['blocks'][b]]


def kname(t, b):
    return '%s/%s__%s' % (OUT, t, b)


def finished(t, b):
    return os.path.exists(kname(t, b) + '.DONE') or os.path.exists(kname(t, b) + '.FAIL')


def used_seconds():
    u = 0.0
    for p in glob.glob(OUT + '/*.DONE') + glob.glob(OUT + '/*.FAIL') + glob.glob(OUT + '/*.OOM'):
        try:
            with open(p) as f:
                u += float(f.read().split()[0])
        except Exception:
            pass
    return u


def stopped():
    return glob.glob(OUTD + '/STOPPED_*')


def write_once(path, text):
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, 'w') as f:
        f.write(text)
    return True


def stop(reason, msg):
    write_once('%s/STOPPED_%s' % (OUTD, reason), '%s %s\n' % (time.strftime('%F %T %Z'), msg))
    print('[%s] STOP %s %s' % (time.strftime('%T'), reason, msg), flush=True)


def claim(t, b, wid):
    import fcntl
    with open(OUTD + '/queue.lock', 'a') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if finished(t, b):
                return False
            cf = kname(t, b) + '.CLAIM'
            if os.path.exists(cf):
                try:
                    c = jload(cf)
                    if c.get('pid') != os.getpid() and pid_alive_sgr(int(c.get('pid', -1))):
                        return False
                except Exception:
                    pass
            Q.dump_json(cf, {'wid': wid, 'pid': os.getpid(), 'host': socket.gethostname(), 'time': time.strftime('%F %T %Z')})
            return True
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def live_claims():
    n = 0
    for cf in glob.glob(OUT + '/*.CLAIM'):
        base = cf[:-len('.CLAIM')]
        if os.path.exists(base + '.DONE') or os.path.exists(base + '.FAIL'):
            continue
        try:
            c = jload(cf)
            if int(c['pid']) != os.getpid() and pid_alive_sgr(int(c['pid'])):
                n += 1
        except Exception:
            pass
    return n


def maybe_all_done():
    ks = keys()
    nfin = sum(1 for t, b in ks if finished(t, b))
    st = [os.path.basename(p) for p in stopped()]
    if nfin == len(ks):
        write_once(OUTD + '/ALL_DONE', json.dumps({'time': time.strftime('%F %T %Z'), 'partial': False, 'n_units': len(ks),
                                                  'n_done': len(glob.glob(OUT + '/*.DONE')), 'n_fail': len(glob.glob(OUT + '/*.FAIL')),
                                                  'used_s': used_seconds()}) + '\n')
    elif st == ['STOPPED_CAP'] and live_claims() == 0:
        miss = {}
        for t, b in ks:
            if not finished(t, b):
                miss[b] = miss.get(b, 0) + 1
        write_once(OUTD + '/ALL_DONE', json.dumps({'time': time.strftime('%F %T %Z'), 'partial': True, 'missing_by_block': miss,
                                                  'used_s': used_seconds()}) + '\n')


def main_run(wid):
    phys = {'gpu0': 0, 'gpu1': 1}[wid]
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(phys), 'CUDA_VISIBLE_DEVICES must be %d' % phys
    os.makedirs(OUT, exist_ok=True)
    if wid == 'gpu1':
        while not os.path.exists(ABL_DONE):
            if stopped() or os.path.exists(OUTD + '/ALL_DONE'):
                print('[%s] gpu1: stop/ALL_DONE before abl finished; exit' % time.strftime('%T'), flush=True)
                return
            print('[%s] gpu1 WAIT for %s' % (time.strftime('%T'), ABL_DONE), flush=True)
            time.sleep(300)
    env = setup(cpu=False)
    torch = env[0]
    dev = torch.device('cuda:0')
    uuid = gpu_uuid(phys)
    items = {i['tag']: i for i in jload(QD + '/ITEMS.json')}
    man = Q.manifest()
    print('[%s] worker %s pid=%d gpu=%s start, used=%.0fs' % (time.strftime('%F %T'), wid, os.getpid(), uuid, used_seconds()), flush=True)
    consec_fail = 0
    while True:
        if stopped():
            break
        todo = None
        for t, b in keys():
            if not finished(t, b) and claim(t, b, wid):
                todo = (t, b)
                break
        if todo is None:
            break
        t, b = todo
        while True:
            fp = foreign_pids(uuid)
            if not fp:
                break
            print('[%s] WAIT foreign pid(s) on GPU %d: %s' % (time.strftime('%T'), phys, fp), flush=True)
            time.sleep(60)
        if used_seconds() >= CAP_S:
            stop('CAP', 'compute %.0fs >= cap %.0fs before %s__%s' % (used_seconds(), CAP_S, t, b))
            break
        it = items[t]
        bs = 4 if os.path.exists(kname(t, b) + '.OOM') else S.TRAIN_BS
        t0 = time.time()
        try:
            for f in ('checkpoint.pth', 'results.json'):
                p = it['dir'] + '/' + f
                if man.get(p) != Q.sha_file(p):
                    stop('SHA', p)
                    raise SystemExit(5)
            job = Job(env, it, b, dev, train_bs=bs)
            rec = job.run()
            del job
            rec['gpu_wall_s'] = time.time() - t0
            rec['worker'] = wid
            Q.dump_json(kname(t, b) + '.json', rec)
            with open(kname(t, b) + '.DONE', 'w') as f:
                f.write('%.1f\n' % rec['gpu_wall_s'])
            consec_fail = 0
            ig = rec['integrity']
            msg = 'fp32_repro=%.1e restored=%s E1=%s' % (ig['fp32_repro_absdiff'], ig['state_restored_exact'], rec['equiv']['E1_pass'])
            if 'B0' in rec['arms']:
                msg += ' B0_repro=%.1e' % rec['arms']['B0']['repro_absdiff_F3_D_l50']
            if 'B0w' in rec['arms']:
                msg += ' B0w_repro=%.1e' % rec['arms']['B0w']['repro_absdiff_F3_S_w4g']
            print('[%s] %s %s__%s %.1fs bs=%d %s used=%.0fs' % (time.strftime('%T'), wid, t, b, rec['gpu_wall_s'], bs, msg, used_seconds()), flush=True)
            if ig['fp32_repro_absdiff'] > REPRO_TOL or ('B0' in rec['arms'] and rec['arms']['B0']['repro_absdiff_F3_D_l50'] > REPRO_TOL):
                stop('REPRO', '%s__%s fp32 %.2e B0 %s' % (t, b, ig['fp32_repro_absdiff'], rec['arms'].get('B0', {}).get('repro_absdiff_F3_D_l50')))
            if not rec['equiv']['E1_pass']:
                stop('E1', '%s E1 %s' % (t, rec['equiv']['E1_max_abs_fp64']))
            if not ig['state_restored_exact']:
                stop('STATE', t)
        except SystemExit:
            break
        except Exception as ex:
            tb = traceback.format_exc()
            is_oom = 'out of memory' in str(ex).lower()
            try:
                del job
            except Exception:
                pass
            torch.cuda.empty_cache()
            if is_oom:
                with open(kname(t, b) + '.OOM', 'a') as f:
                    f.write('%.1f\n%s\n' % (time.time() - t0, tb))
                n_oom = len(glob.glob(OUT + '/*.OOM'))
                print('[%s] %s %s__%s OOM (#%d)' % (time.strftime('%T'), wid, t, b, n_oom), flush=True)
                if bs == 4 or n_oom >= 2:
                    stop('OOM', 'OOM #%d at %s__%s bs=%d' % (n_oom, t, b, bs))
                    break
                continue
            with open(kname(t, b) + '.FAIL', 'w') as f:
                f.write('%.1f\n%s' % (time.time() - t0, tb))
            print('[%s] %s %s__%s FAIL %s' % (time.strftime('%T'), wid, t, b, tb.strip().splitlines()[-1]), flush=True)
            consec_fail += 1
            if consec_fail >= MAX_CONSEC_FAIL:
                stop('FAILS', '%d consecutive FAIL units on %s (last %s__%s)' % (consec_fail, wid, t, b))
                break
        torch.cuda.empty_cache()
    maybe_all_done()
    print('[%s] worker %s end, used=%.0fs stopped=%s all_done=%s' % (time.strftime('%F %T'), wid, used_seconds(), stopped(), os.path.exists(OUTD + '/ALL_DONE')), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--unit', action='store_true')
    ap.add_argument('--smoke', default='')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--worker', default='gpu0')
    ap.add_argument('--blocks', default=','.join(BLOCK_ORDER))
    a = ap.parse_args()
    if a.unit:
        main_unit()
    if a.smoke:
        main_smoke(a.smoke, a.blocks.split(','))
    if a.run:
        main_run(a.worker)
