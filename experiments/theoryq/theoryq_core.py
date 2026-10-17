# -*- coding: utf-8 -*-
"""THEORY_Q predictor core (PREREG_THEORYQ.md sections 2-5). Weight tensors in, numbers out. No images, no labels.

Operators are NOT re-implemented: rtn_sym_tensor / prune_layer come from qfix_a.py (sha256 b36df82f...94c6) and
rtn_sym_channel from qdeploy.py (sha256 644bd5ee...4d96), imported from frozen_src/ after a sha256 check.
AUC = qdeploy_analyze.auc (sha256 f93895df...b899), unchanged.
"""
import os
import sys
import copy
import math
import random
import hashlib

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get('THEORYQ_SRC', os.path.join(HERE, 'frozen_src'))
FROZEN = {'qfix_a.py': 'b36df82ffd238ccd2fadee1b02ab7df016e1bcc6500fc2c865c6adbc52fa94c6',
          'qdeploy.py': '644bd5eeba8c1ded4d61810da8c74ac1ab3909e0e504ce76674ba2c819e5b899',
          'qdeploy_analyze.py': 'f93895df79de58d3e612f1dfeea314fef4329bfa3ca87663f8db4b2f0a6166e2'}
for _f, _h in FROZEN.items():
    with open(os.path.join(SRC, _f), 'rb') as _fh:
        assert hashlib.sha256(_fh.read()).hexdigest() == _h, ('frozen source sha mismatch', _f)
sys.path.insert(0, SRC)
import qfix_a as QA            # noqa: E402  (stdlib-only imports at module level)
import qdeploy as QD           # noqa: E402
import qdeploy_analyze as QAN  # noqa: E402

BITS = 4
SPARSITY = 0.5
BOOT_B = 2000
BOOT_SEED = 20261017
GAMMA_TOL = 1e-3           # BN channel "switched off" iff |gamma_c| <= GAMMA_TOL * max_c' |gamma_c'| (same BN module)
NEAR_TOL = 1e-5            # near-threshold channel: | r_c / A - 1/(2 qm) | <= NEAR_TOL


def _to_axis0(w, axis):
    return w.transpose(0, axis) if axis != 0 else w


def quantize(torch, w, axis, mode, bits=BITS):
    if mode == 'pt':
        return QA.rtn_sym_tensor(torch, w.detach(), bits)
    if mode == 'pc':
        return QD.rtn_sym_channel(torch, w.detach(), bits, axis)[0]
    raise ValueError(mode)


def channel_allzero(w, axis):
    """bool per output channel (along axis): channel identically zero."""
    return (_to_axis0(w, axis).reshape(w.shape[axis], -1) == 0).all(1)


def zeroed_mask(torch, w, axis, mode, bits=BITS, q=None):
    """bool per output channel: NOT all-zero in w but all-zero after RTN (mode 'pt' = w4_pt, 'pc' = w4_pc)."""
    if q is None:
        q = quantize(torch, w, axis, mode, bits)
    return (~channel_allzero(w.detach(), axis)) & channel_allzero(q, axis)


def zeroed_channels(torch, w, axis, mode, bits=BITS):
    """(n_zeroed, n_channels) for zeroed_mask."""
    return int(zeroed_mask(torch, w, axis, mode, bits).sum()), int(w.shape[axis])


def z_stat(torch, tensors, mode, bits=BITS):
    """tensors: list of (name, weight, axis). Z = max_l z_l (primary), Zglob = sum zeroed / sum channels, per-layer list."""
    per = []
    nz_tot, nc_tot = 0, 0
    for n, w, ax in tensors:
        nz, nc = zeroed_channels(torch, w, ax, mode, bits)
        per.append((n, nz / nc))
        nz_tot += nz; nc_tot += nc
    return {'Z': max([z for _, z in per] or [0.0]), 'Zglob': (nz_tot / nc_tot) if nc_tot else 0.0, 'per_layer': per}


def zeroed_energy(torch, w, axis, bits=BITS, dead=None):
    """Zw_l: fraction of ||W_l||_F^2 in channels zeroed by per-tensor RTN (Prop 1 mask fed to Prop 3)."""
    w = w.detach()
    if dead is None:
        dead = zeroed_mask(torch, w, axis, 'pt', bits)
    w0 = _to_axis0(w, axis).reshape(w.shape[axis], -1).double()
    tot = float((w0 ** 2).sum())
    return float((w0[dead] ** 2).sum()) / tot if tot > 0 else 0.0


def gamma_switched_off_copy(torch, model, tol=GAMMA_TOL):
    """Deep copy of a training-graph model in which every BatchNorm channel with |gamma_c| <= tol * max |gamma| (same BN)
    gets gamma_c = 0 exactly. Folding / fusing this copy with the unchanged F3 code zeroes exactly the deployed weight
    channels whose every contributing BN channel was switched off (mapping-free, also for PBE RepConv)."""
    m0 = copy.deepcopy(model)
    n = 0
    with torch.no_grad():
        for mod in m0.modules():
            if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm) and mod.weight is not None:
                g = mod.weight.detach().abs()
                if g.numel() and float(g.max()) > 0:
                    off = g <= tol * g.max()
                    mod.weight[off] = 0.0
                    n += int(off.sum())
    return m0, n


def used_param_names(torch, infer, model, x1, leaf_executed_names):
    """Executed-tensor rule: parameter n is executed iff it receives a (non-None) gradient from infer(model, x1).sum(),
    OR its owner is a leaf Conv/Linear module that the F3 forward hook saw executed (union; covers no_grad sub-paths).
    Returns (set of names, autograd_ok)."""
    for p in model.parameters():
        p.grad = None
    ok = True
    used = set()
    try:
        with torch.enable_grad():
            req = {n: p.requires_grad for n, p in model.named_parameters()}
            for p in model.parameters():
                p.requires_grad_(True)
            out = infer(model, x1)
            out.float().sum().backward()
        used = {n for n, p in model.named_parameters() if p.grad is not None}
        for n, p in model.named_parameters():
            p.grad = None
            p.requires_grad_(req[n])
    except Exception:
        ok = False
        for p in model.parameters():
            p.grad = None
    for n, _ in model.named_parameters():
        owner = n.rsplit('.', 1)[0] if '.' in n else ''
        if owner in leaf_executed_names:
            used.add(n)
    return used, ok


def pruned_energy(torch, w, s=SPARSITY):
    """m_l = ||W_l - prune_layer(W_l, s)||_F^2 / ||W_l||_F^2 with the l50 operator (qfix_a.prune_layer)."""
    w = w.detach()
    p = QA.prune_layer(torch, w, s)
    d = (w - p).double()
    tot = float((w.double() ** 2).sum())
    return float((d ** 2).sum()) / tot if tot > 0 else 0.0


def m_stat(torch, layers, s=SPARSITY):
    """layers: list of (name, weight). Mstat = max_l m_l (primary), E = sum_l sqrt(m_l), Msum = sum_l m_l."""
    per = [(n, pruned_energy(torch, w, s)) for n, w in layers]
    ms = [m for _, m in per]
    return {'Mstat': max(ms or [0.0]), 'E': sum(math.sqrt(m) for m in ms), 'Msum': sum(ms), 'per_layer': per}


# ------------------------------------------------------------------ cell level
def cell_mean(rows, key):
    """rows: dicts with model, ds, key. Cell statistic = arithmetic mean over the included splits."""
    c = {}
    for r in rows:
        c.setdefault((r['model'], r['ds']), []).append(r[key])
    return {k: sum(v) / len(v) for k, v in c.items()}


def cell_max(rows, key):
    c = {}
    for r in rows:
        c.setdefault((r['model'], r['ds']), []).append(r[key])
    return {k: max(v) for k, v in c.items()}


def cell_auc(score, label):
    """score, label: dict cell -> value / bool (True = collapsed). AUC via the frozen qdeploy_analyze.auc."""
    keys = sorted(score)
    return QAN.auc([score[k] for k in keys if label[k]], [score[k] for k in keys if not label[k]])


def _pct(vals):
    vals = sorted(vals)
    return [vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]]


def cell_boot(score, label, B=BOOT_B, seed=BOOT_SEED, null=0.5):
    """Cell bootstrap (66 cells resampled with replacement; the cell is the cluster, as in q_robust.cluster_boot).
    point, percentile CI (q_robust indexing), one-sided p for H0: AUC <= null = (1 + #{AUC* <= null}) / (n_valid + 1)."""
    keys = sorted(score)
    rng = random.Random(seed)
    vals = []
    for _ in range(B):
        samp = [keys[rng.randrange(len(keys))] for _k in keys]
        v = QAN.auc([score[k] for k in samp if label[k]], [score[k] for k in samp if not label[k]])
        if v is not None:
            vals.append(v)
    p = (1 + sum(1 for v in vals if v <= null)) / (len(vals) + 1)
    return {'point': cell_auc(score, label), 'ci95': _pct(vals), 'p_le_null': p, 'null': null, 'n_boot_valid': len(vals),
            'n_pos': sum(1 for k in keys if label[k]), 'n_neg': sum(1 for k in keys if not label[k])}


def ckpt_cluster_boot(rows, key, B=BOOT_B, seed=BOOT_SEED):
    """Checkpoint-level AUC with cell-cluster bootstrap, same algorithm as q_robust.cluster_boot.
    rows: dicts with model, ds, label (bool), key."""
    cells = {}
    for x in rows:
        cells.setdefault((x['model'], x['ds']), []).append(x)
    keys = sorted(cells)
    rng = random.Random(seed)

    def a(rr):
        return QAN.auc([x[key] for x in rr if x['label']], [x[key] for x in rr if not x['label']])
    vals = []
    for _ in range(B):
        samp = []
        for _k in keys:
            samp += cells[keys[rng.randrange(len(keys))]]
        v = a(samp)
        if v is not None:
            vals.append(v)
    return {'point': a(rows), 'ci95': _pct(vals) if vals else [None, None], 'n_cells': len(keys),
            'n_pos': sum(1 for x in rows if x['label']), 'n_neg': sum(1 for x in rows if not x['label'])}


def holm(pvals, alpha=0.05):
    """pvals: dict name -> p. Returns dict name -> (adjusted p, reject)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, run, stop = {}, 0.0, False
    for i, (n, p) in enumerate(items):
        adj = min(1.0, max(run, (m - i) * p))
        run = adj
        rej = (not stop) and adj < alpha
        if not rej:
            stop = True
        out[n] = (adj, rej)
    return out
