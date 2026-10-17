# -*- coding: utf-8 -*-
"""PREREG_SGR operators (torch passed in; no device assumptions).
exact_prune, realloc_sparsity, global_magnitude_counts, MaskScale / STEQuant parametrisations, losses, diagnosis, B3 statistics
correction, tuning loop.  Record operators (prune_layer, rtn_sym_tensor, rtn_sym_channel, pair_dice) are imported from the
frozen F3 / QFIX files by the caller and passed in through the `ops` namespace."""
import math
import copy

C_REALLOC = 0.5
S_MAX = 0.9
STEPS = 300
LR = 1e-4
TRAIN_BS = 8
GAMMA_CLIP = (0.1, 10.0)
EPS = 1e-6


class StopForward(Exception):
    pass


def soft_dice(torch, p, q):
    p = p.flatten(1); q = q.flatten(1)
    return (2 * (p * q).sum(1) + EPS) / (p.sum(1) + q.sum(1) + EPS)


def disagreement(torch, ops, lg, ref):
    """(D_hard, D_soft): D_hard = 1 - mean qdeploy pair_dice of thresholded masks (non-finite image -> empty mask, as F3
    test_stats); D_soft = 1 - mean soft Dice of the sigmoids."""
    fin = torch.isfinite(lg).flatten(1).all(1)
    mq = (torch.sigmoid(lg) > 0.5) & fin.view(-1, 1, 1, 1)
    mr = torch.sigmoid(ref) > 0.5
    ag = ops.pair_dice(torch, mr, mq)
    dh = 1.0 - sum(ag) / len(ag)
    ds = 1.0 - float(soft_dice(torch, torch.sigmoid(torch.nan_to_num(lg.float(), nan=0.0, posinf=50.0, neginf=-50.0)), torch.sigmoid(ref)).mean())
    return dh, ds


def logits(torch, ad, model, x, bs=16):
    out = []
    with torch.no_grad():
        for i in range(0, x.shape[0], bs):
            out.append(ad.infer(model, x[i:i + bs]).float())
    return torch.cat(out, 0)


# ------------------------------------------------------------------ masks
def exact_prune(torch, w, k):
    """zero exactly the k smallest-|w| entries (ties broken by flat index) -> (pruned, keep mask)."""
    keep = torch.ones(w.numel(), dtype=torch.bool)
    if k > 0:
        idx = torch.argsort(w.detach().abs().reshape(-1).cpu(), stable=True)[:k]
        keep[idx] = False
    keep = keep.view_as(w).to(w.device)
    return torch.where(keep, w, torch.zeros_like(w)), keep


def realloc_sparsity(numel, score, s_glob=0.5, c=C_REALLOC, smin=0.0, smax=S_MAX):
    """s_l = clip(t - c a_l, smin, smax), a_l = max(0, score_l) / max score; t by bisection so that sum_l s_l n_l = s_glob N;
    integer zero counts k_l with the rounding residual placed on the largest layers with slack; sum k_l = floor(s_glob N)."""
    names = list(numel)
    N = sum(numel.values())
    mx = max([max(0.0, score.get(n, 0.0)) for n in names] + [0.0])
    a = {n: (max(0.0, score.get(n, 0.0)) / mx if mx > 0 else 0.0) for n in names}

    def tot(t):
        return sum(min(smax, max(smin, t - c * a[n])) * numel[n] for n in names)
    lo, hi = 0.0, 1.0 + c
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if tot(mid) < s_glob * N:
            lo = mid
        else:
            hi = mid
    s = {n: min(smax, max(smin, hi - c * a[n])) for n in names}
    k = {n: int(round(s[n] * numel[n])) for n in names}
    K = int(s_glob * N)
    diff = K - sum(k.values())
    order = sorted(names, key=lambda n: (-numel[n], n))
    i = 0
    while diff != 0:
        n = order[i % len(order)]
        step = 1 if diff > 0 else -1
        room = (int(smax * numel[n]) - k[n]) if step > 0 else (k[n] - int(math.ceil(smin * numel[n])))
        if room > 0:
            d = step * min(abs(diff), room)
            k[n] += d
            diff -= d
        i += 1
        if i > 10 * len(order) + 10:
            raise RuntimeError('cannot place residual %d' % diff)
    return k


def global_magnitude_counts(torch, ws, s_glob=0.5):
    """B2: one global |w| threshold over all layers, exactly floor(s N) zeros (ties by concatenation order) -> {layer: k_l}."""
    names = list(ws)
    allw = torch.cat([ws[n].detach().abs().reshape(-1).cpu() for n in names])
    K = int(s_glob * allw.numel())
    idx = torch.argsort(allw, stable=True)[:K]
    owner = torch.cat([torch.full((ws[n].numel(),), j, dtype=torch.long) for j, n in enumerate(names)])
    cnt = torch.bincount(owner[idx], minlength=len(names))
    return {n: int(cnt[j]) for j, n in enumerate(names)}


def select_lstar(gain_s, gain_h):
    pos = sorted([n for n in gain_s if gain_s[n] > 0], key=lambda n: (-gain_s[n], -gain_h[n], n))
    tot = sum(gain_s[n] for n in pos)
    L, acc = [], 0.0
    for n in pos:
        if acc >= 0.8 * tot:
            break
        L.append(n)
        acc += gain_s[n]
    if not L:
        L = sorted(gain_s, key=lambda n: (-gain_s[n], -gain_h[n], n))[:1]
    return L


# ------------------------------------------------------------------ parametrisations
def make_param_classes(torch):
    nn = torch.nn

    class MaskScale(nn.Module):
        """W * mask * gamma (gamma per output channel, init 1, refolded at export)."""
        def __init__(self, mask, axis, n_ch, train_gamma=True):
            super().__init__()
            self.register_buffer('mask', mask.to(torch.float32))
            sh = [1] * mask.dim()
            sh[axis] = n_ch
            self.gamma = nn.Parameter(torch.ones(sh, device=mask.device), requires_grad=train_gamma)

        def forward(self, w):
            return w * self.mask * self.gamma

    class STEQuant(nn.Module):
        """per-tensor 4-bit RTN with straight-through gradient (F3 rtn_sym_tensor in the forward)."""
        def __init__(self, qfn):
            super().__init__()
            self.qfn = qfn

        def forward(self, w):
            return w + (self.qfn(w) - w).detach()
    return MaskScale, STEQuant


def ch_axis(torch, mod):
    return 1 if isinstance(mod, torch.nn.modules.conv._ConvTransposeNd) else 0


# ------------------------------------------------------------------ tuning
def tune(torch, ad, model, spec, x, target, steps, seed, bs, loss_kind):
    """spec: list of (owner_module, attr, kind, info): kind 'mask' (info=(mask, axis, train_w)), 'ste' (info=qfn, latent
    trainable) ; plus list of bias owners in spec['bias'].  Returns stats dict.  Model stays eval()."""
    from torch.nn.utils import parametrize
    MaskScale, STEQuant = make_param_classes(torch)
    for p in model.parameters():
        p.requires_grad_(False)
    tun, n_w, n_g, n_b = [], 0, 0, 0
    for owner, attr, kind, info in spec['w']:
        if kind == 'mask':
            mask, axis, train_w = info
            w = getattr(owner, attr)
            parametrize.register_parametrization(owner, attr, MaskScale(mask, axis, w.shape[axis]))
            pz = getattr(owner.parametrizations, attr)
            tun.append(pz[0].gamma); n_g += pz[0].gamma.numel()
            if train_w:
                pz.original.requires_grad_(True); tun.append(pz.original); n_w += int(mask.sum())
        elif kind == 'ste':
            parametrize.register_parametrization(owner, attr, STEQuant(info))
            pz = getattr(owner.parametrizations, attr)
            pz.original.requires_grad_(True); tun.append(pz.original); n_w += pz.original.numel()
    for owner in spec['bias']:
        owner.bias.requires_grad_(True); tun.append(owner.bias); n_b += owner.bias.numel()
    st = {'n_tuned_weights': n_w, 'n_tuned_gamma': n_g, 'n_tuned_bias': n_b, 'n_tuned_total': n_w + n_g + n_b,
          'steps_done': 0, 'nonfinite_loss_step': None}
    if steps > 0 and tun:
        opt = torch.optim.Adam(tun, lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0.0)
        gen = torch.Generator().manual_seed(int(seed))
        bce = torch.nn.BCEWithLogitsLoss()
        for s in range(steps):
            b = torch.randint(0, x.shape[0], (bs,), generator=gen).to(x.device)
            zc = ad.infer(model, x[b]).float()
            if loss_kind == 'match':
                zt = target[b]
                loss = (1 - soft_dice(torch, torch.sigmoid(zc), torch.sigmoid(zt)).mean()) + ((zc - zt) ** 2).mean()
            else:
                y = target[b]
                loss = (1 - soft_dice(torch, torch.sigmoid(zc), y).mean()) + bce(zc, y)
            if not bool(torch.isfinite(loss)):
                st['nonfinite_loss_step'] = s
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sch.step()
            st['steps_done'] = s + 1
    # export: mask -> refold (W*mask*gamma); ste -> latent re-quantised exactly
    for owner, attr, kind, info in spec['w']:
        if kind == 'mask':
            parametrize.remove_parametrizations(owner, attr, leave_parametrized=True)
        else:
            parametrize.remove_parametrizations(owner, attr, leave_parametrized=False)
            with torch.no_grad():
                getattr(owner, attr).copy_(info(getattr(owner, attr).detach()))
    for p in model.parameters():
        p.requires_grad_(False)
    return st


# ------------------------------------------------------------------ B3 statistics correction
def _chan_stats(torch, y, is_linear):
    """per-channel sum and sum of squares (float32 per batch, accumulated in float64 by the caller)."""
    y = y.detach().float()
    if is_linear or y.dim() <= 2:
        dims = list(range(y.dim() - 1))                  # channel = last dim
        c = y.numel() // y.shape[-1]
    else:
        dims = [0] + list(range(2, y.dim()))             # channel = dim 1
        c = y.numel() // y.shape[1]
    return y.sum(dims).double(), (y * y).sum(dims).double(), c


def collect_stats(torch, ad, model, mods, x, names, stop_after=None, bs=16):
    """per-channel (sum, sumsq, count, n_calls_first_batch) of the outputs of modules `names` over images x."""
    acc = {n: None for n in names}
    calls = {n: 0 for n in names}
    hs = []

    def mk(n, m):
        is_lin = isinstance(m, torch.nn.Linear)

        def h(m_, i_, o_):
            s1, s2, c = _chan_stats(torch, o_, is_lin)
            if acc[n] is None:
                acc[n] = [s1, s2, c]
            else:
                acc[n][0] += s1; acc[n][1] += s2; acc[n][2] += c
            calls[n] += 1
            if stop_after is not None and n == stop_after:
                raise StopForward()
        return h
    for n in names:
        hs.append(mods[n].register_forward_hook(mk(n, mods[n])))
    ncall_first = None
    try:
        with torch.no_grad():
            for i in range(0, x.shape[0], bs):
                try:
                    ad.infer(model, x[i:i + bs])
                except StopForward:
                    pass
                if ncall_first is None:
                    ncall_first = dict(calls)
    finally:
        for h in hs:
            h.remove()
    out = {}
    for n in names:
        s1, s2, c = acc[n]
        mu = s1 / c
        var = (s2 / c - mu ** 2).clamp(min=0.0)
        out[n] = {'mu': mu, 'sd': var.sqrt(), 'rms': (s2 / c).sqrt(), 'calls': ncall_first[n]}
    return out


def b3_correct(torch, ad, model_c, model_fp, exec_order, x, shift_only=False, bs=16):
    """sequential per-channel output-statistics correction toward the fp32 deployed graph (PREREG_SGR §5 B3).
    shift_only=True (w4): bias correction only.  Returns counters."""
    mods_c = dict(model_c.named_modules())
    mods_f = dict(model_fp.named_modules())
    fp = collect_stats(torch, ad, model_fp, mods_f, x, exec_order, bs=bs)
    n_scaled, n_shifted, n_skipped_multi, n_clipped = 0, 0, 0, 0
    for n in exec_order:
        if fp[n]['calls'] != 1:
            n_skipped_multi += 1
            continue
        cs = collect_stats(torch, ad, model_c, mods_c, x, [n], stop_after=n, bs=bs)[n]
        m = mods_c[n]
        ax = ch_axis(torch, m)
        has_b = getattr(m, 'bias', None) is not None
        with torch.no_grad():
            if shift_only:
                if has_b:
                    m.bias.add_((fp[n]['mu'] - cs['mu']).to(m.bias.dtype))
                    n_shifted += 1
                continue
            if has_b:
                g = fp[n]['sd'] / cs['sd'].clamp(min=1e-8)
            else:
                g = fp[n]['rms'] / cs['rms'].clamp(min=1e-8)
            gc = g.clamp(*GAMMA_CLIP)
            n_clipped += int((gc != g).sum())
            sh = [1] * m.weight.dim(); sh[ax] = -1
            m.weight.mul_(gc.to(m.weight.dtype).view(sh))
            n_scaled += 1
            if has_b:
                m.bias.copy_((gc * (m.bias.double() - cs['mu']) + fp[n]['mu']).to(m.bias.dtype))
                n_shifted += 1
    return {'n_scaled': n_scaled, 'n_shifted': n_shifted, 'n_skipped_multicall': n_skipped_multi, 'n_gamma_clipped': n_clipped}
