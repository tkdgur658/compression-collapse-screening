"""F2b: deployment transformation applied to all 11 models.
Generic BN folding: every BatchNorm whose input tensor is exactly the output tensor of a Conv/ConvTranspose/Linear (hook-traced, each module called once)
is folded into that layer; BN -> Identity. PBE-Net additionally gets RepConv merging (f2_fuse.fuse_pbe).
Equivalence on 4 images from data/cache/v1/<ds>/test_1_*.npz in fp32 and fp64; counts params and thop MACs*2 (same convention as F2)."""
import os as _ccs_os  # path configuration of the released copy (README, Configuration)
CCS_REPO_ROOT = _ccs_os.path.normpath(_ccs_os.path.join(_ccs_os.path.dirname(_ccs_os.path.abspath(__file__)), '../..'))
CCS_WORK_ROOT = _ccs_os.environ.get('CCS_WORK', _ccs_os.path.join(CCS_REPO_ROOT, 'experiments'))
import sys, os, json, copy, glob
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ccs_os.path.join(CCS_WORK_ROOT, 'qfix', 'A', 'code')); sys.path.insert(0, os.path.join(HERE, '_pylib'))
from models import registry
LN = os.environ.get('CCS_CHECKPOINTS', os.path.join(CCS_REPO_ROOT, 'checkpoints'))  # checkpoint folder (README, Checkpoints)
MODELS = ['maresunet', 'utanet', 'mk_unet', 'mhorunet', 'tinyunet', 'ucmnet', 'lbunet', 'cmunext', 'egeunet', 'malunet', 'pbe']
LAYERS = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.Linear)
BNS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

def find_pairs(ad, model, x):
    """hook-trace: returns list of (layer_name, bn_name) with BN input is conv output (object identity), both called once."""
    names = {m: n for n, m in model.named_modules()}
    calls = {}; outs = {}; pairs = []
    hs = []
    def lay_hook(m, i, o): calls[m] = calls.get(m, 0) + 1; outs[id(o)] = (m, o)
    def bn_hook(m, i):
        calls[m] = calls.get(m, 0) + 1
        t = i[0]
        if id(t) in outs and outs[id(t)][1] is t: pairs.append((outs[id(t)][0], m))
    for m in model.modules():
        if isinstance(m, LAYERS): hs.append(m.register_forward_hook(lay_hook))
        if isinstance(m, BNS): hs.append(m.register_forward_pre_hook(bn_hook))
    with torch.no_grad(): ad.infer(model, x)
    for h in hs: h.remove()
    res = []; used = {}
    for l, b in pairs:
        if calls.get(l, 0) != 1 or calls.get(b, 0) != 1: continue
        if not (b.track_running_stats and b.running_mean is not None): continue
        used[l] = used.get(l, 0) + 1
        res.append((l, b))
    # a layer feeding two BNs is skipped
    res = [(l, b) for l, b in res if used[l] == 1]
    return res, names

def fold_into(l, b):
    std = (b.running_var + b.eps).sqrt()
    t = (b.weight / std) if b.weight is not None else 1.0 / std
    sh = (b.bias if b.bias is not None else torch.zeros_like(b.running_mean)) - b.running_mean * t
    t = t.detach(); sh = sh.detach()
    if isinstance(l, nn.ConvTranspose2d):
        if l.groups != 1: return False
        l.weight.data = l.weight.data * t.reshape(1, -1, 1, 1)
    elif isinstance(l, nn.Linear):
        l.weight.data = l.weight.data * t.reshape(-1, 1)
    else:
        l.weight.data = l.weight.data * t.reshape(-1, *([1] * (l.weight.dim() - 1)))
    nb = sh + (l.bias.data * t if l.bias is not None else 0)
    if l.bias is None: l.bias = nn.Parameter(nb.clone())
    else: l.bias.data = nb.clone()
    return True

def set_child(model, name, new):
    parts = name.split('.'); p = model
    for q in parts[:-1]: p = getattr(p, q)
    setattr(p, parts[-1], new)

def generic_fold(ad, model, x, skip=()):
    model = copy.deepcopy(model).eval()
    pairs, names = find_pairs(ad, model, x)
    n = 0
    for l, b in pairs:
        if names[b] in skip: continue
        if fold_into(l, b):
            set_child(model, names[b], nn.Identity()); n += 1
    return model, n, [(names[l], names[b]) for l, b in pairs]

def n_bn(m): return sum(isinstance(x, BNS) for x in m.modules())
def n_params(ad, m): return sum(p.numel() for n, p in m.named_parameters() if not ad.is_train_only(n))

class W(nn.Module):
    def __init__(s, ad, m): super().__init__(); s.ad = ad; s.m = m
    def forward(s, x): return s.ad.infer(s.m, x)

def thop_g(ad, m):
    from thop import profile
    with torch.no_grad(): return profile(W(ad, m).eval(), inputs=(torch.zeros(1, 3, 256, 256),), verbose=False)[0] * 2 / 1e9

def load(ad, name, ds):
    m = ad.build(copy.deepcopy(ad.DEFAULT_CFG))
    if name == 'pbe': ck = os.path.join(LN, 'pbe_reference_k1', ds, 'checkpoint.pth')
    else: ck = os.path.join(LN, 'baselines_k1', name, ds, 'k1', 'checkpoint.pth')
    sd = torch.load(ck, map_location='cpu', weights_only=False)
    m.load_state_dict(sd, strict=True); return m.eval(), ck

if __name__ == '__main__':
    only = sys.argv[1:] or MODELS
    DS = 'PH2'
    f = sorted(glob.glob(os.path.join(LN, 'data', 'cache', 'v1', DS, 'test_1_*.npz')))[0]
    x32 = torch.from_numpy(np.load(f)['images'][:4]).float() / 255.0
    out = {}
    for name in only:
        ad = registry.get(name)
        model, ck = load(ad, name, DS)
        r = dict(ckpt=ck, params_unfused=n_params(ad, model), bn_unfused=n_bn(model))
        if name == 'pbe':
            sys.path.insert(0, HERE)
            import f2_fuse
            fused = f2_fuse.fuse_pbe(model); r['n_folded_generic'] = 'RepConv merge + 33 BN fold (f2_fuse.fuse_pbe)'
            m64 = copy.deepcopy(model).double(); fused64 = f2_fuse.fuse_pbe(m64)
        else:
            fused, nf, pairs = generic_fold(ad, model, x32[:1])
            r['n_folded'] = nf
            m64 = copy.deepcopy(model).double(); fused64, _, _ = generic_fold(ad, m64, x32[:1].double())
        with torch.no_grad():
            a = ad.infer(model, x32); b = ad.infer(fused, x32)
            a64 = ad.infer(m64, x32.double()); b64 = ad.infer(fused64, x32.double())
        d = (a - b).abs().max().item(); sc = a.abs().max().item()
        r.update(max_abs_diff_fp32=d, logit_absmax=sc, rel_diff_fp32=d / max(sc, 1e-12), sign_agree=((a > 0) == (b > 0)).float().mean().item(),
                 max_abs_diff_fp64=(a64 - b64).abs().max().item())
        r.update(params_fused=n_params(ad, fused), bn_fused=n_bn(fused), gflops_unfused=thop_g(ad, copy.deepcopy(model)), gflops_fused=thop_g(ad, copy.deepcopy(fused)))
        out[name] = r
        print(name, {k: v for k, v in r.items() if k != 'ckpt'}, flush=True)
        json.dump(out, open(os.path.join(HERE, 'f2b_fold_out%s.json' % ('' if not sys.argv[1:] else '_part')), 'w'), indent=1)
