"""F2: fuse PBE-Net for deployment (RepConv branches merged, every Conv+BN folded) on a CPU copy of a v1 checkpoint,
check output equivalence on cached 256x256 test images, count inference parameters and thop GFLOPs (MACs*2)."""
import os as _ccs_os  # path configuration of the released copy (README, Configuration)
CCS_REPO_ROOT = _ccs_os.path.normpath(_ccs_os.path.join(_ccs_os.path.dirname(_ccs_os.path.abspath(__file__)), '../..'))
CCS_WORK_ROOT = _ccs_os.environ.get('CCS_WORK', _ccs_os.path.join(CCS_REPO_ROOT, 'experiments'))
import sys, os, json, copy, glob
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ccs_os.path.join(CCS_WORK_ROOT, 'qfix', 'A', 'code')); sys.path.insert(0, os.path.join(HERE, '_pylib'))
from models.pbe import adapter as ad
from models.pbe.src.net import RepConv
LN = os.environ.get('CCS_CHECKPOINTS', os.path.join(CCS_REPO_ROOT, 'checkpoints'))  # checkpoint folder (README, Checkpoints)

def fold(conv, bn):
    """Conv (no bias) + BN (eval) -> (weight, bias)."""
    std = (bn.running_var + bn.eps).sqrt()
    t = (bn.weight / std)
    w = conv.weight * t.reshape(-1, 1, 1, 1)
    b = bn.bias - bn.running_mean * t
    if conv.bias is not None: b = b + conv.bias * t
    return w.detach(), b.detach()

def fuse_conv_bn(conv, bn):
    w, b = fold(conv, bn)
    n = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, conv.dilation, conv.groups, bias=True, dtype=conv.weight.dtype)
    n.weight.data.copy_(w); n.bias.data.copy_(b)
    return n

def fuse_rep(rc):
    k = rc.conv_kxk.kernel_size[0]; C = rc.conv_kxk.out_channels
    w3, b3 = fold(rc.conv_kxk, rc.bn_kxk)
    w1, b1 = fold(rc.conv_1x1, rc.bn_1x1)
    w1 = nn.functional.pad(w1, [k // 2] * 4)
    std = (rc.bn_identity.running_var + rc.bn_identity.eps).sqrt(); t = rc.bn_identity.weight / std
    wi = torch.zeros(C, 1, k, k, dtype=rc.conv_kxk.weight.dtype); wi[:, 0, k // 2, k // 2] = 1.0
    wi = wi * t.reshape(-1, 1, 1, 1).detach(); bi = (rc.bn_identity.bias - rc.bn_identity.running_mean * t).detach()
    conv = nn.Conv2d(C, C, k, 1, k // 2, groups=C, bias=True, dtype=rc.conv_kxk.weight.dtype)
    conv.weight.data.copy_(w3 + w1 + wi); conv.bias.data.copy_(b3 + b1 + bi)
    return nn.Sequential(conv, nn.ReLU(inplace=True))

def fuse_pbe(model):
    m = copy.deepcopy(model).eval()
    def rec(mod):
        for name, ch in list(mod.named_children()):
            if isinstance(ch, RepConv): setattr(mod, name, fuse_rep(ch))
            else: rec(ch)
    rec(m)
    # Conv+BN pairs: EBPB.branch3 (Sequential: conv,bn,relu,conv,bn,relu) and DWSepBottleneck (dw,bn1,pw,bn2)
    for e in (m.base.encoder1[0][2], m.base.encoder2[0][2]):
        s = e.branch3
        e.branch3 = nn.Sequential(fuse_conv_bn(s[0], s[1]), nn.ReLU(inplace=True), fuse_conv_bn(s[3], s[4]), nn.ReLU(inplace=True))
    b = m.base.bottleneck
    b.dw = fuse_conv_bn(b.dw, b.bn1); b.pw = fuse_conv_bn(b.pw, b.bn2)
    b.bn1 = nn.Identity(); b.bn2 = nn.Identity()
    return m

def n_inf(m):
    return sum(p.numel() for n, p in m.named_parameters() if not ad.is_train_only(n))

def n_bn(m): return sum(isinstance(x, nn.BatchNorm2d) for x in m.modules())

if __name__ == '__main__':
    res = {}
    cfg = copy.deepcopy(ad.DEFAULT_CFG)
    rows = []
    for ds in ['PH2', 'cvc', 'etis', 'isic17', 'isic18', 'kvasir']:
        ck = os.path.join(LN, 'pbe_reference_k1', ds, 'checkpoint.pth')
        sd = torch.load(ck, map_location='cpu', weights_only=False)
        model = ad.build(cfg); model.load_state_dict(sd, strict=True); model.eval()
        fused = fuse_pbe(model)
        f = sorted(glob.glob(os.path.join(LN, 'data', 'cache', 'v1', ds, 'test_1_*.npz')))
        f = [x for x in f if x.endswith('.npz')][0]
        d = np.load(f); x = torch.from_numpy(d['images'][:6]).float() / 255.0
        with torch.no_grad():
            a = ad.infer(model, x); b = ad.infer(fused, x)
        diff = (a - b).abs().max().item()
        agree = ((a > 0) == (b > 0)).float().mean().item()
        rows.append(dict(ds=ds, n_img=int(x.shape[0]), max_abs_diff=diff, sign_agree=agree, logit_absmax=a.abs().max().item()))
        print(rows[-1], flush=True)
        if ds == 'PH2':
            res['params_inference_unfused'] = n_inf(model); res['params_inference_fused'] = n_inf(fused)
            res['params_total_unfused'] = sum(p.numel() for p in model.parameters()); res['params_total_fused'] = sum(p.numel() for p in fused.parameters())
            res['n_bn_unfused'] = n_bn(model); res['n_bn_fused'] = n_bn(fused)
            from thop import profile
            xx = torch.zeros(1, 3, 256, 256)
            class W(nn.Module):
                def __init__(s, m): super().__init__(); s.m = m
                def forward(s, x): return s.m(x)
            with torch.no_grad():
                res['gflops_thop_fused'] = profile(W(fuse_pbe(model)), inputs=(xx,), verbose=False)[0] * 2 / 1e9
                m2 = ad.build(cfg); m2.load_state_dict(sd); m2.eval()
                res['gflops_thop_unfused'] = profile(W(m2), inputs=(xx,), verbose=False)[0] * 2 / 1e9
    res['equivalence'] = rows
    res['max_abs_diff_all'] = max(r['max_abs_diff'] for r in rows)
    json.dump(res, open(os.path.join(HERE, 'f2_fuse_out.json'), 'w'), indent=1)
    print({k: v for k, v in res.items() if k != 'equivalence'})
