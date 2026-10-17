"""Check that the clean PBE-Net reproduces the legacy implementation.

Needs the legacy repository on the import path (read-only). Checks, on CPU in float64:
  1. state_dict keys and shapes are identical (legacy Phase57_LapSkipSup + DWSep bottleneck);
  2. a legacy checkpoint loads strictly into the new model;
  3. eval forward, train forward (main output, 5 side outputs, 3 skip heads) are identical;
  4. matching loss and BCE+Dice deep-supervision loss equal the legacy functions;
  5. plain/mbv2 variants match the legacy agent_swap_models SkipPlainAdditive (+ mbv2 swap).
Writes a JSON report.

  python tools/verify_legacy.py --legacy <LEGACY_REPO> \
      --ckpt <legacy checkpoint.pth> --out verify_legacy.json
"""
import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def max_abs(a, b):
    return float((a - b).abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--legacy', required=True); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default='verify_legacy.json')
    a = ap.parse_args()

    # Both repositories have a top-level `models` package: import the legacy one normally and
    # load the new files by path under private names.
    import importlib.util
    sys.path.insert(0, a.legacy)
    os.environ.setdefault('PROJECT_ROOT', a.legacy)
    import agent_swap_models as SW
    from utils import GT_BceDiceLoss

    def by_path(name, rel):
        spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
        mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
        return mod
    PBENet = by_path('_new_net', 'models/pbe/src/net.py').PBENet
    matching_loss = by_path('_new_matching', 'models/pbe/src/matching.py').matching_loss
    seg_loss = by_path('_new_losses', 'common/losses.py').seg_loss

    rep = {'legacy_repo': a.legacy, 'ckpt': a.ckpt, 'checks': {}}
    torch.set_default_dtype(torch.float64)

    def legacy_full():
        m = SW.SkipPredictive(num_classes=1, c_list=[128, 192, 256, 384, 512])
        class DWSep(torch.nn.Module):  # identical to agent_swap_train.py
            def __init__(s, ch):
                super().__init__(); nn = torch.nn
                s.dw = nn.Conv2d(ch, ch, 7, padding=3, groups=ch, bias=False); s.bn1 = nn.BatchNorm2d(ch)
                s.pw = nn.Conv2d(ch, ch, 1, bias=False); s.bn2 = nn.BatchNorm2d(ch); s.act = nn.ReLU(inplace=True)
            def forward(s, x): return s.act(s.bn2(s.pw(s.act(s.bn1(s.dw(x))))))
        m.base.bottleneck = DWSep(512)
        return m, DWSep

    old, DWSep = legacy_full()
    new = PBENet()
    ko = {k: tuple(v.shape) for k, v in old.state_dict().items()}
    kn = {k: tuple(v.shape) for k, v in new.state_dict().items()}
    rep['checks']['state_dict_identical'] = ko == kn
    rep['checks']['n_keys'] = [len(ko), len(kn)]

    sd = torch.load(a.ckpt, map_location='cpu')
    sd = {k: v.double() if v.is_floating_point() else v for k, v in sd.items()}
    old.load_state_dict(sd, strict=True); new.load_state_dict(sd, strict=True)
    old.double(); new.double()

    x = torch.rand(2, 3, 256, 256)
    mask = (torch.rand(2, 1, 256, 256) > 0.7).double()
    old.eval(); new.eval()
    with torch.no_grad():
        rep['checks']['eval_out_maxabs'] = max_abs(old(x), new(x))
    old.train(); new.train()
    torch.manual_seed(0); go, oo, sko = old(x)
    torch.manual_seed(0); on, gn, skn = new(x, return_aux=True)
    rep['checks']['train_out_maxabs'] = max_abs(oo, on)
    rep['checks']['train_side_maxabs'] = max(max_abs(p, q) for p, q in zip(go, gn))
    rep['checks']['train_heads_maxabs'] = max(max_abs(p, q) for p, q in zip(sko, skn))

    # losses: legacy trainer applies sigmoid+clamp then GT_BceDiceLoss; cos_band on sigmoid(head)
    clamp = lambda t: torch.clamp(torch.sigmoid(t), 1e-7, 1 - 1e-7)
    l_old = GT_BceDiceLoss(1, 1)(tuple(clamp(p) for p in go), clamp(oo), mask)
    l_new = seg_loss(on, gn, mask)
    rep['checks']['seg_loss'] = [float(l_old), float(l_new)]

    def legacy_cos(sk, msk):  # copied from agent_swap_train.py (gt_boundary + cos_band + sup_loss)
        import torch.nn.functional as F
        m = (msk > 0.5).float()
        bnd = (F.max_pool2d(m, 3, 1, 1) + F.max_pool2d(-m, 3, 1, 1)).clamp(0, 1)
        tot = 0.0
        for si in sk:
            p = torch.sigmoid(si); b = F.interpolate(bnd, size=p.shape[2:], mode='nearest')
            pf = p.flatten(1) - p.flatten(1).mean(1, keepdim=True)
            bf = b.flatten(1) - b.flatten(1).mean(1, keepdim=True)
            tot = tot + (1.0 - (pf * bf).sum(1) / (pf.norm(dim=1) * bf.norm(dim=1) + 1e-8)).mean()
        return tot / len(sk)
    rep['checks']['match_loss'] = [float(legacy_cos(sko, mask)), float(matching_loss(skn, mask))]

    # ablation variants: architecture-only equivalence (fresh weights copied old -> new)
    for tag, blk in (('plain_ebpb', None), ('plain_mbv2', 'mbv2')):
        o = SW.SkipPlainAdditive(num_classes=1, c_list=[128, 192, 256, 384, 512]); o.base.bottleneck = DWSep(512)
        if blk == 'mbv2':
            SW._replace_twoband(o.base.encoder1, SW._make_mbv2); SW._replace_twoband(o.base.encoder2, SW._make_mbv2)
        n = PBENet(skip='plain', block=blk or 'ebpb')
        same_keys = {k: tuple(v.shape) for k, v in o.state_dict().items()} == {k: tuple(v.shape) for k, v in n.state_dict().items()}
        n.load_state_dict(o.state_dict(), strict=True); o.double().eval(); n.double().eval()
        with torch.no_grad():
            rep['checks'][f'{tag}_keys_identical'] = same_keys
            rep['checks'][f'{tag}_eval_maxabs'] = max_abs(o(x), n(x))

    c = rep['checks']
    ok = (c['state_dict_identical'] and c['eval_out_maxabs'] == 0 and c['train_out_maxabs'] == 0
          and c['train_side_maxabs'] == 0 and c['train_heads_maxabs'] == 0
          and abs(c['seg_loss'][0] - c['seg_loss'][1]) < 1e-12 and abs(c['match_loss'][0] - c['match_loss'][1]) < 1e-12
          and c['plain_ebpb_keys_identical'] and c['plain_mbv2_keys_identical']
          and c['plain_ebpb_eval_maxabs'] == 0 and c['plain_mbv2_eval_maxabs'] == 0)
    rep['pass'] = bool(ok)
    json.dump(rep, open(a.out, 'w'), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
