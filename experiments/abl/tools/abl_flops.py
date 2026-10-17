"""F4 V3 width decision: thop MACs x 2 (F2 convention) of the stage-1/2 feature block (EBPB) vs InvResBlock (mbv2) at the
resolution where it runs, and whole-network inference GFLOPs (adapter.infer graph) for block=ebpb vs mbv2.
usage: python abl_flops.py NET_PY_DIR   (directory holding net.py of e0e2f07)"""
import sys, json, importlib.util, torch
torch.set_num_threads(2)
from thop import profile
spec = importlib.util.spec_from_file_location('net', sys.argv[1] + '/net.py'); net = importlib.util.module_from_spec(spec); spec.loader.exec_module(net)
def g(m, x):
    m.eval()
    with torch.no_grad():
        macs, _ = profile(m, inputs=(x,), verbose=False)
    return macs * 2 / 1e9
out = {}
# stage 1 block input: C=128 at 256x256 (before max-pool); stage 2: C=192 at 128x128
for st, C, H in ((1, 128, 256), (2, 192, 128)):
    x = torch.zeros(1, C, H, H)
    out[f's{st}_ebpb'] = g(net.EBPB(C), x)
    out[f's{st}_mbv2_t0.5'] = g(net.InvResBlock(C), x)
    out[f's{st}_mbv2_t_min(h=8)'] = g(net.InvResBlock(C, t=1e-9), x)
    for h in (2, 4, 6, 8):
        b = net.InvResBlock(C); b.b = torch.nn.Sequential(torch.nn.Conv2d(C, h, 1, bias=False), torch.nn.BatchNorm2d(h), torch.nn.ReLU6(True),
            torch.nn.Conv2d(h, h, 3, padding=1, groups=h, bias=False), torch.nn.BatchNorm2d(h), torch.nn.ReLU6(True), torch.nn.Conv2d(h, C, 1, bias=False), torch.nn.BatchNorm2d(C))
        out[f's{st}_mbv2_h{h}(unclamped,hypothetical)'] = g(b, x)
x = torch.zeros(1, 3, 256, 256)
for blk in ('ebpb', 'mbv2'):
    m = net.PBENet(block=blk)
    out[f'net_{blk}_gflops'] = g(m, x)
    out[f'net_{blk}_params_inference'] = sum(p.numel() for n, p in m.named_parameters() if not n.startswith(('base.gt_conv', 'shead')))
out['blocks_ebpb_sum'] = out['s1_ebpb'] + out['s2_ebpb']; out['blocks_mbv2_sum'] = out['s1_mbv2_t0.5'] + out['s2_mbv2_t0.5']
out['blocks_mbv2_min_sum'] = out['s1_mbv2_t_min(h=8)'] + out['s2_mbv2_t_min(h=8)']
print(json.dumps(out, indent=1))
