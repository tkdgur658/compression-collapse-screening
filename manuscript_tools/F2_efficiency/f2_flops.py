"""F2: FLOPs (thop and ptflops, MACs x 2) at 1x3x256x256 for the 11 inference graphs, plus parameter counts.
Convention of the legacy table (legacy_src/benchmark_efficiency_all.py count_flops): ptflops if importable, else thop; GFLOPs = MACs*2/1e9.
Inference graph = adapter.infer (training-only heads/side outputs are not executed)."""
import os as _ccs_os  # path configuration of the released copy (README, Configuration)
CCS_REPO_ROOT = _ccs_os.path.normpath(_ccs_os.path.join(_ccs_os.path.dirname(_ccs_os.path.abspath(__file__)), '../..'))
CCS_WORK_ROOT = _ccs_os.environ.get('CCS_WORK', _ccs_os.path.join(CCS_REPO_ROOT, 'experiments'))
import sys, os, json, copy, importlib
import torch
torch.set_num_threads(2)
CODE = _ccs_os.path.join(CCS_WORK_ROOT, 'qfix', 'A', 'code')
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '_pylib'))
from models import registry
MODELS = ['pbe', 'maresunet', 'utanet', 'mk_unet', 'mhorunet', 'tinyunet', 'ucmnet', 'lbunet', 'cmunext', 'egeunet', 'malunet']

class InferWrap(torch.nn.Module):
    def __init__(self, ad, m):
        super().__init__(); self.ad = ad; self.m = m
    def forward(self, x):
        return self.ad.infer(self.m, x)

def n_params(ad, m):
    return sum(p.numel() for n, p in m.named_parameters() if not ad.is_train_only(n))

def thop_count(w, x):
    from thop import profile
    macs, p = profile(w, inputs=(x,), verbose=False)
    return macs * 2 / 1e9

def ptflops_count(w, shape):
    from ptflops import get_model_complexity_info
    macs, p = get_model_complexity_info(w, shape, as_strings=False, print_per_layer_stat=False, verbose=False)
    return macs * 2 / 1e9

if __name__ == '__main__':
    only = sys.argv[1:] or MODELS
    try:
        import ptflops; have_pt = True
    except Exception:
        have_pt = False
    out = {}
    x = torch.zeros(1, 3, 256, 256)
    for name in only:
        ad = registry.get(name)
        fresh = lambda: InferWrap(ad, ad.build(copy.deepcopy(ad.DEFAULT_CFG)).eval()).eval()
        m = ad.build(copy.deepcopy(ad.DEFAULT_CFG)).eval()
        r = {'params_inference': n_params(ad, m)}
        with torch.no_grad():
            try: r['gflops_thop'] = thop_count(fresh(), x)
            except Exception as e: r['gflops_thop'] = 'ERR ' + repr(e)[:100]
            if have_pt:
                try: r['gflops_ptflops'] = ptflops_count(fresh(), (3, 256, 256))
                except Exception as e: r['gflops_ptflops'] = 'ERR ' + repr(e)[:100]
        out[name] = r
        print(name, r, flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'f2_flops_out%s.json' % ('_ptflops' if have_pt else '_thop')), 'w'), indent=1)
