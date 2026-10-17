"""PREREG_JETSON step 1: fused deployment graph of the 11 models -> ONNX opset 11 (1x3x256x256, batch 1, adapter.infer output).
Runs on gpu-server, CPU only (CUDA_VISIBLE_DEVICES="", 2 threads). Fusion = F2b code imported unchanged
(f2b_fold_all.generic_fold for baselines, f2_fuse.fuse_pbe for PBE-Net), same checkpoints (PH2 k1) and the same
4 test images (first 4 of PH2 test_1_*.npz) as F2b; equivalence fp32 and fp64 exactly as in f2b_fold_all.__main__.
Also writes, per model, the fused fp32 output on image 0 (reference for the TensorRT engine output check) and the input
as raw float32 (trtexec --loadInputs)."""
import sys, os, json, copy, glob, hashlib, time, traceback
BASE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(BASE, 'code', 'pbe_seg')
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(BASE, 'fusion'))
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(2)
import f2b_fold_all as F  # unchanged F2b code (sha256 9ad43a00...)
import f2_fuse            # unchanged F2 code (sha256 a1ef95d3...)
torch.set_num_threads(2)
from models import registry

# The python package `onnx` is not installed in any environment on gpu-server or on the local PC (no network install allowed).
# torch 2.0.1 calls onnx_proto_utils._add_onnxscript_fn after the C++ serializer has produced the ModelProto bytes; this step
# only imports `onnx` to insert onnx-script FunctionProtos for nodes of a custom (non aten/prim/onnx) domain and returns
# model_bytes unchanged otherwise (torch/onnx/_internal/onnx_proto_utils.py lines 212-244). Replaced in this process by the
# identity; the graph is checked below to contain only the default ONNX domain, so the file is the same as with `onnx` installed.
import torch.onnx._internal.onnx_proto_utils as _opu
def _identity_add_onnxscript_fn(model_bytes, custom_opsets):
    return model_bytes
_opu._add_onnxscript_fn = _identity_add_onnxscript_fn

MODELS = ['pbe', 'maresunet', 'utanet', 'mk_unet', 'mhorunet', 'tinyunet', 'ucmnet', 'lbunet', 'cmunext', 'egeunet', 'malunet']
OUT = os.path.join(BASE, 'onnx_op11')  # run 1 wrote to BASE/onnx, whose name shadowed the onnx package (export failed); kept as record


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def load(ad, name):
    m = ad.build(copy.deepcopy(ad.DEFAULT_CFG))
    ck = os.path.join(BASE, 'ckpt', name, 'checkpoint.pth')
    sd = torch.load(ck, map_location='cpu')
    m.load_state_dict(sd, strict=True)
    return m.eval(), ck


class W(nn.Module):
    def __init__(s, ad, m): super().__init__(); s.ad = ad; s.m = m
    def forward(s, x): return s.ad.infer(s.m, x)


if __name__ == '__main__':
    only = sys.argv[1:] or MODELS
    f = sorted(glob.glob(os.path.join(BASE, 'data', 'test_1_*.npz')))[0]
    x32 = torch.from_numpy(np.load(f)['images'][:4]).float() / 255.0
    x0 = x32[:1].contiguous()
    x0.numpy().astype(np.float32).tofile(os.path.join(OUT, 'input_ph2_test1_img0.bin'))
    res_p = os.path.join(OUT, 'export_out.json')
    out = json.load(open(res_p)) if os.path.exists(res_p) else {}
    out['_env'] = dict(torch=torch.__version__, python=sys.version.split()[0], threads=torch.get_num_threads(),
                       cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'), npz=os.path.basename(f), npz_sha256=sha(f),
                       f2b_fold_all_sha256=sha(F.__file__), f2_fuse_sha256=sha(f2_fuse.__file__), input_shape=list(x0.shape))
    for name in only:
        t0 = time.time()
        ad = registry.get(name)
        model, ck = load(ad, name)
        r = dict(ckpt_sha256=sha(ck))
        if name == 'pbe':
            fused = f2_fuse.fuse_pbe(model); r['fusion'] = 'f2_fuse.fuse_pbe (RepConv merge + BN fold)'
            m64 = copy.deepcopy(model).double(); fused64 = f2_fuse.fuse_pbe(m64)
        else:
            fused, nf, pairs = F.generic_fold(ad, model, x32[:1]); r['fusion'] = 'f2b_fold_all.generic_fold'; r['n_folded'] = nf
            m64 = copy.deepcopy(model).double(); fused64, _, _ = F.generic_fold(ad, m64, x32[:1].double())
        r['bn_unfused'] = F.n_bn(model); r['bn_fused'] = F.n_bn(fused)
        r['params_fused'] = F.n_params(ad, fused)
        with torch.no_grad():
            a = ad.infer(model, x32); b = ad.infer(fused, x32)
            a64 = ad.infer(m64, x32.double()); b64 = ad.infer(fused64, x32.double())
            assert torch.is_tensor(b), type(b)
            d = (a - b).abs().max().item(); sc = a.abs().max().item()
            r.update(max_abs_diff_fp32=d, logit_absmax=sc, rel_diff_fp32=d / max(sc, 1e-12),
                     sign_agree=((a > 0) == (b > 0)).float().mean().item(), max_abs_diff_fp64=(a64 - b64).abs().max().item())
            ref0 = ad.infer(fused, x0)
        r['output_shape'] = list(ref0.shape)
        np.save(os.path.join(OUT, '%s_ref_out_img0.npy' % name), ref0.numpy().astype(np.float32))
        onnx_p = os.path.join(OUT, '%s_fused_op11.onnx' % name)
        try:
            torch.onnx.export(W(ad, fused).eval(), (x0,), onnx_p, opset_version=11, input_names=['input'],
                              output_names=['output'], do_constant_folding=True, export_params=True)
            r['onnx'] = os.path.basename(onnx_p); r['onnx_sha256'] = sha(onnx_p); r['onnx_bytes'] = os.path.getsize(onnx_p)
            r['export_error'] = None
            # domain check on the C++ pretty print of the same export (no python onnx needed)
            pretty = torch.onnx.export_to_pretty_string(W(ad, fused).eval(), (x0,), opset_version=11, input_names=['input'],
                                                       output_names=['output'], do_constant_folding=True)
            open(os.path.join(OUT, '%s_fused_op11.pretty.txt' % name), 'w').write(pretty)
            import re
            doms = sorted(set(re.findall(r'domain:\s*"?([^\s",}]*)', pretty)))
            types = sorted(set(re.findall(r'type:\s*"?([A-Za-z0-9_]+)', pretty)))
            r['pretty_domains'] = doms; r['op_types'] = types
            r['opset_import_line'] = [l.strip() for l in pretty.splitlines() if 'opset_import' in l][:3]
        except Exception as e:
            r['onnx'] = None; r['export_error'] = '%s: %s' % (type(e).__name__, str(e).splitlines()[0][:400])
            r['export_traceback_tail'] = traceback.format_exc().splitlines()[-6:]
        r['wall_s'] = round(time.time() - t0, 1)
        out[name] = r
        print(name, {k: v for k, v in r.items() if k not in ('export_traceback_tail',)}, flush=True)
        json.dump(out, open(res_p, 'w'), indent=1)
