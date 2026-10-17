"""F2: same equivalence check in float64, fusion also computed in float64 (isolates float32 rounding from fusion error)."""
import sys, os, glob, json, copy
sys.argv = sys.argv[:1]
import numpy as np, torch
from f2_fuse import *
rows = []
for ds in ['PH2', 'cvc', 'etis', 'isic17', 'isic18', 'kvasir']:
    ck = os.path.join(LN, 'pbe_reference_k1', ds, 'checkpoint.pth')
    sd = torch.load(ck, map_location='cpu', weights_only=False)
    model = ad.build(copy.deepcopy(ad.DEFAULT_CFG)); model.load_state_dict(sd, strict=True); model.eval()
    model.double(); fused = fuse_pbe(model)  # fuse in float64 so fusion arithmetic is not float32-limited
    fused.double()
    f = sorted(glob.glob(os.path.join(LN, 'data', 'cache', 'v1', ds, 'test_1_*.npz')))[0]
    x = torch.from_numpy(np.load(f)['images'][:6]).double() / 255.0
    with torch.no_grad(): a = ad.infer(model, x); b = ad.infer(fused, x)
    rows.append(dict(ds=ds, max_abs_diff_fp64=(a - b).abs().max().item())); print(rows[-1])
json.dump(rows, open(os.path.join(HERE, 'f2_fuse_fp64_out.json'), 'w'), indent=1)
