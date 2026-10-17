"""Compare two ident.py outputs bitwise. usage: ident_cmp.py ORIG.pt SNAP.pt  (exit 1 on any difference; prints no metric value)"""
import sys

import torch

a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])
errs = []
if list(a['state_dict']) != list(b['state_dict']):
    errs.append('state_dict keys differ')
for k in a['state_dict']:
    if k in b['state_dict'] and not torch.equal(a['state_dict'][k], b['state_dict'][k]):
        errs.append(f'state {k}')
for k in ('out', 'loss', 'ckpt_logits'):
    if not torch.equal(a[k], b[k]):
        errs.append(k)
if list(a['grads']) != list(b['grads']) or any(not torch.equal(a['grads'][k], b['grads'][k]) for k in a['grads']):
    errs.append('grads')
if 'val' in a and 'val' in b:
    dd = abs(a['val']['dice'] - b['val']['dice'])
    di = abs(a['val']['iou'] - b['val']['iou'])
    print(f'[ident] val |dDice|={dd:.2e} |dIoU|={di:.2e} (limit 1e-6)')
    if dd > 1e-6 or di > 1e-6:
        errs.append('val dice/iou')
print('[ident] state_dict tensors', len(a['state_dict']), 'grads', len(a['grads']),
      'RESULT', 'BITWISE-IDENTICAL' if not errs else f'DIFF {errs}')
sys.exit(1 if errs else 0)
