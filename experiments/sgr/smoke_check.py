# Summarise the CPU smoke records (integrity flags only; smoke computes no Dice).
import os as _ccs_os  # path configuration of the released copy (README, Configuration)
CCS_REPO_ROOT = _ccs_os.path.normpath(_ccs_os.path.join(_ccs_os.path.dirname(_ccs_os.path.abspath(__file__)), '../..'))
CCS_WORK_ROOT = _ccs_os.environ.get('CCS_WORK', _ccs_os.path.join(CCS_REPO_ROOT, 'experiments'))
import json, glob, os, sys
B = _ccs_os.path.join(CCS_WORK_ROOT, 'sgr/') + (sys.argv[1] if len(sys.argv) > 1 else 'smoke')
bad, n = [], 0
for p in sorted(glob.glob(B + '/*.json')):
    r = json.load(open(p)); n += 1
    if not r['integrity']['state_restored_exact']: bad.append((p, 'restore'))
    for a, v in r['arms'].items():
        if v.get('finite_smoke') is not True: bad.append((p, a, 'finite'))
        t = v.get('tune') or {}; b3 = v.get('b3') or {}
        for k in ('mask_preserved', 'levels_ok'):
            if t.get(k) is False or b3.get(k) is False: bad.append((p, a, k))
        if v.get('exact_count') is False: bad.append((p, a, 'exact_count'))
        if t and t.get('steps_done') != 2: bad.append((p, a, 'steps', t.get('steps_done')))
        if 'global_sparsity' in v and abs(v['global_sparsity'] - 0.5) > 0.001: bad.append((p, a, 'sparsity', v['global_sparsity']))
print('SMOKE_CHECK records=%d problems=%d %s' % (n, len(bad), bad[:10]))
