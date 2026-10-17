import os as _ccs_os  # path configuration of the released copy (README, Configuration)
CCS_REPO_ROOT = _ccs_os.path.normpath(_ccs_os.path.join(_ccs_os.path.dirname(_ccs_os.path.abspath(__file__)), '../../..'))
CCS_WORK_ROOT = _ccs_os.environ.get('CCS_WORK', _ccs_os.path.join(CCS_REPO_ROOT, 'experiments'))
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)


rep("BASE = _ccs_os.path.join(CCS_WORK_ROOT, 'qdeploy')", "BASE = _ccs_os.path.join(CCS_WORK_ROOT, 'qdeploy_w8a8')   # W8A8 addendum (PREREG_QDEPLOY_ADDENDUM_W8A8.md)")
rep("GPU_CAP_S = 6.0 * 3600.0", "GPU_CAP_S = 1.0 * 3600.0")
rep("SETTINGS = ['O_w4', 'O_a6', 'O_l50', 'D_w4', 'D_a6', 'D_l50', 'S_w4g', 'S_w4c', 'S_a6c', 'S_l50c']",
    "SETTINGS = ['D_w8a8']")
rep("        'S_w4g': ('dep', 'w4_pt'), 'S_w4c': ('train', 'w4_pc'), 'S_a6c': ('dep', 'a6c'), 'S_l50c': ('dep', 'l50c')}",
    "        'S_w4g': ('dep', 'w4_pt'), 'S_w4c': ('train', 'w4_pc'), 'S_a6c': ('dep', 'a6c'), 'S_l50c': ('dep', 'l50c'),\n"
    "        'D_w8a8': ('dep', 'w8a8')}")
rep("""            elif op in ('l50', 'l50c'):""",
    """            elif op == 'w8a8':
                mx = 0
                for n, q in self.targets:
                    w, lv = rtn_sym_channel(torch, self.orig[n], 8, self.axis[n])
                    q.copy_(w)
                    mx = max(mx, int(lv.abs().max()))
                assert mx <= 127
                for n, m in self.leaf:
                    hs.append(m.register_forward_hook(qa.act_hook_factory(torch, 8)))
                integ = {'n_tensors': len(self.targets), 'max_abs_level': mx, 'n_hooked': len(self.leaf),
                         'n_hooked_executed': sum(1 for _, m in self.leaf if id(m) in self.executed)}
            elif op in ('l50', 'l50c'):""")
rep("""            ints = {s: rec['cells'][s]['integrity'] for s in ('D_w4', 'D_l50')}
            print('[%s] %s %.1fs bs=%d E1=%s(%.1e) folded=%s bn %d->%d restored=%s D_w4 n=%d D_l50 minzero=%.3f used=%.0fs' % (
                time.strftime('%T'), tag, rec['gpu_wall_s'], bs, rec['equiv']['E1_pass'], rec['equiv']['E1_max_abs_fp64'] if rec['equiv']['E1_max_abs_fp64'] is not None else -1.0, rec['n_folded'],
                rec['norm_train']['bn'], rec['norm_dep']['bn'], all(rec['state_restored_exact'].values()),
                ints['D_w4']['n_tensors'], ints['D_l50']['min_zero_frac'] or -1, used), flush=True)""",
    """            iw = rec['cells']['D_w8a8']['integrity']
            print('[%s] %s %.1fs bs=%d E1=%s(%.1e) folded=%s bn %d->%d restored=%s w8a8 n=%d maxlev=%d hooked=%d used=%.0fs' % (
                time.strftime('%T'), tag, rec['gpu_wall_s'], bs, rec['equiv']['E1_pass'], rec['equiv']['E1_max_abs_fp64'] if rec['equiv']['E1_max_abs_fp64'] is not None else -1.0, rec['n_folded'],
                rec['norm_train']['bn'], rec['norm_dep']['bn'], all(rec['state_restored_exact'].values()),
                iw['n_tensors'], iw['max_abs_level'], iw['n_hooked_executed'], used), flush=True)""")
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
