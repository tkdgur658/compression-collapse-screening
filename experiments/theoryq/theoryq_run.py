# -*- coding: utf-8 -*-
"""THEORY_Q predictor run (PREREG_THEORYQ.md sections 3, 7). gpu-server CPU only (CUDA_VISIBLE_DEVICES=""), 2 threads, nice 10.
Reads checkpoint weights (read-only, sha-checked against F3 MANIFEST_330) and the STRUCTURAL integrity fields of the F3
records; never reads images, Dice, R or any label field. Writes only under <WORK>/theoryq/out.

  --dryrun TAG,...  : build graphs + integrity + excluded-tensor counts + timing only (no predictor value computed/written)
  --run             : all 330 items, out/<tag>.json + DONE/FAIL markers, restart-safe; stops on the first integrity mismatch
"""
import os
import sys
import io
import json
import time
import math
import argparse
import traceback
import hashlib

sys.dont_write_bytecode = True
assert os.environ.get('CUDA_VISIBLE_DEVICES', None) == '', 'run with CUDA_VISIBLE_DEVICES=""'
HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, 'out')
sys.path.insert(0, HERE)
import torch  # noqa: E402
torch.set_num_threads(2)
import theoryq_core as C  # noqa: E402
QD, QA = C.QD, C.QA
sys.path.insert(1, QD.CODE_ROOT)
sys.path.insert(2, QD.BASE)
from models import registry  # noqa: E402
import f2_fuse  # noqa: E402
import f2b_fold_all as f2b  # noqa: E402
torch.set_num_threads(2)                     # f2_* set 2 at import
F3_FROZEN = {'f2_fuse.py': 'a1ef95d3a5c0efefbb9ef2d18023189bf077863357ab0c61ed36f8875c08706f',
             'f2b_fold_all.py': '9ad43a00f7c3a1e39109052b8eb60137eb2fb2801933967fe32c74577dfed074',
             'FOLD_PAIRS.json': '69e196dd1b41ea99bbf1f3e3401d2e560ef7b010177de3472c747957da7fc1d5',
             'MANIFEST_330.sha256': '58714f2f280ea80153cabff571c13efbc425700967adc2c48b0147fad510c2b6',
             'ITEMS.json': 'cc16bb780d77da85991ea5e76c77e373bb009f37cf6f0e88f6677aaabf672c61'}
for _f, _h in F3_FROZEN.items():
    assert QD.sha_file(os.path.join(QD.BASE, _f)) == _h, ('F3 frozen file mismatch', _f)
DEV = torch.device('cpu')
CAP_S = 3 * 3600.0


def x_fixed():
    return torch.rand(1, 3, 256, 256, generator=torch.Generator().manual_seed(QD.SEED))


def f3_record(tag):
    """structural fields only."""
    r = QD.jload('%s/%s.json' % (QD.OUT, tag))
    c = r['cells']
    return {'n_folded': r['n_folded'], 'params_inf_dep': r['params_inf_dep'], 'params_inf_train': r['params_inf_train'],
            'S_w4g': {k: c['S_w4g']['integrity'][k] for k in ('n_tensors', 'max_levels')},
            'O_w4': {k: c['O_w4']['integrity'][k] for k in ('n_tensors', 'max_levels')},
            'D_w4': {k: c['D_w4']['integrity'][k] for k in ('n_tensors', 'n_tensors_axis_size1', 'n_axis1_tensors', 'max_abs_level')},
            'D_l50': {k: c['D_l50']['integrity'][k] for k in ('n_layers', 'min_zero_frac')}}


def graph_targets(ad, g, used):
    """(all targets, kept targets, excluded unexecuted names, excluded train-only names)."""
    keep, unexec, tonly = [], [], []
    for n, q in g.targets:
        if ad.is_train_only(n):
            tonly.append(n)
        elif n not in used:
            unexec.append(n)
        else:
            keep.append((n, g.orig[n], g.axis[n]))
    return keep, unexec, tonly


def leaf_exec_names(g):
    return {n for n, m in g.leaf if id(m) in g.executed}


def w4_stats(g, kept, harmless=None, with_pc=False):
    """Z family over kept targets of graph g. harmless: dict name -> bool channel mask (dep graph only)."""
    per_z, per_zp, per_zw, per_el, per_pc = [], [], [], [], []
    nz_tot = nc_tot = near = nonfin = n_h_in_zeroed = 0
    for n, w, ax in kept:
        if not bool(torch.isfinite(w).all()):
            nonfin += 1
        q = C.quantize(torch, w, ax, 'pt')
        dead = C.zeroed_mask(torch, w, ax, 'pt', q=q)
        Cn = int(w.shape[ax])
        nz = int(dead.sum())
        per_z.append((n, nz / Cn)); nz_tot += nz; nc_tot += Cn
        if harmless is not None:
            h = harmless[n]
            n_h_in_zeroed += int((dead & h).sum())
            per_zp.append((n, int((dead & ~h).sum()) / Cn))
        per_zw.append((n, C.zeroed_energy(torch, w, ax, dead=dead)))
        per_el.append((n, float(((q == 0) & (w != 0)).double().mean())))
        A = float(w.abs().max())
        if A > 0 and math.isfinite(A):
            r = C._to_axis0(w.detach(), ax).reshape(Cn, -1).abs().amax(1).double() / A
            near += int(((r - 1.0 / (2 * (2 ** (C.BITS - 1) - 1))).abs() <= C.NEAR_TOL).sum())
        if with_pc:
            per_pc.append((n, int(C.zeroed_mask(torch, w, ax, 'pc').sum()) / Cn))

    def mx(per):
        return max([v for _, v in per] or [0.0])

    def top(per):
        return sorted(per, key=lambda t: -t[1])[:3]
    out = {'Z': mx(per_z), 'Zglob': (nz_tot / nc_tot) if nc_tot else 0.0, 'Zw': mx(per_zw), 'Zelem': mx(per_el),
           'top3_z': top(per_z), 'top3_zw': top(per_zw), 'n_near_threshold_channels': near, 'n_nonfinite_tensors': nonfin,
           'n_zeroed_channels': nz_tot, 'n_channels': nc_tot}
    if harmless is not None:
        out.update({'Zprime': mx(per_zp), 'top3_zprime': top(per_zp), 'n_harmless_in_zeroed': n_h_in_zeroed})
    if with_pc:
        out.update({'Zpc': mx(per_pc), 'n_pc_zeroed_tensors': sum(1 for _, v in per_pc if v > 0)})
    return out


def integrity_mine(ad, it, gT, gD, n_folded, m_layers):
    def lv_pt(g):
        return max([int(torch.unique(QA.rtn_sym_tensor(torch, g.orig[n], 4)).numel()) for n, _ in g.targets] or [0])
    mx = 0
    for n, _ in gD.targets:
        _, lv = QD.rtn_sym_channel(torch, gD.orig[n], 4, gD.axis[n])
        mx = max(mx, int(lv.abs().max()))
    zs = [float((QA.prune_layer(torch, w, 0.5) == 0).double().mean()) for _, w in m_layers]
    return {'n_folded': n_folded,
            'params_inf_dep': sum(p.numel() for n, p in gD.model.named_parameters() if not ad.is_train_only(n)),
            'params_inf_train': sum(p.numel() for n, p in gT.model.named_parameters() if not ad.is_train_only(n)),
            'S_w4g': {'n_tensors': len(gD.targets), 'max_levels': lv_pt(gD)},
            'O_w4': {'n_tensors': len(gT.targets), 'max_levels': lv_pt(gT)},
            'D_w4': {'n_tensors': len(gD.targets), 'n_tensors_axis_size1': sum(1 for n, _ in gD.targets if gD.orig[n].shape[gD.axis[n]] == 1),
                     'n_axis1_tensors': sum(1 for n, _ in gD.targets if gD.axis[n] == 1), 'max_abs_level': mx},
            'D_l50': {'n_layers': len(m_layers), 'min_zero_frac': min(zs) if zs else None}}


def compare(mine, ref):
    bad = []
    for k, v in ref.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                a = mine[k][kk]
                if kk == 'min_zero_frac':
                    if a is None or vv is None or abs(a - vv) > 1e-4:
                        bad.append('%s.%s mine=%s f3=%s' % (k, kk, a, vv))
                elif a != vv:
                    bad.append('%s.%s mine=%s f3=%s' % (k, kk, a, vv))
        elif mine[k] != v:
            bad.append('%s mine=%s f3=%s' % (k, mine[k], v))
    return bad


def process(it, pairs, man, dry):
    t0 = time.time()
    for f in ('checkpoint.pth', 'results.json'):
        p = it['dir'] + '/' + f
        if man.get(p) != QD.sha_file(p):
            raise RuntimeError('MANIFEST_MISMATCH ' + p)
    ad = registry.get(it['model'])
    r = QD.jload(it['dir'] + '/results.json')
    assert r['config']['model_cfg'] == ad.DEFAULT_CFG
    state, ck_sha = QD.load_state(torch, it['dir'] + '/checkpoint.pth')
    x1 = x_fixed()
    m = QD.build(ad, state).to(DEV)
    dep, n_folded = QD.make_dep(torch, f2_fuse, f2b, ad, it['model'], m, pairs, x1, DEV)
    gT = QD.Graph(torch, QA, ad, m, x1)
    gD = QD.Graph(torch, QA, ad, dep, x1)
    usedT, okT = C.used_param_names(torch, ad.infer, m, x1, leaf_exec_names(gT))
    usedD, okD = C.used_param_names(torch, ad.infer, dep, x1, leaf_exec_names(gD))
    assert gT.exact() and gD.exact()
    keepT, unexT, toT = graph_targets(ad, gT, usedT)
    keepD, unexD, toD = graph_targets(ad, gD, usedD)
    m_layers = [(n, gD.orig[n + '.weight']) for n, mm in gD.leaf if id(mm) in gD.executed
                and getattr(mm, 'weight', None) is not None and mm.weight.dim() >= 2]
    mine = integrity_mine(ad, it, gT, gD, n_folded, m_layers)
    bad = compare(mine, f3_record(it['tag']))
    rec = {'tag': it['tag'], 'model': it['model'], 'ds': it['ds'], 'k': it['k'], 'ckpt_sha256': ck_sha,
           'integrity_mine': mine, 'integrity_mismatch': bad, 'autograd_ok': {'train': okT, 'dep': okD},
           'excluded': {'dep_unexecuted': unexD, 'dep_train_only': toD, 'train_unexecuted': unexT, 'train_train_only': toT},
           'n_targets': {'dep': len(gD.targets), 'dep_kept': len(keepD), 'train': len(gT.targets), 'train_kept': len(keepT)},
           'n_l50_layers': len(m_layers), 'log10_params_inf_dep': math.log10(mine['params_inf_dep'])}
    if not dry and not bad:
        m0, n_off = C.gamma_switched_off_copy(torch, m)
        dep0, _ = QD.make_dep(torch, f2_fuse, f2b, ad, it['model'], m0, pairs, x1, DEV)
        p0 = dict(dep0.named_parameters())
        harmless = {n: (C.channel_allzero(p0[n].detach(), ax) & ~C.channel_allzero(w, ax)) for n, w, ax in keepD}
        rec['n_bn_channels_switched_off'] = n_off
        rec['n_harmless_channels_dep_kept'] = sum(int(v.sum()) for v in harmless.values())
        rec['dep'] = w4_stats(gD, keepD, harmless=harmless, with_pc=True)
        rec['train'] = w4_stats(gT, keepT)
        ms = C.m_stat(torch, m_layers)
        rec['l50'] = {'Mstat': ms['Mstat'], 'E': ms['E'], 'Msum': ms['Msum'], 'top3_m': sorted(ms['per_layer'], key=lambda t: -t[1])[:3]}
        # sensitivity: Z over ALL targets (F3 operator set, no exclusion)
        allD = [(n, gD.orig[n], gD.axis[n]) for n, _ in gD.targets]
        rec['Z_dep_all_targets'] = C.z_stat(torch, allD, 'pt')['Z']
        rec['Zpc_dep_all_targets'] = C.z_stat(torch, allD, 'pc')['Z']
        del m0, dep0
    rec['cpu_wall_s'] = time.time() - t0
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dryrun', default='')
    ap.add_argument('--run', action='store_true')
    a = ap.parse_args()
    items = QD.jload(QD.BASE + '/ITEMS.json')
    man = QD.manifest()
    pairs = QD.load_pairs()
    os.makedirs(OUTD, exist_ok=True)
    if a.dryrun:
        tags = set(a.dryrun.split(','))
        for it in [i for i in items if i['tag'] in tags]:
            rec = process(it, pairs.get(it['model'], []), man, True)
            ex = rec['excluded']
            print('DRY %s %.1fs mismatch=%s autograd=%s targets dep %d kept %d (unexec %d, train_only %d) | train %d kept %d | l50 layers %d' % (
                rec['tag'], rec['cpu_wall_s'], rec['integrity_mismatch'] or 'none', rec['autograd_ok'], rec['n_targets']['dep'],
                rec['n_targets']['dep_kept'], len(ex['dep_unexecuted']), len(ex['dep_train_only']), rec['n_targets']['train'],
                rec['n_targets']['train_kept'], rec['n_l50_layers']), flush=True)
        return
    if not a.run:
        return
    used = 0.0
    for it in items:
        tag = it['tag']
        if os.path.exists('%s/%s.DONE' % (OUTD, tag)) or os.path.exists('%s/%s.FAIL' % (OUTD, tag)):
            continue
        if used >= CAP_S:
            open(OUTD + '/STOPPED_CAP', 'w').write('%s used %.0fs\n' % (time.strftime('%F %T %Z'), used)); sys.exit(5)
        t0 = time.time()
        try:
            rec = process(it, pairs.get(it['model'], []), man, False)
        except Exception:
            open('%s/%s.FAIL' % (OUTD, tag), 'w').write(traceback.format_exc())
            print('[%s] %s FAIL %s' % (time.strftime('%T'), tag, traceback.format_exc().strip().splitlines()[-1]), flush=True)
            used += time.time() - t0
            continue
        used += rec['cpu_wall_s']
        if rec['integrity_mismatch']:
            QD.dump_json('%s/%s.integrity.json' % (OUTD, tag), rec)
            open(OUTD + '/STOPPED_INTEGRITY', 'w').write('%s %s %s\n' % (time.strftime('%F %T %Z'), tag, rec['integrity_mismatch']))
            print('STOP INTEGRITY', tag, rec['integrity_mismatch'], flush=True)
            sys.exit(6)
        QD.dump_json('%s/%s.json' % (OUTD, tag), rec)
        open('%s/%s.DONE' % (OUTD, tag), 'w').write('%.1f\n' % rec['cpu_wall_s'])
        print('[%s] %s %.1fs integrity ok, excluded dep %d, used %.0fs' % (time.strftime('%T'), tag, rec['cpu_wall_s'],
              len(rec['excluded']['dep_unexecuted']) + len(rec['excluded']['dep_train_only']), used), flush=True)
    print('RUN_END used=%.1fs' % used, flush=True)


if __name__ == '__main__':
    main()
