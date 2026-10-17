# -*- coding: utf-8 -*-
"""THEORY_Q synthetic reverse checks (PREREG_THEORYQ.md section 5). CPU only, fake tensors only, no checkpoint is read.
Writes theoryq_synth.json next to this file and prints one line per check."""
import os
import sys
import json
import math
import random

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch  # noqa: E402
torch.set_num_threads(2)
import theoryq_core as C  # noqa: E402

SEED = 20261017
res = {}


def rec(key, ok, msg):
    res[key] = {'PASS': bool(ok), 'msg': msg}
    print('SYNTH %s %s: %s' % (key, 'PASS' if ok else 'FAIL', msg), flush=True)


def planted(shape, axis, ranges, gen):
    """random tensor whose output channel c (along axis) has max|w| exactly ranges[c] (0 -> all-zero channel)."""
    w = torch.randn(*shape, generator=gen)
    wt = w.transpose(0, axis) if axis != 0 else w
    for c, r in enumerate(ranges):
        a = wt[c].abs().max()
        wt[c] = wt[c] / a * r if r > 0 else 0.0
    return (wt.transpose(0, axis) if axis != 0 else wt).contiguous()


g = torch.Generator().manual_seed(SEED)
torch.manual_seed(SEED)
qm = 2 ** (C.BITS - 1) - 1           # 7: code grid; zero threshold = A / (2 qm) = A / 14
A = 3.0

# S1 planted small-range channels, conv layout (axis 0) and ConvTranspose layout (axis 1)
r = [A, A / 10, A / 20, A / 9, A / 30, A / 13, A / 50, A / 2] + [A / 5] * 7 + [0.0]   # 16 ch; 3 planted <= A/14 (A/13 is above), ch15 all-zero
want = sum(1 for x in r if 0 < x <= A / (2 * qm))
w = planted((16, 8, 3, 3), 0, r, g)
nz, nc = C.zeroed_channels(torch, w, 0, 'pt')
wt = planted((8, 16, 3, 3), 1, r, g)
nzt, nct = C.zeroed_channels(torch, wt, 1, 'pt')
rec('S1_planted_pt', nz == want == 3 and nc == 16 and nzt == want and nct == 16,
    'conv %d/%d, convT %d/%d zeroed (planted %d; pre-existing zero channel not counted)' % (nz, nc, nzt, nct, want))

# S2 boundary: code threshold A/14 vs design threshold A/15
eps = 1e-3
rb = [A, A / 14 * (1 - eps), A / 14 * (1 + eps), A / 14.5, A / 15 * (1 + eps), A / 16]
wb = planted((6, 4, 3, 3), 0, rb, g)
qb = C.QA.rtn_sym_tensor(torch, wb, 4)
z = [bool((qb[c] == 0).all()) for c in range(6)]
design = [0 < x < A / 15 for x in rb]
rec('S2_threshold_code_vs_design', z == [False, True, False, True, True, True] and design == [False, False, False, False, False, True],
    'code zeroes r/A in {%s}; design rule r < A/15 would zero only r/A=1/16 -> channels with r/A in (1/15, 1/14] are the discrepancy'
    % ', '.join('%.4f' % (x / A) for x, zz in zip(rb, z) if zz))

# S3 per-channel: Z = 0 even for range ratio 1e6 (Prop 2)
nz_pc = C.zeroed_channels(torch, w, 0, 'pc')[0] + C.zeroed_channels(torch, wt, 1, 'pc')[0]
big = 0
for i in range(200):
    rr = [10 ** random.Random(SEED + i).uniform(-6, 0) * A for _ in range(32)]
    rr[0] = A
    big += C.zeroed_channels(torch, planted((32, 4, 3, 3), 0, rr, g), 0, 'pc')[0]
rec('S3_per_channel_zero', nz_pc == 0 and big == 0, 'planted tensors %d, 200 tensors with range ratio up to 1e6: %d zeroed channels' % (nz_pc, big))

# S4 analytic count = code count on random tensors (log-uniform channel ranges, boundary band 1e-6 excluded)
mism, tot = 0, 0
rng = random.Random(SEED)
for i in range(300):
    C_ = rng.choice([4, 16, 64]); ax = rng.choice([0, 1])
    rr = [A * 10 ** rng.uniform(-2.5, 0) for _ in range(C_)]
    rr[rng.randrange(C_)] = A
    rr = [x for x in rr]
    shape = (C_, 3, 3, 3) if ax == 0 else (3, C_, 3, 3)
    ww = planted(shape, ax, rr, g)
    Aw = float(ww.abs().max())
    if any(abs(x / Aw * 2 * qm - 1) < 1e-6 for x in rr):
        continue
    ana = sum(1 for x in rr if 0 < x <= Aw / (2 * qm))
    mism += int(ana != C.zeroed_channels(torch, ww, ax, 'pt')[0]); tot += 1
rec('S4_analytic_vs_code', mism == 0 and tot > 250, '%d tensors, %d mismatches between {0<r_c<=A/(2qm)} and the code' % (tot, mism))

# S5 Z statistic over a fake checkpoint (max over layers, Zglob)
layers = [('a', planted((16, 8, 3, 3), 0, [A] + [A / 5] * 15, g), 0),         # 0/16
          ('b', planted((16, 8, 3, 3), 0, [A] + [A / 20] * 3 + [A / 5] * 12, g), 0),   # 3/16
          ('c', planted((8, 16, 3, 3), 1, [A] + [A / 40] * 5 + [A / 5] * 10, g), 1)]   # 5/16 (axis 1)
zs = C.z_stat(torch, layers, 'pt'); zp = C.z_stat(torch, layers, 'pc')
rec('S5_Z_max_over_layers', abs(zs['Z'] - 5 / 16) < 1e-12 and abs(zs['Zglob'] - 8 / 48) < 1e-12 and zp['Z'] == 0.0,
    'Z=%.4f (5/16), Zglob=%.4f (8/48), per-channel Z=%.1f' % (zs['Z'], zs['Zglob'], zp['Z']))

# S6 pruning energy: analytic value + ranking
n = 1000
lin = torch.arange(1, n + 1, dtype=torch.float32)
lin = lin[torch.randperm(n, generator=g)] * torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0)
m_lin = C.pruned_energy(torch, lin.view(10, 100))
ana = sum(i * i for i in range(1, 501)) / sum(i * i for i in range(1, n + 1))
flat = (1.0 + 1e-3 * torch.rand(n, generator=g)).view(10, 100)          # near-constant magnitude -> m ~ 0.5
heavy = (torch.distributions.Pareto(1.0, 1.2).sample((n,)) if True else None).view(10, 100)  # heavy tail -> small m
m_flat, m_heavy = C.pruned_energy(torch, flat), C.pruned_energy(torch, heavy)
ms = C.m_stat(torch, [('lin', lin.view(10, 100)), ('flat', flat), ('heavy', heavy)])
rec('S6_pruned_energy', abs(m_lin - ana) < 1e-6 and m_flat > m_lin > m_heavy and abs(m_flat - 0.5) < 0.01 and ms['Mstat'] == m_flat,
    'linear magnitudes m=%.6f (analytic %.6f), flat %.4f > linear > heavy-tail %.4f; Mstat picks flat; E=%.4f' % (m_lin, ana, m_flat, m_heavy, ms['E']))

# S7 planted pruning energy ranks checkpoints: |w_i| = (i/n)^alpha, analytic m = sum_{i<=n/2}(i/n)^(2a) / sum_i (i/n)^(2a)
cks, anas = [], []
idx = torch.arange(1, n + 1, dtype=torch.float64) / n
for j, al in enumerate([4.0, 2.0, 1.0, 0.5, 0.25]):
    planted_l = (idx ** al)[torch.randperm(n, generator=g)].float().view(10, 100)
    other = (idx ** 8.0)[torch.randperm(n, generator=g)].float().view(10, 100)  # filler layer, m ~ 0.5^17 (below every planted m)
    cks.append(C.m_stat(torch, [('x', other), ('y', planted_l)])['Mstat'])
    anas.append(float((idx[: n // 2] ** (2 * al)).sum() / (idx ** (2 * al)).sum()))
sp = C.QAN.spearman(cks, [0, 1, 2, 3, 4])
err = max(abs(a - b) for a, b in zip(cks, anas))
rec('S7_m_ranks_planted', sp is not None and abs(sp - 1) < 1e-12 and err < 1e-5,
    'Mstat over 5 planted checkpoints %s (analytic %s, max err %.1e), Spearman %.3f' % (['%.4f' % v for v in cks], ['%.4f' % v for v in anas], err, sp))

# S8 AUC rules on 66 fake cells
cells = [('m%d' % (i // 6), 'd%d' % (i % 6)) for i in range(66)]
rng = random.Random(SEED)
sc = {c: rng.random() for c in cells}
lab_perfect = {c: sc[c] > 0.42 for c in cells}
lab_rev = {c: sc[c] < 0.42 for c in cells}
a1 = C.cell_auc(sc, lab_perfect); a0 = C.cell_auc(sc, lab_rev)
rnd = []
for i in range(200):
    rr = random.Random(SEED + 1000 + i)
    lab = {c: rr.random() < 38 / 66 for c in cells}
    rnd.append(C.cell_auc({c: rr.random() for c in cells}, lab))
ties = C.cell_auc({c: 0.0 for c in cells}, lab_perfect)
bt = C.cell_boot(sc, lab_perfect)
rr = random.Random(SEED + 7)
lab_r = {c: rr.random() < 38 / 66 for c in cells}
br = C.cell_boot({c: rr.random() for c in cells}, lab_r)
mean_rnd = sum(rnd) / len(rnd)
rec('S8_auc_cells', a1 == 1.0 and a0 == 0.0 and abs(mean_rnd - 0.5) < 0.02 and max(abs(v - 0.5) for v in rnd) < 0.25 and ties == 0.5
    and bt['ci95'] == [1.0, 1.0] and bt['p_le_null'] < 0.001 and br['ci95'][0] < 0.5 < br['ci95'][1],
    'perfect %.3f, reversed %.3f, random mean over 200 = %.3f (range %.3f-%.3f), all-tied %.2f; perfect boot CI %s p=%.4f; random boot CI [%.3f, %.3f] p=%.3f'
    % (a1, a0, mean_rnd, min(rnd), max(rnd), ties, bt['ci95'], bt['p_le_null'], br['ci95'][0], br['ci95'][1], br['p_le_null']))

# S9 cell statistic with a missing split (malunet isic18 has 4 splits)
rows = [{'model': 'a', 'ds': 'x', 'Z': v} for v in (0.0, 0.1, 0.2, 0.3, 0.4)] + [{'model': 'b', 'ds': 'y', 'Z': v} for v in (0.5, 0.0, 0.1, 0.2)]
cm, cx = C.cell_mean(rows, 'Z'), C.cell_max(rows, 'Z')
rec('S9_cell_stat', abs(cm[('a', 'x')] - 0.2) < 1e-12 and abs(cm[('b', 'y')] - 0.2) < 1e-12 and cx[('b', 'y')] == 0.5,
    'mean a=%.2f b(4 splits)=%.2f, max b=%.2f' % (cm[('a', 'x')], cm[('b', 'y')], cx[('b', 'y')]))

# S10 Holm (hand values)
h = C.holm({'T1': 0.01, 'T3': 0.04}); h2 = C.holm({'T1': 0.03, 'T3': 0.20})
rec('S10_holm', h['T1'] == (0.02, True) and h['T3'] == (0.04, True) and h2['T1'] == (0.06, False) and h2['T3'] == (0.2, False),
    'p (0.01, 0.04) -> %s; p (0.03, 0.20) -> %s' % (h, h2))

# S11 Prop 3 bound on a bias-free toy DAG (conv chain + additive skip + concat skip + ReLU + maxpool + nearest upsample)
torch.manual_seed(SEED)
nnf = torch.nn.functional
Wd = {'c1': torch.randn(8, 3, 3, 3) * 0.3, 'c2': torch.randn(8, 8, 3, 3) * 0.2, 'c3': torch.randn(8, 8, 3, 3) * 0.2,
      'c4': torch.randn(8, 16, 3, 3) * 0.2, 'c5': torch.randn(1, 8, 1, 1)}


def net(W, x):
    h1 = torch.relu(nnf.conv2d(x, W['c1'], padding=1))
    h2 = torch.relu(nnf.conv2d(nnf.max_pool2d(h1, 2), W['c2'], padding=1))
    h3 = torch.relu(nnf.conv2d(h2, W['c3'], padding=1)) + h2                          # additive skip
    u = nnf.interpolate(h3, scale_factor=2, mode='nearest')                             # op norm 2
    h4 = torch.relu(nnf.conv2d(torch.cat([u, h1], 1), W['c4'], padding=1))            # concat skip
    return nnf.conv2d(h4, W['c5'])


def N(w):
    return math.sqrt(w.shape[2] * w.shape[3]) * float(w.norm())


# path-sum Lipschitz bound P_out: x->c1->[pool]->c2->{c3 + id}->up(2)->c4 ; x->c1->c4 (concat); -> c5
P1 = N(Wd['c1']); P2 = N(Wd['c2']) * P1; P3 = (N(Wd['c3']) + 1.0) * P2
Pout = N(Wd['c5']) * N(Wd['c4']) * (2.0 * P3 + P1)
Wp = {k: C.QA.prune_layer(torch, v, 0.5) for k, v in Wd.items()}
S = sum(math.sqrt(C.pruned_energy(torch, v)) for v in Wd.values())
# operator-norm check for the conv bound sqrt(k k) ||K||_F (power iteration)
x0 = torch.randn(1, 8, 32, 32)
for _ in range(50):
    y = nnf.conv2d(x0, Wd['c2'], padding=1); x0 = nnf.conv_transpose2d(y, Wd['c2'], padding=1); x0 = x0 / x0.norm()
opn = float(nnf.conv2d(x0, Wd['c2'], padding=1).norm())
worst, ok = 0.0, True
for i in range(50):
    x = torch.randn(2, 3, 32, 32) * (1 + i)
    lhs = float((net(Wd, x) - net(Wp, x)).norm()); rhs = Pout * float(x.norm()) * S
    ok = ok and lhs <= rhs
    worst = max(worst, lhs / rhs)
rec('S11_prop3_bound_toy_dag', ok and opn <= N(Wd['c2']) * (1 + 1e-6),
    'conv op norm %.3f <= sqrt(9)||K||_F %.3f; ||f-f~|| <= P_out ||x|| sum sqrt(m_l) on 50 inputs, max ratio lhs/rhs = %.2e (bound loose)'
    % (opn, N(Wd['c2']), worst))


# S12 harmless (BN switched-off) channels: gamma ~ 0 -> after folding the copy, exactly those channels are all-zero
nnm = torch.nn
toy = nnm.Sequential(nnm.Conv2d(4, 8, 3, bias=False), nnm.BatchNorm2d(8)).eval()
with torch.no_grad():
    toy[1].weight.copy_(torch.tensor([1.0, 0.5, 1e-4, 3.0, 0.0, 1.5, 2.9e-3, 3.1e-3]))   # tol 1e-3 * max 3.0 = 3e-3 (no exact-boundary value)
    toy[1].running_var.uniform_(0.5, 2.0, generator=g)


def fold_w(m):
    bn = m[1]
    t = bn.weight / (bn.running_var + bn.eps).sqrt()
    return (m[0].weight * t.view(-1, 1, 1, 1)).detach()


m0, noff = C.gamma_switched_off_copy(torch, toy)
h = C.channel_allzero(fold_w(m0), 0) & ~C.channel_allzero(fold_w(toy), 0)
rec('S12_harmless_bn_channels', noff == 3 and h.tolist() == [False, False, True, False, False, False, True, False]
    and abs(float(toy[1].weight[2]) - 1e-4) < 1e-9,
    'switched-off BN channels %d (gamma 1e-4, 0, 2.9e-3 <= 3e-3; 3.1e-3 kept); harmless after fold %s (gamma=0 channel already zero, not counted); original untouched'
    % (noff, [i for i, v in enumerate(h.tolist()) if v]))


# S13 executed-tensor rule: unused branch and unused Parameter are excluded, used ones kept
class Toy(nnm.Module):
    def __init__(s):
        super().__init__()
        s.a = nnm.Conv2d(3, 4, 3, padding=1); s.unused = nnm.Conv2d(3, 4, 3, padding=1)
        s.scale = nnm.Parameter(torch.ones(1, 4, 1, 1)); s.dead = nnm.Parameter(torch.ones(2, 2))

    def forward(s, x):
        return s.a(x) * s.scale


tm = Toy().eval()
used, okg = C.used_param_names(torch, lambda m_, x_: m_(x_), tm, torch.rand(1, 3, 8, 8, generator=g), set())
used2, _ = C.used_param_names(torch, lambda m_, x_: m_(x_), tm, torch.rand(1, 3, 8, 8, generator=g), {'unused'})
rec('S13_executed_rule', okg and used == {'a.weight', 'a.bias', 'scale'} and used2 == used | {'unused.weight', 'unused.bias'}
    and all(p.grad is None for p in tm.parameters()),
    'autograd used=%s; union with hook-executed leaf adds %s; grads cleared' % (sorted(used), sorted(used2 - used)))

res['ALL_PASS'] = all(v['PASS'] for v in res.values() if isinstance(v, dict))
res['torch'] = torch.__version__
with open(os.path.join(HERE, 'theoryq_synth.json'), 'w', encoding='utf8') as f:
    json.dump(res, f, indent=1)
print('SYNTH ALL', 'PASS' if res['ALL_PASS'] else 'FAIL')
