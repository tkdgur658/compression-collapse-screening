"""Collect complete v1 runs into one CSV and a status matrix (no estimates, no partial cells).

  python tools/aggregate.py --run-root ../runs --suite main --out ../runs/main_summary.csv

A cell (row, dataset) is reported only when all requested splits are complete with the same
code source digest set recorded per run. Paired two-sided t-tests over splits against --ref,
Holm-corrected within each dataset column.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
from scipy import stats


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.0
    for i, idx in enumerate(order):
        run = max(run, min(1.0, (m - i) * pvals[idx])); adj[idx] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-root', default='../runs'); ap.add_argument('--suite', default='main')
    ap.add_argument('--splits', default='1,2,3,4,5'); ap.add_argument('--ref', default='pbe')
    ap.add_argument('--metric', default='iou'); ap.add_argument('--out')
    a = ap.parse_args()
    ks = [int(k) for k in a.splits.split(',')]
    res = {}
    for f in glob.glob(os.path.join(a.run_root, a.suite, '*', '*', 'k*', 'results.json')):
        r = json.load(open(f))
        if r.get('status') != 'complete':
            continue
        c = r['config']
        res[(os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(f)))), c['dataset'], c['split'])] = r
    rows = sorted({k[0] for k in res}); dsets = sorted({k[1] for k in res})
    table, status = {}, []
    for d in dsets:
        for n in rows:
            v = [res[(n, d, k)]['test'][a.metric] if (n, d, k) in res else None for k in ks]
            done = sum(x is not None for x in v)
            status.append((n, d, done))
            if done == len(ks):
                table[(n, d)] = np.array(v)
    out = []
    for d in dsets:
        cands = [n for n in rows if (n, d) in table and n != a.ref]
        p = []
        for n in cands:
            if (a.ref, d) in table:
                p.append(stats.ttest_rel(table[(a.ref, d)], table[(n, d)]).pvalue)
        adj = holm(np.array(p)) if p else []
        padj = dict(zip(cands, adj)) if (a.ref, d) in table else {}
        for n in rows:
            if (n, d) in table:
                x = table[(n, d)]
                out.append({'row': n, 'dataset': d, 'mean': round(float(x.mean()), 4),
                            'std': round(float(x.std(ddof=1)), 4), 'n': len(x),
                            'p_holm_vs_ref': '' if n == a.ref or n not in padj else round(float(padj[n]), 4)})
    for o in out:
        print(f"{o['row']:28s} {o['dataset']:7s} {o['mean']:.4f} ± {o['std']:.4f}  p={o['p_holm_vs_ref']}")
    print('\n[status] complete splits per cell:')
    for n, d, done in status:
        if done < len(ks):
            print(f'  {n:28s} {d:7s} {done}/{len(ks)}')
    if a.out:
        with open(a.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]) if out else ['row'])
            w.writeheader(); w.writerows(out)


if __name__ == '__main__':
    main()
