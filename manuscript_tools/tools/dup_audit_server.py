# -*- coding: utf-8 -*-
"""Duplicate-image audit across the protocol-v1 splits (Pass 33, [internal decision], 2026-10-17).

Read-only on the archive. Writes only into <WORK>/dup_audit/. CPU only, no torch import.
Three hash levels per image (sha256):
  cache : bytes of the uint8 array images[i] (3x256x256) in the v1 cache npz actually used by training/evaluation
  file  : bytes of the original image file in <data_root>/<folder>/<split>_<k>/images/<name> (the file the cache was built from)
  pixel : bytes of the decoded original RGB array (PIL .convert('RGB'), native resolution, shape prefixed)
Masks: sha256 of the cache mask array, used to report whether duplicated images also carry identical masks, and the IoU
between the two cache masks of each duplicate pair.
Pair counts are counted on the second set: train-val = val images identical to >=1 train image; train-test = test images
identical to >=1 train image; val-test = test images identical to >=1 val image; test-any = test images identical to >=1
train or val image (the set removed in the appendix Table 1). Within-split = n - number of distinct hashes.
Cause: (a) name overlap between sets (split-generation error) vs (b) same content under different names in the
dataset union (dataset-level duplicates); raw source folders, where present, are checked for the same duplicates.
"""
import os as _ccs_os  # path configuration of the released copy (README, Configuration)
CCS_REPO_ROOT = _ccs_os.path.normpath(_ccs_os.path.join(_ccs_os.path.dirname(_ccs_os.path.abspath(__file__)), '../..'))
CCS_WORK_ROOT = _ccs_os.environ.get('CCS_WORK', _ccs_os.path.join(CCS_REPO_ROOT, 'experiments'))
CCS_ARCHIVE_ROOT = _ccs_os.environ.get('CCS_ARCHIVE', _ccs_os.path.join(CCS_REPO_ROOT, 'archive'))
CCS_DATA_ROOT = _ccs_os.environ.get('CCS_DATA', _ccs_os.path.join(CCS_REPO_ROOT, 'datasets'))
import glob
import hashlib
import io
import json
import os
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
from PIL import Image

CACHE = _ccs_os.path.join(CCS_ARCHIVE_ROOT, 'cache/v1')
DATA = CCS_DATA_ROOT
OUT = _ccs_os.path.join(CCS_WORK_ROOT, 'dup_audit')
DS = {'PH2': 'PH2', 'isic17': 'isic2017', 'isic18': 'isic2018', 'kvasir': 'kvasir_seg', 'cvc': 'cvc_clinicdb', 'etis': 'etis'}
RAW = {'etis': DATA + '/TestDataset/ETIS-LaribPolypDB/images', 'cvc': DATA + '/cvc_clinicdb_raw/PNG/Original',
       'kvasir': DATA + '/Kvasir-SEG/images', 'cvc_pranet': DATA + '/TestDataset/CVC-ClinicDB/images',
       'kvasir_pranet': DATA + '/TestDataset/Kvasir/images'}
SPLITS = [1, 2, 3, 4, 5]
SETS = ['train', 'val', 'test']

_pix_by_file = {}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def file_hashes(path):
    b = open(path, 'rb').read()
    fh = sha(b)
    if fh not in _pix_by_file:  # decode once per distinct file content
        a = np.ascontiguousarray(np.asarray(Image.open(io.BytesIO(b)).convert('RGB')))
        _pix_by_file[fh] = sha(str(a.shape).encode() + a.tobytes())
    return fh, _pix_by_file[fh]


def load_set(ds, s, k):
    fs = [f for f in glob.glob('%s/%s/%s_%d_*.npz' % (CACHE, ds, s, k)) if not f.endswith('.tmp.npz')]
    assert len(fs) == 1, (ds, s, k, fs)
    meta = json.load(open(fs[0] + '.json'))
    z = np.load(fs[0])
    names = [str(x) for x in z['names']]
    im, mk = z['images'], z['masks']
    ch = [sha(np.ascontiguousarray(im[i]).tobytes()) for i in range(im.shape[0])]
    mh = [sha(np.ascontiguousarray(mk[i]).tobytes()) for i in range(mk.shape[0])]
    fh, ph = [], []
    for n in names:
        a, b = file_hashes(os.path.join(meta['split_dir'], 'images', n))
        fh.append(a); ph.append(b)
    assert meta['n'] == len(names)
    return dict(cache_file=os.path.basename(fs[0]), split_dir=meta['split_dir'], names=names,
                cache=ch, file=fh, pixel=ph, mask=mh, mask_arr=mk)


def pair(A, B, lvl):
    """B images identical (level lvl) to >=1 A image -> list of (B name, [A names])"""
    idx = {}
    for n, h in zip(A['names'], A[lvl]):
        idx.setdefault(h, []).append(n)
    return [(n, idx[h]) for n, h in zip(B['names'], B[lvl]) if h in idx]


def within(S, lvl):
    return len(S[lvl]) - len(set(S[lvl]))


def raw_folder(path):
    fs = sorted(os.listdir(path))
    rec = {}
    for f in fs:
        a, b = file_hashes(os.path.join(path, f))
        rec[f] = (a, b)
    groups = {}
    for f, (a, b) in rec.items():
        groups.setdefault(b, []).append(f)
    dup = [g for g in groups.values() if len(g) > 1]
    fgroups = {}
    for f, (a, b) in rec.items():
        fgroups.setdefault(a, []).append(f)
    return dict(path=path, n_files=len(fs), n_distinct_pixel=len(groups), n_distinct_file=len(fgroups),
                pixel_dup_groups=sorted(dup), file_dup_groups=sorted(g for g in fgroups.values() if len(g) > 1)), rec


def main():
    t0 = time.time()
    res = {'levels': {'cache': 'sha256 of uint8 cache array images[i] (3x256x256, protocol-v1 cache = training/evaluation input)',
                      'file': 'sha256 of the original image file bytes in the split folder',
                      'pixel': 'sha256 of the decoded original RGB array at native resolution (PIL convert RGB)'},
           'count_rule': 'pair counts are counted on the second set (val for train-val, test for train-test, val-test, test-any)',
           'datasets': {}}
    for ds, folder in DS.items():
        R = {'splits': {}}
        unions = []
        for k in SPLITS:
            S = {s: load_set(ds, s, k) for s in SETS}
            d = {'n': {s: len(S[s]['names']) for s in SETS}, 'cache_files': {s: S[s]['cache_file'] for s in SETS}}
            for lvl in ('cache', 'file', 'pixel'):
                tv, tt, vt = pair(S['train'], S['val'], lvl), pair(S['train'], S['test'], lvl), pair(S['val'], S['test'], lvl)
                TV = dict(names=S['train']['names'] + S['val']['names'], **{lvl: S['train'][lvl] + S['val'][lvl]})
                ta = pair(TV, S['test'], lvl)
                d[lvl] = {'train_val': len(tv), 'train_test': len(tt), 'val_test': len(vt), 'test_any': len(ta),
                          'within_train': within(S['train'], lvl), 'within_val': within(S['val'], lvl),
                          'within_test': within(S['test'], lvl)}
                if lvl == 'cache':
                    d['pairs_cache'] = {'train_val': tv, 'train_test': tt, 'val_test': vt, 'test_any': ta}
                    d['test_removed'] = sorted(n for n, _ in ta)
                    # train side count (number of train images that have a copy in val or test) for the SGR cross-check
                    VT = dict(names=S['val']['names'] + S['test']['names'], cache=S['val']['cache'] + S['test']['cache'])
                    d['train_with_copy_in_val_or_test'] = len(pair(VT, S['train'], 'cache'))
            # name overlap between sets (split-generation error check)
            nm = {s: set(S[s]['names']) for s in SETS}
            d['name_overlap'] = {'train_val': len(nm['train'] & nm['val']), 'train_test': len(nm['train'] & nm['test']),
                                 'val_test': len(nm['val'] & nm['test'])}
            # masks of duplicated pairs (cache level)
            allm = {}
            for s in SETS:
                for n, h, m in zip(S[s]['names'], S[s]['cache'], S[s]['mask']):
                    allm.setdefault(h, []).append((s, n, m))
            grp = [g for g in allm.values() if len(g) > 1]
            d['dup_groups_cache'] = [[(s, n) for s, n, _ in g] for g in grp]
            d['dup_groups_mask_identical'] = sum(1 for g in grp if len(set(m for _, _, m in g)) == 1)
            d['dup_groups_n'] = len(grp)
            # IoU between the cache masks of the members of each duplicate group (same image, separate annotation?)
            ma = {(s, n): S[s]['mask_arr'][i].astype(bool) for s in SETS for i, n in enumerate(S[s]['names'])
                  if any((s, n) == (gs, gn) for g in grp for gs, gn, _ in g)}
            ious = []
            for g in grp:
                a, b = ma[(g[0][0], g[0][1])], ma[(g[1][0], g[1][1])]
                u = int((a | b).sum())
                ious.append({'members': [(s, n) for s, n, _ in g], 'fg_pixels': [int(a.sum()), int(b.sum())],
                             'mask_iou': round(float((a & b).sum()) / u, 4) if u else 1.0})
            d['dup_groups_mask_iou'] = ious
            R['splits'][str(k)] = d
            unions.append({n: (c, f, p) for s in SETS for n, c, f, p in zip(S[s]['names'], S[s]['cache'], S[s]['file'], S[s]['pixel'])})
            print('%s k%d done %.0fs' % (ds, k, time.time() - t0), flush=True)
        # dataset union (same file set in every split?)
        U = unions[0]
        R['union_same_names_all_splits'] = all(set(u) == set(U) for u in unions)
        R['union_same_content_all_splits'] = all(u == U for u in unions)
        R['n_union'] = len(U)
        R['union_pixel_hash'] = {n: h[2] for n, h in U.items()}
        for lvl, j in (('cache', 0), ('file', 1), ('pixel', 2)):
            g = {}
            for n, h in U.items():
                g.setdefault(h[j], []).append(n)
            R['union_dup_groups_' + lvl] = sorted(sorted(x) for x in g.values() if len(x) > 1)
            R['union_n_distinct_' + lvl] = len(g)
        res['datasets'][ds] = R
    # raw source folders
    res['raw'] = {}
    for key, path in RAW.items():
        if os.path.isdir(path):
            info, rec = raw_folder(path)
            info['pixel_hash'] = {f: b for f, (a, b) in rec.items()}
            res['raw'][key] = info
            print('raw %s done %.0fs' % (key, time.time() - t0), flush=True)
    res['seconds'] = round(time.time() - t0, 1)
    os.makedirs(OUT, exist_ok=True)
    with open(OUT + '/AUDIT.json', 'w') as f:
        json.dump(res, f, indent=1)
    print('ALL_DONE %.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
