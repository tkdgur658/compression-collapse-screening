"""Data pipeline shared by every model (protocol v1).

Splits are the fixed folders data/<dir>/{train,val,test}_<k>/{images,masks} of the legacy
repository (k = 1..5). Skin datasets are 60/20/20 and colonoscopy datasets 80/10/10.

Deterministic preprocessing (cached once per split):
  RGB uint8 -> per-image min-max to [0, 1] -> aspect-ratio-preserving resize to 256
  (bilinear, antialias) -> centred zero padding to 256x256.
  Mask: grey/255 -> same resize (bilinear) and padding -> binarised at 0.5.
Training augmentation (per sample): hflip p=0.5, vflip p=0.5, rotation p=0.5 with an angle
drawn uniformly from [0, 360) for every sample (image bilinear, mask nearest, fill 0).
"""
import hashlib
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

DATASETS = {
    # name: (folder under data root, modality)
    'PH2': ('PH2', 'dermoscopy'),
    'isic17': ('isic2017', 'dermoscopy'),
    'isic18': ('isic2018', 'dermoscopy'),
    'kvasir': ('kvasir_seg', 'colonoscopy'),
    'cvc': ('cvc_clinicdb', 'colonoscopy'),
    'etis': ('etis', 'colonoscopy'),
}
SIZE = 256
CACHE_VERSION = 'v1'


def _stem(fname, is_mask):
    s = os.path.splitext(fname)[0]
    for suf in ('_segmentation', '_lesion', '_mask'):
        if is_mask and s.endswith(suf):
            s = s[: -len(suf)]
    return s


def list_pairs(split_dir):
    """Pair images and masks by file stem; refuse silently mismatched folders."""
    imgs = sorted(os.listdir(os.path.join(split_dir, 'images')))
    msks = sorted(os.listdir(os.path.join(split_dir, 'masks')))
    mmap = {_stem(m, True): m for m in msks}
    pairs = []
    for f in imgs:
        st = _stem(f, False)
        if st not in mmap:
            raise FileNotFoundError(f'no mask for {f} in {split_dir}')
        pairs.append((os.path.join(split_dir, 'images', f), os.path.join(split_dir, 'masks', mmap[st])))
    if len(pairs) != len(msks):
        raise ValueError(f'{split_dir}: {len(imgs)} images vs {len(msks)} masks')
    return pairs


def _keep_ratio(t, size, mode):
    """t: (C,H,W) float tensor -> (C,size,size) with centred zero padding."""
    h, w = t.shape[-2:]
    scale = min(size / h, size / w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    if mode == 'bilinear':
        t = F.interpolate(t[None], size=(nh, nw), mode='bilinear', align_corners=False, antialias=True)[0]
    else:
        t = F.interpolate(t[None], size=(nh, nw), mode=mode)[0]
    ph, pw = size - nh, size - nw
    return F.pad(t, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), value=0.0)


def preprocess_pair(img_path, msk_path, size=SIZE):
    img = np.asarray(Image.open(img_path).convert('RGB'), dtype=np.float32)
    lo, hi = img.min(), img.max()
    img = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
    msk = np.asarray(Image.open(msk_path).convert('L'), dtype=np.float32) / 255.0
    it = _keep_ratio(torch.from_numpy(img).permute(2, 0, 1), size, 'bilinear')
    mt = _keep_ratio(torch.from_numpy(msk)[None], size, 'bilinear')
    return it.clamp(0, 1), (mt > 0.5).float()


def build_cache(data_root, dataset, split, k, cache_root):
    """Preprocess one split once; images stored as uint8 (x255), masks as uint8 {0,1}."""
    folder = DATASETS[dataset][0]
    split_dir = os.path.join(data_root, folder, f'{split}_{k}')
    pairs = list_pairs(split_dir)
    h = hashlib.sha256()
    for ip, mp in pairs:
        for p in (ip, mp):
            st = os.stat(p)
            h.update(f'{os.path.relpath(p, data_root)}|{st.st_size}'.encode())
    key = h.hexdigest()[:16]
    out = os.path.join(cache_root, CACHE_VERSION, dataset, f'{split}_{k}_{key}.npz')
    if not os.path.exists(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        ims, mks = [], []
        for ip, mp in pairs:
            it, mt = preprocess_pair(ip, mp)
            ims.append((it * 255).round().byte().numpy())
            mks.append(mt.byte().numpy())
        tmp = out + '.tmp.npz'
        np.savez(tmp, images=np.stack(ims), masks=np.stack(mks),
                 names=np.array([os.path.basename(p[0]) for p in pairs]))
        os.replace(tmp, out)
        with open(out + '.json', 'w') as f:
            json.dump({'split_dir': split_dir, 'n': len(pairs), 'list_key': key,
                       'cache_version': CACHE_VERSION}, f, indent=1)
    return out


class SegCache(Dataset):
    def __init__(self, cache_file, train):
        z = np.load(cache_file)
        self.images = torch.from_numpy(z['images'])  # (N,3,S,S) uint8
        self.masks = torch.from_numpy(z['masks'])    # (N,1,S,S) uint8
        self.names = list(z['names'])
        self.train = train

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img = self.images[i].float() / 255.0
        msk = self.masks[i].float()
        if self.train:
            if random.random() < 0.5:
                img, msk = TF.hflip(img), TF.hflip(msk)
            if random.random() < 0.5:
                img, msk = TF.vflip(img), TF.vflip(msk)
            if random.random() < 0.5:
                ang = random.uniform(0.0, 360.0)
                img = TF.rotate(img, ang, interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)
                msk = TF.rotate(msk, ang, interpolation=TF.InterpolationMode.NEAREST, fill=0.0)
        return img, msk
