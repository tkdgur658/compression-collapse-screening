# G2 (FIVES): is an S-standard remedy (resize / tiling / AMP / full-res) enough to leave no memory-bound gap?
# Pre-registration: <WORK>/s4mem/G2_PREREG.md  (read it first; this file implements it).
# Usage (run with `python -I`):
#   g2a_train.py prep [--out DIR] [--limit-train N --limit-val N]
#   g2a_train.py run --model {ucmnet,utanet,maresunet} --variant {native,rs256,rs512,rs1024} [--out DIR]
# Reads ONLY the FIVES official train/ folders. The official test/ folder is never opened.
import os as _ccs_os  # path configuration of the released copy (README, Configuration)
CCS_REPO_ROOT = _ccs_os.path.normpath(_ccs_os.path.join(_ccs_os.path.dirname(_ccs_os.path.abspath(__file__)), '../../..'))
CCS_WORK_ROOT = _ccs_os.environ.get('CCS_WORK', _ccs_os.path.join(CCS_REPO_ROOT, 'experiments'))
CCS_ARCHIVE_ROOT = _ccs_os.environ.get('CCS_ARCHIVE', _ccs_os.path.join(CCS_REPO_ROOT, 'archive'))
import os, sys, json, time, copy, glob, gc, argparse
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
sys.dont_write_bytecode = True
import numpy as np
import cv2
cv2.setNumThreads(1)
import multiprocessing as mp
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_V2 = _ccs_os.path.join(CCS_ARCHIVE_ROOT, 'code_v2')          # imported, never edited
BASE = _ccs_os.path.join(CCS_WORK_ROOT, 's4mem')
OUT_DEFAULT = BASE + "/g2a"
SPLIT_SEED, N_VAL, SEED = 20261014, 100, 1
STEPS, BATCH, CROP = 8000, 16, 256
LR, WD, ETA_MIN, CLIP = 1e-3, 1e-2, 1e-6, 5.0
SIZE = 2048
RES = (256, 512, 1024)                                         # resize remedies (downscale -> upsample)
PATCHES, OVERLAPS = (256, 512), (0, 50)
TBS = {256: 8, 512: 4}                                         # tile batch for the metric passes (memory is measured at 1)
GRIDS = (128, 256, 512)                                        # seam-grid strides (px); band = +-8 px
BAND = 8
PROBE_SIZES = tuple(range(256, 2049, 256))
CPU_THREADS = 16                                               # only used by the full-res CPU fallback
NPOOL = 8


def train_root():
    return glob.glob(BASE + "/data/FIVES/extracted/*/")[0] + "train"


# ------------------------------------------------------------------ prep (cache) ----------------------------------
def _prep_one(args):
    from PIL import Image
    i, ip, mp_ = args
    assert "/test/" not in ip and "/test/" not in mp_
    img = np.array(Image.open(ip).convert("RGB"), dtype=np.uint8)
    msk = (np.array(Image.open(mp_))[..., 0] > 127).astype(np.uint8)       # same rule as G1 (channel 0 > 127)
    assert img.shape == (SIZE, SIZE, 3) and msk.shape == (SIZE, SIZE)
    lo, hi = float(img.min()), float(img.max())
    x = (img.astype(np.float32) - lo) / (hi - lo) if hi > lo else np.zeros(img.shape, np.float32)
    m = msk.astype(np.float32)
    ds = {}
    for R in RES:
        f = SIZE // R
        di = x.reshape(R, f, R, f, 3).mean((1, 3)).astype(np.float16)
        dm = m.reshape(R, f, R, f).mean((1, 3)).astype(np.float16)           # soft area-mean label (exact in fp16)
        ds[R] = (di, dm)
    return i, img, msk, (lo, hi), ds


def cmd_prep(a):
    from numpy.lib.format import open_memmap
    out = a.out
    cache = out + "/cache"
    if os.path.exists(cache + "/PREP_DONE"):
        print("prep already done"); return
    root = train_root()
    names = sorted(os.path.basename(p) for p in glob.glob(root + "/Original/*.png"))
    gnames = sorted(os.path.basename(p) for p in glob.glob(root + "/Ground truth/*.png"))
    assert names == gnames and len(names) == 600, (len(names), len(gnames))
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(names))
    val = sorted(names[j] for j in perm[:N_VAL]); val_set = set(val)
    tr = [n for n in names if n not in val_set]
    assert len(tr) == 500 and len(val) == 100 and not (set(tr) & val_set)
    if a.limit_train: tr = tr[:a.limit_train]
    if a.limit_val: val = val[:a.limit_val]
    tmp = cache + ".tmp"
    os.makedirs(tmp, exist_ok=True)
    json.dump({"split_seed": SPLIT_SEED, "train": tr, "val": val, "source": "FIVES official train/ only"},
              open(out + "/split.json" if not (a.limit_train or a.limit_val) else tmp + "/split.json", "w"), indent=1)
    if not (a.limit_train or a.limit_val):
        open(out + "/val_list.txt", "w").write("\n".join(val) + "\n")
    with mp.get_context("fork").Pool(16) as pool:
        for tag, lst in (("train", tr), ("val", val)):
            n = len(lst)
            I = open_memmap(f"{tmp}/{tag}_img.npy", "w+", np.uint8, (n, SIZE, SIZE, 3))
            M = open_memmap(f"{tmp}/{tag}_msk.npy", "w+", np.uint8, (n, SIZE, SIZE))
            LH = np.zeros((n, 2), np.float32)
            DI = {R: open_memmap(f"{tmp}/{tag}_img_{R}.npy", "w+", np.float16, (n, R, R, 3)) for R in RES}
            DM = {R: open_memmap(f"{tmp}/{tag}_msk_{R}.npy", "w+", np.float16, (n, R, R)) for R in RES}
            tasks = [(i, f"{root}/Original/{nm}", f"{root}/Ground truth/{nm}") for i, nm in enumerate(lst)]
            t0 = time.time()
            for i, img, msk, lh, ds in pool.imap_unordered(_prep_one, tasks):
                I[i] = img; M[i] = msk; LH[i] = lh
                for R in RES:
                    DI[R][i], DM[R][i] = ds[R]
                if i % 50 == 0: print(tag, i, n, "%.0fs" % (time.time() - t0), flush=True)
            np.save(f"{tmp}/{tag}_lohi.npy", LH)
            json.dump(lst, open(f"{tmp}/{tag}_names.json", "w"))
            I.flush(); M.flush()
            for R in RES: DI[R].flush(); DM[R].flush()
            del I, M, DI, DM
    open(tmp + "/PREP_DONE", "w").write(time.strftime("%F %T"))
    os.rename(tmp, cache)
    print("prep done")


# ------------------------------------------------------------------ training --------------------------------------
import torch.utils.data as _tud


class CropStream(_tud.IterableDataset):
    """Infinite stream of augmented (3,CROP,CROP) crops. native: from 2048 uint8 cache; rs R: from the area-downscaled cache."""
    def __init__(self, cache, R, seed, n_limit=None):
        self.cache, self.R, self.seed, self.n_limit = cache, R, seed, n_limit

    def __iter__(self):
        import torch
        wi = torch.utils.data.get_worker_info()
        rng = np.random.RandomState(self.seed * 7919 + (wi.id if wi else 0))
        c = self.cache
        if self.R is None:
            I = np.load(c + "/train_img.npy", mmap_mode="r"); M = np.load(c + "/train_msk.npy", mmap_mode="r")
            LH = np.load(c + "/train_lohi.npy"); S = SIZE
        else:
            I = np.load(f"{c}/train_img_{self.R}.npy", mmap_mode="r"); M = np.load(f"{c}/train_msk_{self.R}.npy", mmap_mode="r"); S = self.R
        n = I.shape[0] if not self.n_limit else min(self.n_limit, I.shape[0])
        cs = min(CROP, S)
        while True:
            i = rng.randint(n)
            y = rng.randint(S - cs + 1); x = rng.randint(S - cs + 1)
            im = np.asarray(I[i, y:y + cs, x:x + cs]).astype(np.float32)
            mk = np.asarray(M[i, y:y + cs, x:x + cs]).astype(np.float32)
            if self.R is not None: mk = (mk >= 0.5).astype(np.float32)        # area-downscaled label thresholded at 0.5
            if self.R is None:
                lo, hi = LH[i]
                im = (im - lo) / (hi - lo) if hi > lo else np.zeros_like(im)
            if rng.rand() < 0.5: im, mk = im[:, ::-1], mk[:, ::-1]           # horizontal flip
            if rng.rand() < 0.5: im, mk = im[::-1], mk[::-1]                 # vertical flip
            k = rng.randint(4)
            if k: im, mk = np.rot90(im, k, (0, 1)), np.rot90(mk, k, (0, 1))   # rot90
            yield (torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))),
                   torch.from_numpy(np.ascontiguousarray(mk[None])))


def cuda_ok(dev):
    return dev.type == "cuda"


def val_dice(model, a, dev, cache, R):
    """Mean per-image Dice on the 100 train-internal val images at the training's own evaluation resolution:
    native -> 2048^2 via non-overlapping 256 tiles (the training window); rs R -> R^2 whole image, label = area-downscaled GT >= 0.5."""
    import torch
    model.eval()
    ds = []
    if R is None:
        vimg = np.load(cache + "/val_img.npy", mmap_mode="r"); lh = np.load(cache + "/val_lohi.npy"); vm = np.load(cache + "/val_msk.npy", mmap_mode="r")
        win = gauss_win(CROP, dev)
        for i in range(vimg.shape[0]):
            lo, hi = lh[i]
            x = (np.asarray(vimg[i]).astype(np.float32) - lo) / (hi - lo) if hi > lo else np.zeros((SIZE, SIZE, 3), np.float32)
            x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
            prob, _ = tile_infer(model, a, x, CROP, 0, 16, False, dev, win)
            P = prob > 0.5; G = np.asarray(vm[i]).astype(bool)
            ds.append(2.0 * (P & G).sum() / max(P.sum() + G.sum(), 1))
    else:
        vimg = np.load(f"{cache}/val_img_{R}.npy", mmap_mode="r"); vm = np.load(f"{cache}/val_msk_{R}.npy", mmap_mode="r")
        for i in range(vimg.shape[0]):
            x = torch.from_numpy(np.ascontiguousarray(np.asarray(vimg[i]).astype(np.float32).transpose(2, 0, 1)))
            prob, _ = whole_infer(model, a, x, False, dev)
            P = prob > 0.5; G = np.asarray(vm[i]).astype(np.float32) >= 0.5
            ds.append(2.0 * (P & G).sum() / max(P.sum() + G.sum(), 1))
    model.train()
    return float(np.mean(ds))


def train(a, model_name, variant, args, out, dev):
    import torch
    from torch.utils.data import DataLoader
    from common.losses import seg_loss
    cache = args.out + "/cache"
    R = None if variant == "native" else int(variant[2:])
    torch.manual_seed(SEED); np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    model = a.build(copy.deepcopy(a.DEFAULT_CFG)).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=ETA_MIN)
    start, hist, t_prev = 0, [], 0.0
    vcurve, best, best_step = [], -1.0, 0
    ti = out + "/train_info.json"
    if os.path.exists(ti) and os.path.exists(out + "/best.pth"):
        print("[train] reusing finished training", flush=True)
        return json.load(open(ti))
    ck = out + "/ckpt_last.pth"
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        start, hist, t_prev = s["step"], s["hist"], s["elapsed"]
        vcurve, best, best_step = s["vcurve"], s["best"], s["best_step"]
        print(f"[resume] from step {start}", flush=True)
    ds = CropStream(cache, R, SEED + start, args.n_train_limit)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers, pin_memory=cuda_ok(dev),
                    prefetch_factor=4 if args.workers > 0 else None)
    it = iter(dl)
    model.train()
    t0 = time.time(); acc = 0.0; nacc = 0; nskip = 0
    for step in range(start, args.steps):
        x, y = next(it)
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        o = a.forward_train(model, x)
        loss = seg_loss(o["out"], o.get("side"), y) + a.extra_loss(o, y, a.DEFAULT_CFG)
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        if bool(torch.isfinite(gn)): opt.step()
        else: nskip += 1; print(f"[warn] non-finite grad norm at step {step + 1}: update skipped ({nskip} so far)", flush=True)
        sched.step()
        acc += loss.detach().float(); nacc += 1
        if (step + 1) % 200 == 0 or step + 1 == args.steps:
            lv = float(acc) / nacc; acc = 0.0; nacc = 0
            if not np.isfinite(lv): raise FloatingPointError(f"non-finite loss at step {step + 1}")
            el = t_prev + time.time() - t0
            hist.append({"step": step + 1, "loss": lv, "elapsed_s": el, "lr": sched.get_last_lr()[0]})
            print(f"[step {step + 1}/{args.steps}] loss={lv:.4f} elapsed={el:.0f}s", flush=True)
        if (step + 1) % 1000 == 0 or step + 1 == args.steps:
            vd = val_dice(model, a, dev, cache, R)
            vcurve.append({"step": step + 1, "val_dice": vd})
            if vd > best:
                best, best_step = vd, step + 1
                torch.save(model.state_dict(), out + "/best.pth")
            print(f"[val] step {step + 1} dice={vd:.4f} best={best:.4f}@{best_step}", flush=True)
        if (step + 1) % 1000 == 0 and step + 1 < args.steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step + 1,
                        "hist": hist, "elapsed": t_prev + time.time() - t0, "vcurve": vcurve, "best": best,
                        "best_step": best_step}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
    train_s = t_prev + time.time() - t0
    torch.save(model.state_dict(), out + "/final.pth")
    del it, dl, opt, sched, model
    gc.collect()
    if cuda_ok(dev): torch.cuda.empty_cache()
    tinfo = {"n_params": n_params, "train_seconds": train_s, "history": hist, "steps": args.steps, "resumed_from": start,
             "val_curve": vcurve, "skipped_nonfinite_updates": nskip, "best_val_dice": best, "best_step": best_step,
             "selection": "best val Dice (mean per image) at own evaluation resolution, evaluated every 1000 steps",
             "val_resolution": "2048 via non-overlapping 256 tiles" if R is None else f"{R} whole image"}
    json.dump(tinfo, open(out + "/train_info.json", "w"))
    if os.path.exists(ck): os.remove(ck)
    return tinfo


# ------------------------------------------------------------------ inference ------------------------------------
def gauss_win(p, dev):
    import torch
    ax = torch.arange(p, dtype=torch.float32) - (p - 1) / 2.0
    g = torch.exp(-ax ** 2 / (2 * (p / 8.0) ** 2))                          # nnU-Net style: sigma = patch/8
    w = g[:, None] * g[None, :]
    return (w / w.max()).to(dev)


def tile_infer(model, a, x, p, ov, tbs, amp, dev, win):
    """x: (3,2048,2048) float32 CPU. Gaussian-weighted average of tile probabilities. Returns (prob np float32, finite_flag)."""
    import torch
    H = W = SIZE
    stride = p * (100 - ov) // 100
    coords = [(y, xx) for y in range(0, H - p + 1, stride) for xx in range(0, W - p + 1, stride)]
    acc = torch.zeros(H, W, device=dev); wacc = torch.zeros(H, W, device=dev)
    bad = torch.zeros((), dtype=torch.bool, device=dev)
    with torch.no_grad():
        for i in range(0, len(coords), tbs):
            c = coords[i:i + tbs]
            xb = torch.stack([x[:, y:y + p, xx:xx + p] for y, xx in c]).to(dev)
            if amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    lg = a.infer(model, xb)
            else:
                lg = a.infer(model, xb)
            pr = torch.sigmoid(lg.float())[:, 0]
            bad |= ~torch.isfinite(pr).all()
            for j, (y, xx) in enumerate(c):
                acc[y:y + p, xx:xx + p] += pr[j] * win; wacc[y:y + p, xx:xx + p] += win
        res = (acc / wacc.clamp_min(1e-12)).cpu().numpy()
    return res, not bool(bad)


def whole_infer(model, a, x, amp, dev):
    """Direct whole-image forward. x: (3,H,W) float32 CPU. Returns prob (H,W) np float32 and finite flag."""
    import torch
    with torch.no_grad():
        xb = x[None].to(dev)
        if amp:
            with torch.autocast("cuda", dtype=torch.float16):
                lg = a.infer(model, xb)
        else:
            lg = a.infer(model, xb)
        pr = torch.sigmoid(lg.float())[0, 0]
        fin = bool(torch.isfinite(pr).all())
        return pr.cpu().numpy(), fin


def measure(fn, dev):
    """Peak GPU memory of fn() in MiB (allocated = reproducible, reserved = descriptive). OOM -> {'oom': True}."""
    import torch
    if not cuda_ok(dev):
        fn(); return {"peak_alloc_mib": None, "note": "cpu run, no GPU memory"}
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated() / 2 ** 20
    try:
        fn(); torch.cuda.synchronize()
        r = {"peak_alloc_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
             "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2 ** 20, "weights_mib": base}
    except torch.cuda.OutOfMemoryError:
        r = {"oom": True, "weights_mib": base}
    gc.collect(); torch.cuda.empty_cache()
    return r


# ------------------------------------------------------------------ metrics (workers) -----------------------------
_W = {}
def _wk_init(valmsk):
    sys.path.insert(0, HERE)
    import g1
    _W["g1"] = g1
    _W["msk"] = np.load(valmsk, mmap_mode="r")
    _W["bands"] = {}
    cv2.setNumThreads(1)


def _band(s):
    b = _W["bands"].get(s)
    if b is None:
        # pixel is in the band iff it lies within BAND px of some interior seam line k*s (0 < k*s < SIZE)
        line = np.zeros(SIZE, bool)
        for k in range(s, SIZE, s):
            line[max(k - BAND, 0):min(k + BAND + 1, SIZE)] = True
        b = line[None, :] | line[:, None]
        _W["bands"][s] = b
    return b


def _region_cl(g1, SG, SP, G, P, M):
    nSG = int((SG & M).sum())
    if nSG == 0: return None, 0
    nSP = int((SP & M).sum())
    a = int((SP & M & G).sum()); b = int((SG & M & P).sum())
    tp = a / nSP if nSP else 0.0
    return g1.f1(tp, b / nSG), nSG


def eval_task(args):
    from skimage.morphology import skeletonize
    idx, packed = args
    g1 = _W["g1"]
    G = np.asarray(_W["msk"][idx]).astype(bool)
    gt = g1.prep_gt(G, None)
    out = {}
    for vn, pb in packed.items():
        P = np.unpackbits(pb)[:SIZE * SIZE].reshape(SIZE, SIZE).astype(bool)
        d = g1.fin(g1.counts(gt, P))                         # identical to G1: dice, cldice, thin3/thin5 clDice
        SP = skeletonize(P) > 0
        seam = {}
        for s in GRIDS:
            B = _band(s)
            cb, nb = _region_cl(g1, gt["SG"], SP, G, P, B)
            ci, ni = _region_cl(g1, gt["SG"], SP, G, P, ~B)
            seam[str(s)] = {"band": cb, "n_band": nb, "interior": ci, "n_int": ni}
        d["seam"] = seam
        out[vn] = d
    return idx, out


# ------------------------------------------------------------------ run ------------------------------------------
def cmd_run(args):
    import torch
    sys.path.insert(0, CODE_V2); sys.path.insert(0, HERE)
    from models import registry
    out = f"{args.out}/{args.model}/{args.variant}"
    os.makedirs(out, exist_ok=True)
    cache = args.out + "/cache"
    assert os.path.exists(cache + "/PREP_DONE"), "run prep first"
    dev = torch.device(args.device)
    # metric pool is forked before any CUDA use
    pool = None if args.train_only else mp.get_context("fork").Pool(NPOOL, initializer=_wk_init, initargs=(cache + "/val_msk.npy",))
    a = registry.get(args.model)
    info = {"model": args.model, "variant": args.variant, "seed": SEED, "steps": args.steps, "batch": args.batch,
            "crop": CROP, "smoke": args.smoke, "device": args.device, "torch": torch.__version__,
            "started": time.strftime("%F %T")}
    tr = train(a, args.model, args.variant, args, out, dev)
    info.update(tr)
    if args.train_only:
        print("[done] train-only", args.model, args.variant, flush=True); return
    # ---- fresh model for evaluation (clean memory accounting)
    model = a.build(copy.deepcopy(a.DEFAULT_CFG))
    model.load_state_dict(torch.load(out + "/best.pth", map_location="cpu"))      # best-val-Dice checkpoint
    model.eval()
    mdev = copy.deepcopy(model).to(dev).eval()
    vimg = np.load(cache + "/val_img.npy", mmap_mode="r"); vlh = np.load(cache + "/val_lohi.npy")
    names = json.load(open(cache + "/val_names.json"))
    nval = len(names)
    R = None if args.variant == "native" else int(args.variant[2:])
    rimg = np.load(f"{cache}/val_img_{R}.npy", mmap_mode="r") if R else None
    amp_ok = cuda_ok(dev)

    def get_full(i):
        lo, hi = vlh[i]
        x = (np.asarray(vimg[i]).astype(np.float32) - lo) / (hi - lo) if hi > lo else np.zeros((SIZE, SIZE, 3), np.float32)
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))

    def get_rs(i):
        return torch.from_numpy(np.ascontiguousarray(np.asarray(rimg[i]).astype(np.float32).transpose(2, 0, 1)))

    probes = {}
    runs = {}      # variant -> (callable(i) -> (prob, finite), callable_mem())
    modes = {}
    if R is None:
        # --- OOM-boundary probe (batch 1, whole-image forward on random input), fp32 and AMP
        for prec in ("fp32", "amp"):
            if prec == "amp" and not amp_ok: continue
            probes[prec] = {}
            dead = False
            for s in (PROBE_SIZES if cuda_ok(dev) else (256,)):
                if dead:
                    probes[prec][str(s)] = {"oom": True, "inferred": True}; continue
                xs = torch.rand(3, s, s)
                r = measure(lambda: whole_infer(mdev, a, xs, prec == "amp", dev), dev)
                probes[prec][str(s)] = r; dead = bool(r.get("oom"))
                print("[probe]", prec, s, r, flush=True)
        # --- tiling variants
        for p in PATCHES:
            win = gauss_win(p, dev)
            for ov in OVERLAPS:
                for prec in ("fp32", "amp"):
                    if prec == "amp" and not amp_ok: continue
                    vn = f"tile_p{p}_o{ov}_{prec}"
                    f = (lambda x, p=p, ov=ov, prec=prec, win=win, tb=TBS[p]:
                         tile_infer(mdev, a, x, p, ov, tb, prec == "amp", dev, win))
                    fm = (lambda x, p=p, ov=ov, prec=prec, win=win:
                          tile_infer(mdev, a, x, p, ov, 1, prec == "amp", dev, win))
                    runs[vn] = (f, fm)
        # --- full-resolution direct inference
        full_fp32_oom = bool(probes.get("fp32", {}).get(str(SIZE), {}).get("oom"))
        if cuda_ok(dev) and not full_fp32_oom:
            runs["full_fp32"] = ((lambda x: whole_infer(mdev, a, x, False, dev)), (lambda x: whole_infer(mdev, a, x, False, dev)))
            modes["full_fp32"] = "gpu"
        else:
            def f_cpu(x):
                torch.set_num_threads(CPU_THREADS)
                return whole_infer(model, a, x, False, torch.device("cpu"))
            runs["full_fp32"] = (f_cpu, f_cpu if not cuda_ok(dev) else None)
            modes["full_fp32"] = "cpu_fallback" if cuda_ok(dev) else "cpu"
        if amp_ok and not bool(probes.get("amp", {}).get(str(SIZE), {}).get("oom")):
            runs["full_amp"] = ((lambda x: whole_infer(mdev, a, x, True, dev)), (lambda x: whole_infer(mdev, a, x, True, dev)))
            modes["full_amp"] = "gpu"
        getter = get_full
    else:
        vn = f"rs{R}_fp32"
        runs[vn] = ((lambda x: whole_infer(mdev, a, x, False, dev)), (lambda x: whole_infer(mdev, a, x, False, dev)))
        modes[vn] = "gpu" if cuda_ok(dev) else "cpu"
        getter = get_rs

    # --- batch-1 peak-memory measurement (image 0) for every variant
    mem = {}
    x0 = getter(0)
    for vn, (f, fm) in runs.items():
        if fm is None:
            mem[vn] = {"oom": True, "note": "GPU fp32 full-res OOM; evaluated on CPU (non-executable on any GPU budget)"}
        else:
            mem[vn] = measure(lambda: fm(x0), dev)
        print("[mem]", vn, mem[vn], flush=True)
    if R is None and "full_fp32" in mem and cuda_ok(dev) and modes.get("full_fp32") == "cpu_fallback":
        mem["full_fp32"]["oom"] = True
    # a variant that already OOMs at batch 1 is not evaluated on GPU (resize variants only; native tile variants are small)
    skip = {vn for vn in runs if mem[vn].get("oom") and modes.get(vn) != "cpu_fallback"}
    # --- per-image metrics
    pending = []; fin_cnt = {vn: 0 for vn in runs}; tsec = {vn: 0.0 for vn in runs}
    t0 = time.time()
    for i in range(nval):
        x = getter(i)
        packed = {}
        for vn, (f, fm) in runs.items():
            if vn in skip: continue
            t1 = time.time()
            try:
                prob, fin = f(x)
            except torch.cuda.OutOfMemoryError:
                skip.add(vn); mem[vn]["oom_during_eval"] = True; torch.cuda.empty_cache(); continue
            tsec[vn] += time.time() - t1
            if R:                                                       # upsample prediction to native resolution
                prob = cv2.resize(prob, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
            fin_cnt[vn] += int(not fin)
            packed[vn] = np.packbits(prob > 0.5)
        pending.append(pool.apply_async(eval_task, ((i, packed),)))
        while sum(1 for p_ in pending if not p_.ready()) > 16:
            time.sleep(0.5)
        if i % 10 == 0: print(f"[eval] {i + 1}/{nval} {time.time() - t0:.0f}s", flush=True)
    per = {vn: [] for vn in runs if vn not in skip}
    for p_ in pending:
        idx, o = p_.get()
        for vn, d in o.items():
            d["idx"] = idx; d["name"] = names[idx]
            per[vn].append(d)
    pool.close(); pool.join()
    info.update({"finished": time.strftime("%F %T"), "n_val": nval, "probes": probes, "memory": mem, "modes": modes,
                 "nonfinite_images": fin_cnt, "infer_seconds": tsec, "per_image": per, "skipped_oom": sorted(skip),
                 "gpu": torch.cuda.get_device_name(0) if cuda_ok(dev) else None,
                 "protocol": {"split_seed": SPLIT_SEED, "n_val": N_VAL, "patches": PATCHES, "overlaps": OVERLAPS,
                              "grids": GRIDS, "band": BAND, "res": RES}})
    json.dump(info, open(out + "/results.json.tmp", "w"))
    os.replace(out + "/results.json.tmp", out + "/results.json")
    print("[done]", args.model, args.variant, flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep"); p.add_argument("--out", default=OUT_DEFAULT)
    p.add_argument("--limit-train", type=int, default=0); p.add_argument("--limit-val", type=int, default=0)
    r = sp.add_parser("run"); r.add_argument("--model", required=True)
    r.add_argument("--variant", required=True, choices=["native"] + [f"rs{x}" for x in RES])
    r.add_argument("--out", default=OUT_DEFAULT); r.add_argument("--device", default="cuda:0")
    r.add_argument("--steps", type=int, default=STEPS); r.add_argument("--batch", type=int, default=BATCH)
    r.add_argument("--workers", type=int, default=4); r.add_argument("--n-train-limit", type=int, default=0)
    r.add_argument("--smoke", action="store_true"); r.add_argument("--train-only", action="store_true")
    a = ap.parse_args()
    if a.cmd == "prep": cmd_prep(a)
    else: cmd_run(a)


if __name__ == "__main__":
    main()
