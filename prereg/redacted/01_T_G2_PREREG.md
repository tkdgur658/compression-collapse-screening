Prediction: 50%-overlap tiling keeps >= 0.9x thin-vessel clDice of full-resolution inference with probability ~0.75 -> axis closes is the default expectation; survival only if seam-band connectivity breaks categorically.

# G2 pre-registration (FIVES, memory axis) — written 2026-10-14 BEFORE any G2 training or evaluation was run

Status of this file: Part A (first runs) is final. Part B (full S-set design, C rule, verdict mechanics) is appended below before any Part B evaluation is run; no training in Part A is evaluated on the S-set until Part B is frozen.

## Part A. First runs (shared prerequisites of every G2 branch) — no verdict is drawn from them alone

What: ONE U-Net-family model, `utanet` (registry `models.utanet`, default config; it is a ResNet34 U-Net, the TA-MoSC/MoE branch is off in the default config, 24.1 M parameters), trained twice with the same optimizer, step budget and seed:
1. `rs256` — whole-image training at 256x256: FIVES image area-downscaled by 8 (2048 -> 256), label area-downscaled by 8 and thresholded at 0.5.
2. `native` — native-resolution training with random 256x256 crops of the 2048x2048 image (uniform positions, label native).

Common settings (fixed now): FIVES official TRAIN only (600 images); 100 images held out as train-internal validation (`numpy.random.RandomState(20261014).permutation` of the sorted names, first 100; list saved at `g2a/val_list.txt` and `g2a/split.json`), 500 for training; the official FIVES test folder is never opened by any G2 code. Per-image min-max to [0,1] computed on the full 2048 image (crops and tiles use the same constants). Loss BCE + soft Dice (code_v2 `common.losses.seg_loss`), AdamW lr 1e-3, wd 1e-2, betas (0.9, 0.999), cosine to 1e-6 over 8000 steps, gradient-norm clip 5.0 (an update whose clipped gradient norm is non-finite is skipped and counted), batch 16, 8000 steps, augmentation horizontal flip, vertical flip, rot90 (each independent), seed 1, cudnn deterministic. Code: `scripts/g2_first_train.py` (frozen copy of `g2a_train.py` as of this write, sha256 prefix 910fb8608f9c1c55), launcher `scripts/g2_first.sh`, imported models from `<ARCHIVE>/code_v2` (not edited).

Checkpoint rule: every 1000 steps compute mean per-image Dice on the 100 validation images at the run's own evaluation resolution (`native`: 2048x2048 via non-overlapping 256 tiles, threshold 0.5; `rs256`: 256x256 whole image against the thresholded downscaled label, threshold 0.5); keep the best as `best.pth` (ties: earliest). `final.pth` is also kept. Later S-evaluations (0%/50% overlap sliding window, upsample variants) are made from `best.pth` of these same runs.

Prediction (stated before any run): (i) both runs reach validation Dice >= 0.80 at their own resolution (probability 0.85). (ii) When `rs256` is upsampled to 2048 its thin-vessel (k=3) clDice median stays below the G1 label-only ceiling plus 0.05, i.e. < 0.38 (probability 0.85). (iii) `native` tiled with 50% overlap reaches a thin-vessel clDice median above `rs256` upsampled by at least 0.15 (probability 0.8). These predictions are descriptive; they are not a gate.

Use: these two checkpoints are inputs to the Part B S-set (rows `rs256` and tiling/full-resolution of `native` for `utanet`). Part A by itself decides nothing: the C rule of Part B needs the full-resolution reference and the memory measurements of the same model.

Time estimate (no timing run allowed): the earlier protocol-v1 `utanet` run on ISIC17 took 42 min for ~10 300 steps including 100 per-epoch validations, i.e. <= 0.25 s/step; 8000 steps -> 20-33 min per run plus 8 validation passes (native: 100 images x 64 tiles, about 15-40 s each). Cache preparation (decoding 700 PNGs of 2048^2, writing ~11 GB of arrays under `g2a/cache`) 10-15 min once. Total Part A: about 1.0-1.5 h.

Outputs: `<WORK>/s4mem/g2a/utanet/{rs256,native}/{best.pth,final.pth,train_info.json}`, logs `g2a/logs/`, markers `g2a/markers/first__utanet__<variant>.DONE|.FAIL`, end marker `g2a/FIRST_DONE`.

Launch (by the user; GPU1): `nohup bash <WORK>/s4mem/scripts/g2_first.sh 1 > <WORK>/s4mem/g2a_first.launch.log 2>&1 < /dev/null &`

## Part B. Full S-set design (to be appended before any S-evaluation)

Written 2026-10-14, before any S-evaluation. Part A text above is unchanged by this append. Code: `scripts/g2a_train.py` (same training code path as `g2_first_train.py`; Part A runs are reused, not retrained, when `train_info.json` + `best.pth` exist), `scripts/g1.py` (G1 metric code, imported unchanged), `scripts/g2a_queue.sh`, `scripts/g2a_summary.py`.

### B1. Models (3 of the 11 registry models, default configs, imported from code_v2, not edited)
- `utanet` (U-Net family): ResNet34 U-Net, 24.1 M parameters; the "transformer attention" branch is off in the default config, so it is treated as a plain U-Net, not as the transformer-like model.
- `ucmnet` (lightweight): 47 k parameters, deep supervision (5 side outputs, weights 0.1..0.5 as in protocol v1).
- `maresunet` (attention / transformer-like): ResNet encoder with position attention (PAM, linear global attention over all H*W positions) and channel attention (CAM) on the encoder features, 26.3 M parameters. Its receptive field is the whole input, so tiling changes what the model sees; this is the registry model with global attention. If it cannot run a variant (OOM) that is recorded, not repaired.
Selection reason: one per requested role, chosen from parameter counts and source inspection before any FIVES result; no model was dropped or swapped after seeing results. Queue order ucmnet, utanet, maresunet.

### B2. Training (fixed)
Part A settings for every training. Variants per model: `native` (random 256 crops of the 2048 image) and `rs256`, `rs512`, `rs1024` (image and label area-downscaled by 8, 4, 2; label thresholded at 0.5; rs256 trains on the whole 256 image, rs512/rs1024 on random 256 crops of the downscaled image; inference on the whole downscaled image, prediction probability bilinearly upsampled to 2048, threshold 0.5). 4 variants x 3 models = 12 trainings, 8000 steps, seed 1, best.pth by validation Dice at the run's own resolution (Part A rule). Val Dice for selection and the reported val metrics use the same 100 images (small optimism, identical for every variant). Official FIVES test never opened.

### B3. S set (every variant is evaluated on the same 100 validation images at 2048x2048 against the native label)
1. Resize: `rs256_fp32`, `rs512_fp32`, `rs1024_fp32` (above).
2. Tiling from the `native` model, fp32: patch 256 and 512, overlap 0% and 50% (stride = patch x (1 - overlap)), Gaussian blending (nnU-Net style window, sigma = patch/8, normalised to max 1, probability averaging, threshold 0.5): `tile_p{256,512}_o{0,50}_fp32`. Patch-512 inference uses the weights trained on 256 crops (fully convolutional models; no 512-patch retraining) - a stated limitation; the robustness line in the summary excludes patch-512.
3. The same four with inference under `torch.autocast(float16)`: `tile_p*_o*_amp` (non-finite outputs are counted and reported, not repaired).
4. Direct whole-image 2048 inference of the native model: `full_fp32` (the unlimited reference) and `full_amp`. If `full_fp32` does not fit in the 24 GiB GPU (OOM), it is computed on CPU in fp32 and is marked non-executable at every budget. The OOM boundary is probed for sizes 256..2048 step 256, fp32 and AMP, batch 1 (table in the summary). Training-side gradient checkpointing is not used.

### B4. Peak memory and budgets
For every variant, peak GPU memory = `torch.cuda.max_memory_allocated` over one complete batch-1 inference of validation image 0 (tile batch 1; model weights fp32 resident; the 2048x2048 probability and weight accumulators resident on the GPU; input image on CPU and copied tile by tile; for resize and full variants the whole input is on the GPU). The CUDA context is not counted; reserved memory is stored as descriptive. Executable within budget B iff peak <= B, B in {2 GiB = 2048 MiB, 0.5 GiB = 512 MiB}. Metric passes use tile batch 8 (patch 256) / 4 (patch 512) with identical arithmetic.

### B5. Metrics
Identical code to G1 (`scripts/g1.py`: `prep_gt`, `counts`, `fin`): Dice, clDice (skimage skeleton), thin-vessel clDice k=3 (primary) and k=5, per image, medians over the 100 validation images (images without a GT skeleton of diameter <= k are excluded from that k-median, the same set for every variant). Tile-seam connectivity: seam lines at all multiples k x stride (0 < k x stride < 2048) with stride 128 / 256 / 512; band = pixels within +-8 px of a seam line, interior = the rest; clDice restricted to a region R: Tprec_R = |S_P in R and in V_G| / |S_P in R|, Tsens_R = |S_G in R and in V_P| / |S_G in R|, with skeletons computed on the whole image (not cut at seams); band and interior values are reported for every variant on all three grids, the full-resolution prediction on the same grid is the comparison. Seam break (descriptive flag) iff (band/interior of the tiled variant) / (band/interior of `full_fp32`, same grid) < 0.8 for medians over images.

### B6. Fixed C rule (mechanical; `g2a_summary.py` computes it)
ratio(v) = median thin3 clDice of variant v / median thin3 clDice of `full_fp32` of the same model.
- C holds at budget B for a model iff EVERY S variant executable within B has ratio < 0.8 (the executable set includes the full-resolution variants if they fit).
- C holds at B overall iff it holds for all three models. Holding at 2 GiB implies holding at 0.5 GiB.
- A model with an empty executable set at B is VACUOUS (flagged, counted as holding).
- If any model has an executable 50%-overlap tiling variant with ratio >= 0.8, C fails (the memory axis closes on FIVES). Verdict strings written by the summarizer: `C HOLDS`, `C FAILS (memory axis closes on FIVES)`, `INCOMPLETE`.
- The absolute values (thin3 clDice, clDice, Dice, peak memory) are reported before the ratios; the ratio rule is the only verdict. Seam flags and the patch-512-excluded robustness line are descriptive.
- Failed or missing jobs make the verdict provisional (`Jobs complete: NO`). No threshold, k, patch, overlap, budget or step count is changed after results are seen; implementation bugs are logged below as amendments.

### B7. Time estimate (no timing run was allowed; from protocol-v1 run times and model sizes)
Training (10 new runs after Part A): ucmnet 4 x 10-20 min, utanet 2 x 20-33 min, maresunet 4 x 15-27 min = 2.5-4.5 h. Evaluation: native eval per model 20-50 min (8 tiled passes + full-resolution, CPU fallback for full_fp32 likely for utanet and maresunet), resize evals 3-6 min each = 1.5-3 h. Cache prep (Part A) 10-15 min. Whole G2 about 5-8 h after Part A (Part A 1-1.5 h).

### Amendments
(none)
