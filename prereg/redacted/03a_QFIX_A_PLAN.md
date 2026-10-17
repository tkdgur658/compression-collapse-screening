# A_PLAN — QFIX stage A (collapse attribution), frozen before any stage-A measurement

Written 2026-10-16, before the synthetic check and the 30-checkpoint run.  Design = PREREG_QFIX.md §1 (sha256 1bac4879…94ea, unchanged).
Script: `qfix/A/qfix_a.py` (server copy `<WORK>/qfix/A/qfix_a.py`), **sha256 b36df82ffd238ccd2fadee1b02ab7df016e1bcc6500fc2c865c6adbc52fa94c6**.
Only a smoke test was run before freezing (etis k1: module counts, integrity counters, wall time; no Dice/retention printed).

## 1. Assets
30 PBE-Net v1 checkpoints (registry `pbe`, DEFAULT_CFG, commit e0e2f07): 20 under `<archive>_archive/pbe_seg/runs/main/pbe/<ds>/k<k>` with `status: complete` (isic17 k2 FAILED.txt skipped) + 10 under `<WORK>/pbe_v1_fill/runs/pbe`. Val/test caches = each results.json `data_cache`. Server code copied to `qfix/A/code` (read-only source untouched).

## 2. Component groups (complete partition of PBENet, `comp_of()` raises on any unmapped name)
| group | modules (net.py names) | pre-registered? |
|---|---|---|
| stem | `base.encoder1.0.0` (1×1 conv 3→c0), `base.encoder1.0.1` (RepConv), `base.encoder1.0.3` (PRCM) | yes: input stem |
| feat | `base.encoder1.0.2`, `base.encoder2.0.2` (eBPB) | yes: feature block(s) |
| skip1, skip2, skip3 | `proj1`, `proj2`, `proj3` (predictive skip W_i, one group per stage) | yes: predictive skip per stage |
| bhead | `shead1-3` (boundary heads) + `base.gt_conv1-4` (deep-supervision heads) | yes: boundary head(s) — **train-only, not executed at inference**; their single-component loss is 0 by construction (kept as a check) |
| final | `base.final` | yes: final 1×1 head |
| enc2…enc5 | `base.encoder2` (minus eBPB), `base.encoder3`, `base.encoder4`, `base.encoder5` | added: remaining backbone, needed so that the groups partition the "all-components" setting |
| bott | `base.bottleneck` | added |
| dec1…dec4 | `base.decoder1`…`base.decoder4` | added |
The backbone is split at stage granularity (not one "rest" group) so that a single large rest group cannot satisfy H_A trivially.

## 3. Operators and metrics (record conventions)
- w4_pt: per-tensor symmetric RTN, 4 bit, on every parameter with dim ≥ 2 of the group (incl. PRCM `basis`), 9101 `rtn_sym_tensor`.
- a6: dynamic per-tensor symmetric 6-bit fake-quant on the OUTPUT of every Conv/Linear module of the group (9101 hook). Batch 16 with the amax taken per sample = the record's batch-1 per-forward scale.
- l50: layer-wise 50 % unstructured magnitude pruning of every executed Conv/Linear weight of the group (9601 `prune_layer`, executed set from forward hooks on the first val image).
- "all": the same operator on every group simultaneously (= the record's whole-network setting).
- Threshold sigmoid > 0.5, FP32 (TF32 off, cudnn deterministic), one GPU (CUDA_VISIBLE_DEVICES=1).
- Dice per image = 2TP/(2TP+FP+FN), 1 if both empty (record IoU convention u = 0 → 1); val Dice = mean over the run's val split. IoU kept as well.
- retention R = Dice_q / Dice_fp32 (val). Label-free agreement = mean over the first 8 images of the run's test cache of Dice(fp32 mask, q mask) (masks key not used for this).

## 4. Diagnostics (FP32, the same 8 test images, per group; only groups with executed modules)
- act_max = max over the group's Conv/Linear outputs of max|y|.
- ch_ratio = max over modules of (max_c a_c / median_c a_c), a_c = per-output-channel max|y| over images and space (median floored at 1e-6·max).
- dead_act_a6 = fraction of output channels with a_c < amax_module/(2·31) (rounded to zero by per-tensor a6); dead_act_zero = fraction with a_c = 0.
- w_range = max |w| over the group's dim ≥ 2 tensors; w_outlier = max over tensors of max|w|/rms(w); dead_w_w4 = fraction of output channels whose max|w_c| < max|w|/(2·7) (zeroed by per-tensor w4).
- Spearman (average ranks) between each diagnostic and the per-component loss, per setting, pooled over (run × executed group); scopes: all 90 cells and collapsed cells only. Descriptive; no pass rule.

## 5. H_A (decisive, frozen)
- Cell = (run, setting); 30 × 3 = 90 cells. **Collapsed cell: R_all < 0.8.**
- Per cell and group g: loss_g = max(0, 1 − R_g) (R_g = retention when only g is quantised/pruned).
- Pooled over the set C of collapsed cells (all settings together): L_g = Σ_{cells∈C} loss_g; T = Σ_g L_g; top-2 = the two groups with the largest L_g.
- **share = (L_top1 + L_top2) / T. H_A PASS iff share ≥ 0.70; else FAIL.** If T = 0 (no single-group loss at all) → FAIL (loss is pure interaction). If C is empty → NOT TESTABLE.
- Reported, not decisive: the same pooled share per setting (top-2 chosen per setting); mean over collapsed cells of the per-cell share carried by the pooled top-2; median per-cell share with the cell's own top-2; additivity = Σ_g loss_g / (1 − R_all) (median), which flags interaction-driven collapse.

## 6. Synthetic reverse check (frozen)
kvasir k1; planted component enc3: output channel 0 of `base.encoder3.0.1.conv_kxk` × 1000 with the following eval-mode BN (`bn_kxk`) running_mean × 1000 and running_var × 10⁶ — FP32 function unchanged, but a strong outlier in that one component. The planted model is the new FP32 reference; the full per-group sweep is rerun for w4_pt and a6. **PASS iff |ΔDice_fp32| < 1e-3 and the top-1 attributed group is enc3 under both w4_pt and a6.** (Gaussian noise that changes the FP32 function would move the reference itself, so a function-preserving outlier is the clean planted culprit.)

Result (filled after the run, one line): **frozen rule FAIL** — FP32 Dice unchanged (0.8856 → 0.8856), but top-1 stayed the natural culprit `stem` (w4 0.682, a6 0.805); planted enc3 ranked 2nd under w4 (loss 0.000 → 0.100) and 3rd under a6 (0.052 → 0.190). Post hoc (not part of the frozen rule): the planted-minus-clean loss change is non-zero ONLY for enc3 (w4 +0.100, a6 +0.138; every other group Δ = 0.000), i.e. the attribution is component-specific, but a planted effect smaller than the native stem loss cannot become top-1.
