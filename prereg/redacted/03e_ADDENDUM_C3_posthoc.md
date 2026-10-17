# PREREG_QFIX_ADDENDUM_C3 — 사후 튜닝(사전등록 밖)

Written 2026-10-16 04:27 KST, before any C3 training step. Server `BASE=<WORK>/qfix/C3`.
**Status: post hoc, outside the pre-registered branch.** `PREREG_QFIX.md` (sha256 1bac4879…4ea) and its C1 decision (C1_RESULT: TRADE-OFF, (ii) failed) stay unchanged. C3 does not replace or overturn them. Every C3 output carries the label "사후 튜닝(사전등록 밖)".

## 1. Design (one fixed design, study lead decision, not a grid)
C3 equals C1 (`B_fix_design.md` §3, `C_PLAN.md`) except for two changes. Each change sits behind an explicit flag of `qfix_c3_cat.py`, and the defaults reproduce C1.
1. `--mode-weights 5,1,1,1`: from epoch 11 the mode probabilities are fp32 5/8, w4_pt 1/8, a6 1/8, l50 1/8. **Interpretation recorded:** "compressed-batch ratio 1/4 → 1/8" is read as *each* compressed mode 1/8, so all compressed modes together get 3/8. The draw is `random.Random(7919+k).choices(modes, weights=[5,1,1,1])`.
2. `--final-fp32-epochs 10`: epochs 91–100 of the protocol-v1 schedule are fp32-only. fp32 batches run the original `pbe` forward_train with BN in train mode. BN running statistics therefore update normally and are re-estimated on fp32 batches.

Everything else is unchanged from C1: protocol v1 (100 epochs, batch 16, AdamW 1e-3/1e-2, cosine to 1e-6, clip 5, seed = k), the 10-epoch fp32 warm-up, the auxiliary loss, the record operators with STE, BN running-stat mode in compressed batches, checkpoint = max fp32 val IoU (v1 rule), and the test split sealed during training (`skip_test=True`). There are 30 runs (6 ds × 5 splits).

## 2. Criteria (fixed before launch; selection uses validation only; test is read once, by `eval_C3.py --decide`)
- **(i)** At least **14/18** cells (6 ds × {w4_pt, a6, l50}) have a 5-split mean val retention R = Dice_q/Dice_fp32 of 0.8 or more.
- **(ii)** fp32 test Dice drop vs v1 is at most 1 v1 split SD (one-sided, Δ_d ≥ −σ_d, σ = sample SD over the 5 v1 splits) on **at least 4/6** datasets.
- **(iii)** The label-free AUC (score = 1 − agree_all, label = R < 0.8, Mann–Whitney) is **0.95 or more**. The C1 fallback rule is unchanged: if a class has fewer than 5 cells, the pooled v1 + C3 180 cells are used.
- The conclusion mapping is the C1 mapping with the C3 thresholds, prefixed with the label. A C3 pass is reported only as a post-hoc result next to the pre-registered C1 outcome.

## 3. Files (sha256, frozen; `BASE/SHA256SUMS` is checked by every queue start, `BASE/eval/EVAL_SHA256SUMS`)
| file | sha256 | note |
|---|---|---|
| `qfix_c3_cat.py` | 5c8e747381483a6d007990d79fb249700b344728fc1691fa2f01a5af1f330fee | copy of C1 `qfix_c_cat.py` (39703e21…): BASE, `configure()` flags, `select_mode()`, verify of the fp32-only final epochs |
| `run_C3.sh` | be2cd16d74a34bf50a87c276452f15856ee85e7ea855dd29745aab063278df45 | copy of `run_C1.sh` (526ec0ed…): C3 flags, foreign-GPU-process wait, GPU-h budget guard |
| `run_C3_seq.sh` | f4e84730d7073b97ea6f7a9f10b80a2e98cd485f29835db37478f5a329ada299 | waiter: queue GPU1 first, then queue GPU0 |
| `qfix_c3_smoke.py` | 95f04d17522043ee3640a926a8464260ae55ec5264b10b6a96b6a1fbb0685d99 | C1 smoke + C3 checks |
| `c1_run_seconds.tsv` | ac234fcb6da93cf9c172afcd454ccd17443bf54a9a343be9e011f634321b5030 | C1 per-run wall seconds (from C1 queue logs), used by the budget guard |
| `eval/eval_C3.py` | 437160cf34c1ba389a2f32827e31f3d4238be9641254fe0e5d6391badbd742a2 | copy of `eval_C1.py` (152694ec…): paths, C3 CAT, criteria (i) ≥14, (ii) ≥4/6, label |
| `eval/run_eval.sh` | 32a1e41a6a0087d9296e7d636c2c8a84ede8646ff1b2ad61d4fc2fec07d8e28a | copy of C1 `run_eval.sh`, paths only |
| unchanged | `rec/qfix_a.py` b36df82f…c6c6, `code/common/trainer.py` 852c7b0c…89b7, `eval/B_baseline_18cells.json` eaadb4cc…5250 | byte copies from `qfix/C` |

## 4. Smoke (before launch; GPU 1)
- Training (`logs/smoke.json`): the fp32 path equals the original pbe forward (outputs, loss and BN buffers equal). Three steps per mode (fp32, w4_pt, a6, l50) give finite loss and grad norm. BN is unchanged in compressed steps and updated in fp32 steps. Training op equals record op bit-exactly for w4_pt, a6 and l50.
- 1000 draws with Random(7920): fp32 638 / w4 132 / a6 112 / l50 118 against 625/125/125/125, with every |z| < 1.3. These equal `random.choices(weights=5,1,1,1)` draw for draw.
- Schedule: epochs 1–10 fp32 only, 11–90 all modes, 91–100 fp32 only. Through the real `forward_train`, the val `infer` call moves the epoch from 90 to 91. Epoch 90 drew {fp32 14, w4 2, a6 2, l50 6}, epoch 91 drew fp32 24/24, and BN running stats updated at epoch 91.
- Eval (`eval/smoke/smoke_check.json`, v1 etis k1): all cells match stage A (max |Δ| = 0). Test Dice is 0.689721 vs 0.689708 in v1. The decision code on the v1 stand-in gives N = 0/18, AUC 0.978, Δ = 0, (ii) 6/6.

## 5. GPU budget (cumulative QFIX ≤ 20 GPU-h)
- Used: A 0.33 + C1 training 9.49 (measured from the C1 queue logs: GPU0 5.64 h, GPU1 3.85 h) + C1 eval 0.33 = **10.15**. C2 is counted at its bound of 1.0. C3 smoke used about 0.05.
- Mode-mix factor (C1 timing ratios w4 0.986, a6 1.326, l50 1.065): C1 1.085, C3 = 0.1 + 0.8·(5/8 + 3/8·1.126) + 0.1 = 1.038, giving a C3/C1 ratio of 0.957.
- Run in parallel like C1, C3 training would take 9.49 × 0.957 = 9.08 GPU-h, and the total would be 10.15 + 1.0 + 9.08 + 0.33 eval + 0.05 = **20.6 > 20**. **C3 therefore does not use the C1 parallel schedule.**
- Run sequentially (same per-GPU layout, one queue at a time): C1 shows a contention slowdown. isic18 k5 ran alone in 37.2 min, while k1–k4 took 59.5 min on average with the other GPU training (ratio 0.625). Applying that ratio to the C1 time that ran concurrently gives about 6.31 GPU-h for C3 training. With eval, the total is **≈ 17.8** (central estimate).
- Hard guard in `run_C3.sh`: a run starts only if the C3 training seconds used so far plus that run's C1 duration (a conservative estimate) stay within 8.47 GPU-h (= 20 − 10.15 − 1.0 − 0.33 − 0.05). Otherwise the guard writes `BUDGET_STOP_gpu<G>` and the queue ends.
