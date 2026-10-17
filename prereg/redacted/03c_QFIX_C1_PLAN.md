# C_PLAN — QFIX stage C1 (PBE-Net compression-aware training), frozen before launch

Written 2026-10-15 (KST) before launching the training queues. Design = `PREREG_QFIX.md` §3 (sha256 1bac4879…4ea, unchanged) and `B_fix_design.md` §3, §5, §6 (B2 recipe, followed as written). Server root `BASE=<WORK>/qfix/C` (the study lead's brief puts the work under `qfix/C`, not the `qfix/C1` named in B §6.1; outputs go to `runs/<ds>/k<k>/`).

## 1. Code and changed files (sha256, fixed before launch; `BASE/SHA256SUMS`, checked by the queue at start)
| file | sha256 | note |
|---|---|---|
| `code/` | copy of `<ARCHIVE>/code` (commit e0e2f07, source digest 496cc2db… verified after copy, `__pycache__` excluded) | archive untouched |
| `code/common/trainer.py` (only changed file in the snapshot) | **852c7b0cab632d1e362025b0af5be0df8dc67d14b72d1fdb96e973ba263689b7** (original 3581da5f…17e3, kept as `rec/trainer.py.e0e2f07`) | new keyword `skip_test=False`. Default = protocol v1 byte-identical behaviour. `True` = no test loader, no test metric, `test: null`, `test_sealed: true` |
| `qfix_c_cat.py` (new, outside the snapshot) | **39703e2160497073677cdb0d0a441fbb6b6bf324091b321193049418b194d13c** | adapter `pbe_cat` + runner (B §6.1 item 2) |
| `rec/qfix_a.py` | **b36df82ffd238ccd2fadee1b02ab7df016e1bcc6500fc2c865c6adbc52fa94c6** | byte copy of the stage-A script, the source of the record operators. `qfix_c_cat.py` checks its sha and imports `rtn_sym_tensor`, `act_hook_factory`, `prune_layer` from it unchanged |
| `run_C1.sh` | **526ec0edb2e8337e6851cf285caaaa474246d69e7fa0dbb2c71be2e1bac30462** | queue |
| `qfix_c_smoke.py` | 6cf9c32e440187ae8edddb2240f2d03f13976fb760242ccfb94fc8e530cb2d6a | smoke + timing only |

The source digest recorded in each C1 `results.json` (`code.source_sha256`) therefore differs from 496cc2db (only trainer.py changed). `code.commit` stays e0e2f07.

### Implementation notes (where the brief left a choice)
- Operators: the record functions are called unchanged. STE form = `q + (x − x.detach())`. This equals B §3.2 `x + (q − x).detach()` in value and gradient, and its forward value is bit-exactly the record value `q` (smoke: training op output == record op output, max |Δ| = 0 for w4_pt, a6, l50).
- l50 uses the record `prune_layer` (CPU kthvalue), not a GPU kthvalue. Values are the same, and the cost was measured (1.07×).
- Sets: w4 = every floating parameter with dim ≥ 2 (62 tensors). a6 = output of every Conv/Linear (53 modules, train-only heads included). l50 = executed Conv/Linear weights (46, from an eval-mode zeros(1,3,256,256) forward hook). BN = every `_BatchNorm` (33, all BatchNorm2d).
- Mode RNG `random.Random(7919+k)`, drawn only from epoch 11 on. Epoch counter advances when `infer` was called (the per-epoch val evaluation). `cat_log.json` per run records mode counts per epoch.
- checkpoint = protocol v1 rule (max fp32 val mean IoU). The test split is sealed: `skip_test=True`, so the training queue never builds a test loader. fp32 test Dice is computed once by `eval_C1.py` after all 30 runs are DONE.

## 2. Exact command per run (cwd `BASE/code`)
```
CUDA_VISIBLE_DEVICES=<G> OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
taskset -c 0-7,12-31 python -B BASE/qfix_c_cat.py --ds <D> --k <K>
  >> BASE/logs/cat_<D>_k<K>.log 2>&1
```
`qfix_c_cat.py` runs a precheck first: all 3 original caches must exist and their names must equal the v1 run's `data_cache`, else exit 3, so no cache is written into the archive. It then calls `trainer.run(adapter, DEFAULT_CFG ∪ {cat}, D, K, BASE/runs/D/kK, DATA, CACHE, 'cuda:0', skip_test=True)` with seed = K. After training it verifies the run: status complete, model `pbe_cat`, cfg incl. CAT, protocol == v1, 100 history rows, data_cache == v1, params == v1, test sealed. Failure → exit 4.

Queues: `nohup bash run_C1.sh 0` (isic18 k1–5, cvc k1–5, PH2 k1–5) and `nohup bash run_C1.sh 1` (isic17 k1–5, kvasir k1–5, etis k1–5), `< /dev/null`, `flock BASE/gpu<G>.lock`. Markers in each run dir: `DONE` (exit 0), `FAIL` (precheck, or 2 attempts with non-zero rc), `MISMATCH` (verify failed, results kept). One automatic retry on a non-zero rc. A previous partial attempt is moved to `<run>_attempt<n>`, never deleted. `ALL_DONE_gpu<G>` is written when a queue ends.

## 3. Smoke (GPU 0, isic18 k1 train cache, `logs/smoke.json`)
- fp32 path vs original `pbe` forward_train on a fixed batch: outputs equal (bitwise), loss equal, BN buffers equal. Gradients are not bitwise equal, but original-vs-original gradients differ at the same scale (max rel 1.0e-3 vs 0.8e-3), so this is CUDA backward nondeterminism, not a code difference.
- 3 optimiser steps per mode: all losses and grad norms finite. BN running stats unchanged in every compressed step (w4_pt, a6, l50) and updated in fp32 steps. All BN back in train mode after each step.
- training op == record op (eval forward): w4_pt, a6, l50 bit-exact. Weights unchanged by the operator.

## 4. Timing gate (GPU 0, 50 timed steps/mode after 5 warm-up, trainer DataLoader, `logs/timing.json`)
| mode | ms/step | ratio |
|---|---|---|
| fp32 | 156.9 | 1.000 |
| w4_pt | 154.8 | 0.986 |
| a6 | 208.1 | 1.326 |
| l50 | 167.2 | 1.065 |

Factor = 0.1·1 + 0.9·mean(ratios) = **1.085**. Projection with B §5 base minutes (PH2 4.1, etis 5.2, cvc 16.3, kvasir 22.3, isic17 48.6, isic18 58.2 per run): training 13.99 + start-up 0.5 + Q/test evaluation 0.4 + smoke/timing 0.1 = **15.0 GPU-h ≤ 16 → launch**. Wall: GPU0 ≈ 7.4 h, GPU1 ≈ 7.1 h. Peak allocated memory 6.5 GB.

## 5. Evaluation (written, not run until 30 DONE): `BASE/eval/eval_C1.py`, `BASE/eval/run_eval.sh`
sha256 (`BASE/eval/EVAL_SHA256SUMS`): eval_C1.py 152694ecf7f4ea55cf053f97acad0117153c90f3dea1e11a1a5b94d71a8f1166, run_eval.sh e026b33cf6aa1710af912c31e1752d69886e1dcdd09c7715eef8899527f9acdd, B_baseline_18cells.json eaadb4cc7058974bdf4f34c5fb2dd5ad4cf77e62d913476551bd6b86268a5250. These were written after launch, before any C1 output existed.
Smoke on one existing v1 checkpoint (archive etis k1, GPU 1, `eval/smoke/smoke_check.json`):
- fp32 val and every w4/a6/l50 cell (all + 16 groups) are identical to the stage-A record (max |Δ| = 0). ckpt sha equal.
- fp32 test Dice 0.689721 vs v1 results.json 0.689708 (Δ 1.3e-5, GPU nondeterminism). Weights restored exactly.
- The decision code was run on the 30 stage-A records standing in for C1. It gave N = 0/18 (checkpoint level 6/90), AUC 0.978, Δ = 0 → (ii) pass, and the cell means match `B_baseline_18cells.json` (max |Δ| 3e-17). All of these equal the B §1 values.

## 6. Launch record
2026-10-15 22:13:40 KST: `run_C1.sh 0` (queue pid 4128303) and `run_C1.sh 1` (queue pid 4128321). First runs isic18 k1 (python pid 4128317, GPU 0) and isic17 k1 (python pid 4128334, GPU 1). Both logged `[ep 001]` and `[ep 002]` by 22:15:55. Warm-up epochs track the v1 history closely (isic18 k1 ep1 val IoU 0.6834 vs v1 0.6804; isic17 k1 0.7039 vs 0.7023). Exact equality is not expected because of CUDA backward nondeterminism.

See the docstring of `eval_C1.py`. It uses the record Engine from `rec/qfix_a.py`, unchanged. It computes fp32 val, fp32 test (once, with `common.metrics.evaluate`, the same function the v1 trainer used), the Q protocol, the 18-cell table, and criteria (i)–(iii) per B §6.4. Outputs go to `BASE/eval/C1_RESULT.md/json` and `C1_attribution.json`.
