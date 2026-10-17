# PREREG_QDEPLOY ADDENDUM W8A8 — one added cell type on the deployment graph (FROZEN before the run)

Written 2026-10-16 ([internal session]) on [internal decision] decision ("QDEPLOY ADDENDUM — w8a8 셀 추가, 추론만, GPU0, cap 1 GPU-h, 사전등록 한 쪽, 결과 1회, Q 표에 열 추가"). Parent: PREREG_QDEPLOY.md (frozen sha 573b72eb…), result RESULT_QDEPLOY.md (decide run once, 2026-10-16). Nothing of this addendum has been run when this file is frozen.

## What is added
- One setting **D_w8a8** on the deployment graph (RepConv merged, BN folded, the same F2/F2b fusion and FOLD_PAIRS.json as F3): weights per-output-channel symmetric RTN 8-bit (`rtn_sym_channel`, bits 8, same axis rule as D_w4) **and at the same time** dynamic per-tensor symmetric 8-bit fake-quantization of every Conv/Linear output (`qfix_a.act_hook_factory(torch, 8)`, the D_a6 hook with 8 bits). Both are active in one forward pass.
- Same 330 checkpoints (ITEMS.json, MANIFEST_330.sha256 unchanged; manifest check before each item), same validation and 8 test images, same fp32 references, batch 16 (4 after one OOM).
- Code: `qdeploy_w8a8/qdeploy.py` = byte copy of the frozen `qdeploy/qdeploy.py` with only these changes: BASE path, SETTINGS = ['D_w8a8'], SPEC entry ('dep', 'w8a8'), the operator branch `w8a8` (integrity: max |weight level| ≤ 127, number of hooked modules recorded), compute cap 1 GPU-h, progress print line. Diff recorded at freeze in `W8A8_DIFF.txt`.

## Label and outputs (unchanged definitions)
- Per checkpoint: R = val Dice (D_w8a8) / val Dice (fp32, deployment graph). Cell (model, dataset) = mean R over 5 splits. **L80**: collapsed iff mean R < 0.8; L90 reported too. Exclusions: same rules as F3 (FAIL markers, manifest mismatch, old FAILED.txt rule, E1 failure → excluded from D/S).
- Reported once: collapsed cells all / PBE-Net / baselines (of 66 / 6 / 60), checkpoint-level count, list of collapsed cells; if ≥ 1 collapsed checkpoint, AUC of output agreement vs foreground-ratio and SQNR controls with the F3 functions (split mean and pooled); otherwise "no positives, AUC undefined".

## Prediction (descriptive, not a gate)
- P-W1: L80 collapse in at most 2 of 66 cells (p = 0.8). P-W2: PBE-Net 0 of 6 at L80 (p = 0.7).

## Stop rules and result once
- GPU 0 only, after checking no other process is on it; wait while a foreign process is present. Cap 1 GPU-h of compute; on reaching it, stop and report partial counts as provisional. Second OOM → stop.
- The analysis runs once after ALL_DONE (`run_decide_w8a8.sh` refuses before ALL_DONE and if a result exists). No change of bits, granularity, threshold, or checkpoints after seeing results.
- Manuscript: one added column / row "D w8a8" in Table tab:collapse_deploy and one sentence in 4g, numbers from RESULT_W8A8.json only.
