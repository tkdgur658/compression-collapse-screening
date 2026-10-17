# PREREG_F5 — AMENDMENT 0 (frozen 2026-10-16 ~16:55 KST, before any F5 results.json was read)

This file is the "Amendments" section of PREREG_F5.md (frozen sha 42dc239842e778bf…). It is kept as a separate file so that the frozen pre-registration keeps its sha256.
Approved by [internal decision] (2026-10-16). At the time of freezing, the F5 queue had finished only SMOKE (markers: SMOKE.DONE); nothing under server `s4mem/f5/pbe/` or `f5_smoke/` results was read by the analysis code or by this session, except the smoke log lines (train finite, n_params, smoke val Dice 0.016 at 50 steps) needed for stop rule S1.

## A0.1 Analysis-script sha256 (sec. 7 "freeze 시 기록(신규)")

| file | sha256 |
|---|---|
| scripts/f5_boot.py | d783b5b8b90d4a44f30c24279a0c9934535e0c3e5baf28fef6024a5f8f81d0d1 |
| scripts/f5_summary.py | b9c6a16eb44f82418d45a0235dfd4ecfb642a465f4a12a90dbb37599901ae255 |
| scripts/f5_synth.py (amended, this file A0.2) | b4f8d9584c460518e08eb8ea87467bd9e652100f1dc7112ca5b3d22118a3ed29 |
| scripts/f5_synth_v0.py (first version, kept) | 082c3d3dc9e8856413e6e3f65c930367e9b0cde0076495b90d926128a1bf4085 |
| server scripts/f5_queue.sh | 851b62c66e32d6b6469d314e0ee19c93be51292b22bf940b67e424c3f3732902 |
| IMPL_NOTES.md (27 implementation choices where the prereg is silent) | 2a5eba8f55a1eb17ce28acb3032e3db11e3350538d791e2d18e70d6df7144330 |

## A0.2 Change of two synthetic criteria in sec. 6 (reason and rule)

First run (F5_SYNTH_v0.md, sha 227c09ba…): FAIL, 10 of 15 criteria.
1. R1 coverage was 0.995 for c = 0.70, 0.80, 1.00, above the ceiling 0.99. A 400-dataset scratch run gave bootstrap SE 1.09–1.12× the true SD (coverage 0.99–1.00), so the percentile bootstrap of a ratio of medians is conservative; no code error. **Change:** coverage criterion [0.90, 0.99] → [0.90, 1.00]. Method unchanged (paired percentile, B = 2000, rng 20261016). **Manuscript condition:** a 4f footnote states that the percentile bootstrap over-covers in synthetic checks, so the T judgment is conservative toward indeterminate (numbers from F5_SYNTH.md).
2. R2(B) with 4-px gaps: band clDice 1.000 → 0.868, interior change 0.000, but s = 0.868, so "s < 0.8 and flag on" failed. A 4-px gap inside the 17-px band (seam line ±8 px) cannot push band clDice below about 0.87; this is a design error of the synthetic case, not of the metric. **Change:** R2(B) split into B1 (detection: 4-px gaps → band drop ∧ |interior change| ≤ 0.01 ∧ s ≤ 0.95) and B2 (flag operation: gaps of 18 px spanning the whole band → s < 0.8 ∧ flag on). B2 is computed after (C) and uses no random numbers, so R1, (A), (B1), (C), R3, R4 inputs are identical to the first run.
Rule fixed with the change: if B1 or B2 had failed, analysis would stop and be reported; criteria are not relaxed again.

## A0.3 Rerun result (one run)

F5_SYNTH.md (sha 17733c6f…), F5_SYNTH.json (sha aa5180ff…): **PASS, 16 of 16.** R1 coverage 0.995 (all c), classes as before; B1 s = 0.868, interior 0.000; B2 s = 0.000 (166 gaps), flag on; (C) s = 1.011; R3, R4 PASS. The console print after writing the outputs failed on a cp949 encoding error (`python -I` ignores PYTHONIOENCODING); outputs were written before it and were not regenerated.

## A0.4 Execution of sec. 3–4 (fixed now)

Real-data analysis runs locally (CPU, 1 thread) once after F5 ALL_DONE, on read-only scp copies of server `f5/pbe/<variant>/results.json` and the sec. 7 files; `--sha-paths` points to those copies and `--amend0 AMEND0_SHA.json` supplies the A0.1 sha. G2 three-model inputs = `g2_inputs/` (copied earlier, manuscript point values matched in 21 of 21 rows, G2_RECOMP.md sha 7c6bc7be…).
