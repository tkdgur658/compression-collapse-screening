# F5 — check of the sec. 8 predictions (descriptive, not a gate)

Source: F5_SUMMARY.md (generated once by f5_summary.py, 2026-10-16 19:33 KST, all sec. 7 sha OK, Jobs complete YES) and f5_inputs/pbe/*/train_info.json.

| # | prediction (p) | observed | hit |
|---|---|---|---|
| P1 | native best val Dice ≥ 0.80 (0.8) | 0.894 (step 8000); rs256 0.803, rs512 0.885, rs1024 0.903 | yes |
| P2a | tile_p256 fp32 peak ≤ 512 MiB (0.9) | 210 MiB | yes |
| P2b | tile_p512 fp32 peak ≤ 512 MiB (0.35) | 581 MiB → not in E (O4 rule) | no (the more likely outcome under p = 0.35) |
| P2c | full_fp32 > 2 GiB (0.9) | 8391 MiB | yes |
| P2d | rs512 > 512 MiB (0.6) | 545 MiB | yes |
| P2e | rs1024 > 512 MiB (0.95) | 2113 MiB | yes |
| P3 | tile_p256_o50_fp32 r ≥ 0.8 with CI lower ≥ 0.8 (0.6); basis: PRCM global-mean gate changes context per tile | r = 0.972 [0.935, 1.033] → preserved | yes |
| P4a | Δseam(256) CI lower > 0 (0.8) | +0.080 [+0.070, +0.091] | yes |
| P4b | Δratio(256) CI contains 0 or negative (0.75) | −0.108 [−0.159, −0.052] (negative) | yes |
| P5 | rs256 r < 0.8 (0.85) | 0.671 [0.614, 0.742] | yes |
| P6 | PBE-Net AMP non-finite ≥ 1 image (0.2) | 0 (all AMP rows finite) | no |
| P7a | G2: all fp32 tiled rows preserved except UCMNet p256 (0.85) | yes (UTANet 4/4, MAResUNet 4/4, UCMNet p512 2/2) | yes |
| P7b | UCMNet tile_p256_o0_fp32 indeterminate (0.6) | r = 0.790 [0.755, 0.848] → indeterminate; p256_o50 not preserved (0.721 [0.691, 0.778]) | yes |

PRCM prediction (sec. 0.1 rule): the pre-registered concern that the global-mean channel gate would break tiling at 256 px did not materialise (P3 hit: preserved at both overlaps). Interpretation sentences are limited to this hit/miss.

Note: PBE-Net training-graph n_params 1,335,880 exceeds the 1,334,337 inference parameters (Table 2 parentheses) by 1,543, consistent with the training-only heads being excluded from the inference count (not verified here).
