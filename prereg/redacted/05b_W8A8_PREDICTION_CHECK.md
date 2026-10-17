# W8A8 addendum — outcome of the two descriptive predictions (05a, frozen sha cac69ed1)

Source: server `qdeploy_w8a8/RESULT_W8A8.json` (sha 8db4ef4d…, decide run once 2026-10-16 20:2x KST; local copy `iso/s4/qdeploy/RESULT_W8A8.json`).

| # | prediction (p) | observed | hit |
|---|---|---|---|
| P-W1 | L80 collapse in at most 2 of 66 cells (0.8) | 4 of 66 cells, all MAResUNet (CVC-ClinicDB, Kvasir, ISIC 2017, ISIC 2018); 9 of 329 checkpoints | no |
| P-W2 | PBE-Net 0 of 6 at L80 (0.7) | 0 of 6 | yes |

Not predicted, reported: with 9 collapsed checkpoints, output agreement AUC (pooled 0.892, 95% CI 0.655–1.000) does not exceed the foreground-ratio (0.904) or SQNR (0.902) controls; the AUC fact of setting D is therefore stated for setting D only.
