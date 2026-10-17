# SUMMARY (exploratory acquisition-degradation collapse search; machine-written)

dataset=kvasir n_val_full=100 n_used=40 (strided subset) models=11 batch=1 CPU val split k=1 only; package_status=COMPLETE
chance IoU (on the used images): all-bg=0.0000 all-fg=0.1606 max=0.1606
baseline "fell" = identity > chance+0.05 and condition <= chance+0.05; PBE retains = identity > chance+0.10 and condition >= chance+0.10; C_CANDIDATE = >= 8 of 10 baselines fell and PBE retains; OBSERVATION = >= 8 fell and PBE does not retain

mean val IoU (E = error; - = not run)

| model | identity | gauss_s005 | gauss_s01 | gauss_s02 | gauss_s04 | poisson_p30 | poisson_p10 | poisson_p3 | poisson_p1 | sp_5 | sp_20 | sp_50 | jpeg_q50 | jpeg_q20 | jpeg_q10 | jpeg_q5 | gblur_s1 | gblur_s2 | gblur_s4 | gblur_s8 | mblur_l9 | mblur_l21 | mblur_l41 | dsup_16 | occl_25 | occl_50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pbe | 0.847 | 0.180 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.811 | 0.720 | 0.563 | 0.368 | 0.830 | 0.693 | 0.443 | 0.190 | 0.730 | 0.411 | 0.157 | 0.180 | 0.399 | 0.020 |
| maresunet | 0.824 | 0.802 | 0.771 | 0.413 | 0.036 | 0.727 | 0.397 | 0.014 | 0.004 | 0.654 | 0.011 | 0.000 | 0.812 | 0.752 | 0.612 | 0.318 | 0.834 | 0.795 | 0.733 | 0.392 | 0.801 | 0.609 | 0.290 | 0.388 | 0.419 | 0.071 |
| utanet | 0.850 | 0.715 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.816 | 0.676 | 0.446 | 0.278 | 0.820 | 0.799 | 0.692 | 0.223 | 0.788 | 0.487 | 0.113 | 0.231 | 0.415 | 0.037 |
| mk_unet | 0.846 | 0.716 | 0.291 | 0.000 | 0.000 | 0.165 | 0.014 | 0.000 | 0.001 | 0.146 | 0.000 | 0.000 | 0.816 | 0.751 | 0.664 | 0.479 | 0.812 | 0.772 | 0.454 | 0.137 | 0.767 | 0.459 | 0.223 | 0.114 | 0.413 | 0.059 |
| mhorunet | 0.792 | 0.755 | 0.718 | 0.578 | 0.323 | 0.685 | 0.537 | 0.268 | 0.118 | 0.669 | 0.434 | 0.285 | 0.769 | 0.764 | 0.695 | 0.507 | 0.786 | 0.775 | 0.757 | 0.571 | 0.773 | 0.731 | 0.551 | 0.591 | 0.388 | 0.070 |
| tinyunet | 0.847 | 0.117 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.829 | 0.761 | 0.592 | 0.333 | 0.787 | 0.655 | 0.298 | 0.007 | 0.681 | 0.184 | 0.005 | 0.009 | 0.411 | 0.049 |
| ucmnet | 0.753 | 0.726 | 0.594 | 0.399 | 0.350 | 0.520 | 0.420 | 0.289 | 0.246 | 0.541 | 0.379 | 0.281 | 0.750 | 0.681 | 0.681 | 0.489 | 0.761 | 0.755 | 0.710 | 0.574 | 0.754 | 0.707 | 0.570 | 0.547 | 0.359 | 0.046 |
| lbunet | 0.705 | 0.688 | 0.606 | 0.434 | 0.235 | 0.558 | 0.442 | 0.247 | 0.155 | 0.633 | 0.368 | 0.175 | 0.688 | 0.661 | 0.628 | 0.475 | 0.686 | 0.677 | 0.676 | 0.527 | 0.681 | 0.643 | 0.486 | 0.505 | 0.301 | 0.020 |
| cmunext | 0.819 | 0.705 | 0.083 | 0.000 | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.806 | 0.754 | 0.658 | 0.461 | 0.814 | 0.771 | 0.669 | 0.361 | 0.766 | 0.679 | 0.458 | 0.367 | 0.428 | 0.057 |
| egeunet | 0.672 | 0.604 | 0.489 | 0.402 | 0.063 | 0.483 | 0.299 | 0.015 | 0.000 | 0.435 | 0.327 | 0.025 | 0.677 | 0.641 | 0.629 | 0.519 | 0.674 | 0.628 | 0.590 | 0.501 | 0.643 | 0.591 | 0.516 | 0.504 | 0.325 | 0.014 |
| malunet | 0.723 | 0.656 | 0.530 | 0.376 | 0.200 | 0.475 | 0.384 | 0.164 | 0.055 | 0.482 | 0.347 | 0.210 | 0.707 | 0.714 | 0.639 | 0.579 | 0.720 | 0.698 | 0.626 | 0.492 | 0.693 | 0.661 | 0.568 | 0.497 | 0.365 | 0.050 |

NaN/Inf logits: none

| condition | baselines fell (of 10) | healthy at identity | undetermined | PBE retains | verdict |
|---|---|---|---|---|---|
| gauss_s005 | 1 | 10 | 0 | False | REJECT |
| gauss_s01 | 3 | 10 | 0 | False | REJECT |
| gauss_s02 | 4 | 10 | 0 | False | REJECT |
| gauss_s04 | 7 | 10 | 0 | False | REJECT |
| poisson_p30 | 4 | 10 | 0 | False | REJECT |
| poisson_p10 | 4 | 10 | 0 | False | REJECT |
| poisson_p3 | 7 | 10 | 0 | False | REJECT |
| poisson_p1 | 9 | 10 | 0 | False | OBSERVATION |
| sp_5 | 4 | 10 | 0 | False | REJECT |
| sp_20 | 5 | 10 | 0 | False | REJECT |
| sp_50 | 8 | 10 | 0 | False | OBSERVATION |
| jpeg_q50 | 0 | 10 | 0 | True | REJECT |
| jpeg_q20 | 0 | 10 | 0 | True | REJECT |
| jpeg_q10 | 0 | 10 | 0 | True | REJECT |
| jpeg_q5 | 0 | 10 | 0 | True | REJECT |
| gblur_s1 | 0 | 10 | 0 | True | REJECT |
| gblur_s2 | 0 | 10 | 0 | True | REJECT |
| gblur_s4 | 0 | 10 | 0 | True | REJECT |
| gblur_s8 | 2 | 10 | 0 | False | REJECT |
| mblur_l9 | 0 | 10 | 0 | True | REJECT |
| mblur_l21 | 1 | 10 | 0 | True | REJECT |
| mblur_l41 | 2 | 10 | 0 | False | REJECT |
| dsup_16 | 2 | 10 | 0 | False | REJECT |
| occl_25 | 0 | 10 | 0 | True | REJECT |
| occl_50 | 10 | 10 | 0 | False | OBSERVATION |

Package mechanical verdict (order C_CANDIDATE > OBSERVATION > HOLD_INCOMPLETE > REJECT): OBSERVATION
Conditions C_CANDIDATE: none
Conditions OBSERVATION: ['poisson_p1', 'sp_50', 'occl_50']
Conditions HOLD_INCOMPLETE: none
(Cross-dataset rule: a condition is a C lead only if C_CANDIDATE on >= 4 of the 6 dataset packages 9501-9506 incl. >= 1 skin and >= 1 polyp; evaluated by the aggregator.)

Integrity: degradation_determinism_check=True alias_groups_after=210
