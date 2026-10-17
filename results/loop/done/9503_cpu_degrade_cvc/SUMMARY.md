# SUMMARY (exploratory acquisition-degradation collapse search; machine-written)

dataset=cvc n_val_full=61 n_used=40 (strided subset) models=11 batch=1 CPU val split k=1 only; package_status=COMPLETE
chance IoU (on the used images): all-bg=0.0000 all-fg=0.0860 max=0.0860
baseline "fell" = identity > chance+0.05 and condition <= chance+0.05; PBE retains = identity > chance+0.10 and condition >= chance+0.10; C_CANDIDATE = >= 8 of 10 baselines fell and PBE retains; OBSERVATION = >= 8 fell and PBE does not retain

mean val IoU (E = error; - = not run)

| model | identity | gauss_s005 | gauss_s01 | gauss_s02 | gauss_s04 | poisson_p30 | poisson_p10 | poisson_p3 | poisson_p1 | sp_5 | sp_20 | sp_50 | jpeg_q50 | jpeg_q20 | jpeg_q10 | jpeg_q5 | gblur_s1 | gblur_s2 | gblur_s4 | gblur_s8 | mblur_l9 | mblur_l21 | mblur_l41 | dsup_16 | occl_25 | occl_50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pbe | 0.864 | 0.002 | 0.001 | 0.079 | 0.042 | 0.001 | 0.043 | 0.111 | 0.041 | 0.016 | 0.093 | 0.083 | 0.837 | 0.732 | 0.493 | 0.223 | 0.846 | 0.782 | 0.640 | 0.462 | 0.819 | 0.685 | 0.492 | 0.414 | 0.361 | 0.083 |
| maresunet | 0.839 | 0.768 | 0.488 | 0.187 | 0.161 | 0.552 | 0.289 | 0.166 | 0.141 | 0.429 | 0.233 | 0.124 | 0.833 | 0.741 | 0.492 | 0.176 | 0.832 | 0.789 | 0.617 | 0.421 | 0.808 | 0.616 | 0.437 | 0.395 | 0.398 | 0.080 |
| utanet | 0.826 | 0.351 | 0.291 | 0.028 | 0.000 | 0.298 | 0.011 | 0.000 | 0.016 | 0.154 | 0.000 | 0.003 | 0.815 | 0.679 | 0.411 | 0.209 | 0.809 | 0.721 | 0.555 | 0.319 | 0.776 | 0.590 | 0.387 | 0.289 | 0.349 | 0.055 |
| mk_unet | 0.872 | 0.671 | 0.438 | 0.299 | 0.284 | 0.437 | 0.329 | 0.319 | 0.181 | 0.402 | 0.179 | 0.129 | 0.843 | 0.791 | 0.537 | 0.242 | 0.844 | 0.738 | 0.531 | 0.332 | 0.772 | 0.579 | 0.397 | 0.331 | 0.406 | 0.091 |
| mhorunet | 0.741 | 0.556 | 0.348 | 0.129 | 0.013 | 0.360 | 0.176 | 0.081 | 0.068 | 0.245 | 0.028 | 0.003 | 0.723 | 0.662 | 0.477 | 0.304 | 0.740 | 0.725 | 0.691 | 0.552 | 0.724 | 0.671 | 0.522 | 0.546 | 0.272 | 0.056 |
| tinyunet | 0.849 | 0.000 | 0.088 | 0.155 | 0.100 | 0.163 | 0.192 | 0.146 | 0.126 | 0.000 | 0.002 | 0.088 | 0.793 | 0.699 | 0.397 | 0.246 | 0.797 | 0.710 | 0.532 | 0.281 | 0.759 | 0.577 | 0.360 | 0.269 | 0.361 | 0.045 |
| ucmnet | 0.677 | 0.640 | 0.467 | 0.332 | 0.225 | 0.554 | 0.348 | 0.286 | 0.112 | 0.493 | 0.311 | 0.212 | 0.677 | 0.633 | 0.480 | 0.320 | 0.679 | 0.681 | 0.648 | 0.573 | 0.685 | 0.678 | 0.531 | 0.560 | 0.237 | 0.082 |
| lbunet | 0.658 | 0.600 | 0.450 | 0.237 | 0.179 | 0.405 | 0.247 | 0.173 | 0.156 | 0.400 | 0.233 | 0.135 | 0.626 | 0.550 | 0.390 | 0.239 | 0.647 | 0.637 | 0.608 | 0.487 | 0.640 | 0.605 | 0.514 | 0.481 | 0.193 | 0.058 |
| cmunext | 0.740 | 0.517 | 0.353 | 0.136 | 0.000 | 0.335 | 0.093 | 0.000 | 0.002 | 0.334 | 0.017 | 0.000 | 0.729 | 0.621 | 0.377 | 0.254 | 0.721 | 0.532 | 0.351 | 0.251 | 0.674 | 0.510 | 0.376 | 0.228 | 0.309 | 0.118 |
| egeunet | 0.599 | 0.461 | 0.315 | 0.186 | 0.015 | 0.225 | 0.052 | 0.005 | 0.000 | 0.298 | 0.091 | 0.000 | 0.572 | 0.539 | 0.373 | 0.277 | 0.591 | 0.562 | 0.521 | 0.470 | 0.568 | 0.526 | 0.459 | 0.481 | 0.240 | 0.115 |
| malunet | 0.590 | 0.517 | 0.430 | 0.245 | 0.049 | 0.334 | 0.265 | 0.050 | 0.003 | 0.297 | 0.189 | 0.029 | 0.577 | 0.534 | 0.424 | 0.289 | 0.584 | 0.560 | 0.528 | 0.469 | 0.574 | 0.532 | 0.484 | 0.470 | 0.189 | 0.060 |

NaN/Inf logits: none

| condition | baselines fell (of 10) | healthy at identity | undetermined | PBE retains | verdict |
|---|---|---|---|---|---|
| gauss_s005 | 1 | 10 | 0 | False | REJECT |
| gauss_s01 | 1 | 10 | 0 | False | REJECT |
| gauss_s02 | 3 | 10 | 0 | False | REJECT |
| gauss_s04 | 6 | 10 | 0 | False | REJECT |
| poisson_p30 | 0 | 10 | 0 | False | REJECT |
| poisson_p10 | 3 | 10 | 0 | False | REJECT |
| poisson_p3 | 5 | 10 | 0 | False | REJECT |
| poisson_p1 | 7 | 10 | 0 | False | REJECT |
| sp_5 | 1 | 10 | 0 | False | REJECT |
| sp_20 | 5 | 10 | 0 | False | REJECT |
| sp_50 | 9 | 10 | 0 | False | OBSERVATION |
| jpeg_q50 | 0 | 10 | 0 | True | REJECT |
| jpeg_q20 | 0 | 10 | 0 | True | REJECT |
| jpeg_q10 | 0 | 10 | 0 | True | REJECT |
| jpeg_q5 | 0 | 10 | 0 | True | REJECT |
| gblur_s1 | 0 | 10 | 0 | True | REJECT |
| gblur_s2 | 0 | 10 | 0 | True | REJECT |
| gblur_s4 | 0 | 10 | 0 | True | REJECT |
| gblur_s8 | 0 | 10 | 0 | True | REJECT |
| mblur_l9 | 0 | 10 | 0 | True | REJECT |
| mblur_l21 | 0 | 10 | 0 | True | REJECT |
| mblur_l41 | 0 | 10 | 0 | True | REJECT |
| dsup_16 | 0 | 10 | 0 | True | REJECT |
| occl_25 | 0 | 10 | 0 | True | REJECT |
| occl_50 | 10 | 10 | 0 | False | OBSERVATION |

Package mechanical verdict (order C_CANDIDATE > OBSERVATION > HOLD_INCOMPLETE > REJECT): OBSERVATION
Conditions C_CANDIDATE: none
Conditions OBSERVATION: ['sp_50', 'occl_50']
Conditions HOLD_INCOMPLETE: none
(Cross-dataset rule: a condition is a C lead only if C_CANDIDATE on >= 4 of the 6 dataset packages 9501-9506 incl. >= 1 skin and >= 1 polyp; evaluated by the aggregator.)

Integrity: degradation_determinism_check=True alias_groups_after=210
