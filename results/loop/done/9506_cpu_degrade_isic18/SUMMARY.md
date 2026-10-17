# SUMMARY (exploratory acquisition-degradation collapse search; machine-written)

dataset=isic18 n_val_full=538 n_used=40 (strided subset) models=11 batch=1 CPU val split k=1 only; package_status=COMPLETE
chance IoU (on the used images): all-bg=0.0000 all-fg=0.2564 max=0.2564
baseline "fell" = identity > chance+0.05 and condition <= chance+0.05; PBE retains = identity > chance+0.10 and condition >= chance+0.10; C_CANDIDATE = >= 8 of 10 baselines fell and PBE retains; OBSERVATION = >= 8 fell and PBE does not retain

mean val IoU (E = error; - = not run)

| model | identity | gauss_s005 | gauss_s01 | gauss_s02 | gauss_s04 | poisson_p30 | poisson_p10 | poisson_p3 | poisson_p1 | sp_5 | sp_20 | sp_50 | jpeg_q50 | jpeg_q20 | jpeg_q10 | jpeg_q5 | gblur_s1 | gblur_s2 | gblur_s4 | gblur_s8 | mblur_l9 | mblur_l21 | mblur_l41 | dsup_16 | occl_25 | occl_50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pbe | 0.785 | 0.577 | 0.187 | 0.000 | 0.000 | 0.116 | 0.000 | 0.000 | 0.000 | 0.434 | 0.001 | 0.000 | 0.773 | 0.772 | 0.714 | 0.594 | 0.777 | 0.766 | 0.758 | 0.730 | 0.770 | 0.744 | 0.683 | 0.721 | 0.407 | 0.185 |
| maresunet | 0.776 | 0.774 | 0.767 | 0.728 | 0.664 | 0.767 | 0.733 | 0.683 | 0.523 | 0.743 | 0.662 | 0.491 | 0.778 | 0.779 | 0.777 | 0.701 | 0.774 | 0.772 | 0.764 | 0.718 | 0.771 | 0.749 | 0.664 | 0.710 | 0.494 | 0.384 |
| utanet | 0.776 | 0.661 | 0.495 | 0.000 | 0.000 | 0.140 | 0.000 | 0.000 | 0.000 | 0.324 | 0.000 | 0.000 | 0.777 | 0.784 | 0.769 | 0.661 | 0.772 | 0.757 | 0.747 | 0.720 | 0.757 | 0.730 | 0.664 | 0.709 | 0.452 | 0.420 |
| mk_unet | 0.799 | 0.671 | 0.605 | 0.491 | 0.351 | 0.554 | 0.464 | 0.383 | 0.346 | 0.633 | 0.403 | 0.344 | 0.790 | 0.784 | 0.755 | 0.631 | 0.777 | 0.752 | 0.707 | 0.651 | 0.759 | 0.711 | 0.647 | 0.645 | 0.451 | 0.405 |
| mhorunet | 0.777 | 0.779 | 0.754 | 0.693 | 0.612 | 0.735 | 0.677 | 0.625 | 0.532 | 0.748 | 0.624 | 0.501 | 0.775 | 0.772 | 0.751 | 0.707 | 0.770 | 0.764 | 0.751 | 0.714 | 0.765 | 0.742 | 0.687 | 0.713 | 0.446 | 0.372 |
| tinyunet | 0.783 | 0.014 | 0.022 | 0.000 | 0.000 | 0.044 | 0.002 | 0.000 | 0.000 | 0.088 | 0.000 | 0.000 | 0.783 | 0.792 | 0.768 | 0.622 | 0.793 | 0.751 | 0.717 | 0.663 | 0.753 | 0.707 | 0.646 | 0.658 | 0.411 | 0.072 |
| ucmnet | 0.785 | 0.782 | 0.756 | 0.707 | 0.599 | 0.744 | 0.667 | 0.530 | 0.363 | 0.741 | 0.637 | 0.414 | 0.787 | 0.786 | 0.770 | 0.728 | 0.785 | 0.786 | 0.770 | 0.726 | 0.783 | 0.755 | 0.698 | 0.722 | 0.479 | 0.319 |
| lbunet | 0.788 | 0.780 | 0.718 | 0.574 | 0.347 | 0.653 | 0.543 | 0.345 | 0.154 | 0.667 | 0.481 | 0.186 | 0.783 | 0.778 | 0.777 | 0.708 | 0.775 | 0.767 | 0.758 | 0.726 | 0.767 | 0.747 | 0.687 | 0.723 | 0.479 | 0.361 |
| cmunext | 0.771 | 0.664 | 0.458 | 0.178 | 0.075 | 0.428 | 0.196 | 0.173 | 0.011 | 0.443 | 0.132 | 0.005 | 0.772 | 0.770 | 0.753 | 0.665 | 0.778 | 0.764 | 0.748 | 0.710 | 0.770 | 0.740 | 0.669 | 0.700 | 0.472 | 0.242 |
| egeunet | 0.775 | 0.741 | 0.645 | 0.456 | 0.173 | 0.587 | 0.429 | 0.179 | 0.043 | 0.604 | 0.397 | 0.040 | 0.774 | 0.773 | 0.775 | 0.674 | 0.761 | 0.757 | 0.746 | 0.710 | 0.759 | 0.738 | 0.688 | 0.706 | 0.494 | 0.394 |
| malunet | 0.788 | 0.774 | 0.730 | 0.566 | 0.415 | 0.662 | 0.525 | 0.399 | 0.317 | 0.705 | 0.494 | 0.305 | 0.789 | 0.788 | 0.785 | 0.733 | 0.787 | 0.784 | 0.769 | 0.728 | 0.783 | 0.759 | 0.697 | 0.730 | 0.531 | 0.409 |

NaN/Inf logits: none

| condition | baselines fell (of 10) | healthy at identity | undetermined | PBE retains | verdict |
|---|---|---|---|---|---|
| gauss_s005 | 1 | 10 | 0 | True | REJECT |
| gauss_s01 | 1 | 10 | 0 | False | REJECT |
| gauss_s02 | 3 | 10 | 0 | False | REJECT |
| gauss_s04 | 4 | 10 | 0 | False | REJECT |
| poisson_p30 | 2 | 10 | 0 | False | REJECT |
| poisson_p10 | 3 | 10 | 0 | False | REJECT |
| poisson_p3 | 4 | 10 | 0 | False | REJECT |
| poisson_p1 | 5 | 10 | 0 | False | REJECT |
| sp_5 | 1 | 10 | 0 | True | REJECT |
| sp_20 | 3 | 10 | 0 | False | REJECT |
| sp_50 | 6 | 10 | 0 | False | REJECT |
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
| occl_50 | 2 | 10 | 0 | False | REJECT |

Package mechanical verdict (order C_CANDIDATE > OBSERVATION > HOLD_INCOMPLETE > REJECT): REJECT
Conditions C_CANDIDATE: none
Conditions OBSERVATION: none
Conditions HOLD_INCOMPLETE: none
(Cross-dataset rule: a condition is a C lead only if C_CANDIDATE on >= 4 of the 6 dataset packages 9501-9506 incl. >= 1 skin and >= 1 polyp; evaluated by the aggregator.)

Integrity: degradation_determinism_check=True alias_groups_after=210
