# SUMMARY (exploratory acquisition-degradation collapse search; machine-written)

dataset=isic17 n_val_full=430 n_used=40 (strided subset) models=11 batch=1 CPU val split k=1 only; package_status=COMPLETE
chance IoU (on the used images): all-bg=0.0000 all-fg=0.1959 max=0.1959
baseline "fell" = identity > chance+0.05 and condition <= chance+0.05; PBE retains = identity > chance+0.10 and condition >= chance+0.10; C_CANDIDATE = >= 8 of 10 baselines fell and PBE retains; OBSERVATION = >= 8 fell and PBE does not retain

mean val IoU (E = error; - = not run)

| model | identity | gauss_s005 | gauss_s01 | gauss_s02 | gauss_s04 | poisson_p30 | poisson_p10 | poisson_p3 | poisson_p1 | sp_5 | sp_20 | sp_50 | jpeg_q50 | jpeg_q20 | jpeg_q10 | jpeg_q5 | gblur_s1 | gblur_s2 | gblur_s4 | gblur_s8 | mblur_l9 | mblur_l21 | mblur_l41 | dsup_16 | occl_25 | occl_50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pbe | 0.838 | 0.650 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.046 | 0.000 | 0.000 | 0.832 | 0.824 | 0.809 | 0.734 | 0.838 | 0.836 | 0.816 | 0.752 | 0.835 | 0.801 | 0.715 | 0.755 | 0.477 | 0.364 |
| maresunet | 0.838 | 0.830 | 0.825 | 0.791 | 0.642 | 0.819 | 0.775 | 0.650 | 0.444 | 0.798 | 0.652 | 0.402 | 0.833 | 0.831 | 0.817 | 0.773 | 0.838 | 0.834 | 0.808 | 0.734 | 0.832 | 0.787 | 0.694 | 0.736 | 0.575 | 0.411 |
| utanet | 0.843 | 0.818 | 0.759 | 0.722 | 0.308 | 0.761 | 0.649 | 0.153 | 0.002 | 0.770 | 0.537 | 0.036 | 0.840 | 0.828 | 0.811 | 0.736 | 0.835 | 0.837 | 0.804 | 0.729 | 0.835 | 0.785 | 0.703 | 0.734 | 0.455 | 0.367 |
| mk_unet | 0.837 | 0.809 | 0.611 | 0.209 | 0.254 | 0.362 | 0.265 | 0.348 | 0.319 | 0.767 | 0.442 | 0.156 | 0.833 | 0.820 | 0.809 | 0.733 | 0.836 | 0.821 | 0.783 | 0.714 | 0.826 | 0.780 | 0.692 | 0.718 | 0.455 | 0.330 |
| mhorunet | 0.839 | 0.837 | 0.814 | 0.752 | 0.581 | 0.795 | 0.734 | 0.584 | 0.520 | 0.810 | 0.677 | 0.498 | 0.837 | 0.821 | 0.808 | 0.755 | 0.838 | 0.833 | 0.815 | 0.753 | 0.833 | 0.800 | 0.721 | 0.755 | 0.528 | 0.404 |
| tinyunet | 0.855 | 0.564 | 0.031 | 0.000 | 0.000 | 0.010 | 0.000 | 0.000 | 0.000 | 0.443 | 0.000 | 0.000 | 0.849 | 0.846 | 0.828 | 0.767 | 0.855 | 0.831 | 0.792 | 0.705 | 0.832 | 0.761 | 0.678 | 0.710 | 0.352 | 0.087 |
| ucmnet | 0.836 | 0.832 | 0.833 | 0.802 | 0.655 | 0.820 | 0.791 | 0.618 | 0.460 | 0.818 | 0.741 | 0.522 | 0.833 | 0.818 | 0.807 | 0.755 | 0.837 | 0.835 | 0.818 | 0.758 | 0.833 | 0.799 | 0.712 | 0.756 | 0.490 | 0.244 |
| lbunet | 0.836 | 0.815 | 0.775 | 0.603 | 0.337 | 0.707 | 0.552 | 0.332 | 0.202 | 0.760 | 0.493 | 0.257 | 0.829 | 0.820 | 0.810 | 0.739 | 0.838 | 0.836 | 0.817 | 0.749 | 0.835 | 0.797 | 0.710 | 0.752 | 0.468 | 0.381 |
| cmunext | 0.845 | 0.817 | 0.709 | 0.454 | 0.000 | 0.525 | 0.336 | 0.000 | 0.000 | 0.606 | 0.107 | 0.000 | 0.838 | 0.831 | 0.820 | 0.784 | 0.843 | 0.829 | 0.800 | 0.734 | 0.835 | 0.784 | 0.703 | 0.736 | 0.460 | 0.377 |
| egeunet | 0.830 | 0.808 | 0.724 | 0.468 | 0.226 | 0.602 | 0.409 | 0.153 | 0.034 | 0.698 | 0.369 | 0.194 | 0.829 | 0.821 | 0.810 | 0.753 | 0.833 | 0.828 | 0.808 | 0.753 | 0.827 | 0.792 | 0.717 | 0.756 | 0.239 | 0.268 |
| malunet | 0.822 | 0.823 | 0.781 | 0.637 | 0.458 | 0.726 | 0.583 | 0.365 | 0.258 | 0.749 | 0.589 | 0.453 | 0.821 | 0.816 | 0.802 | 0.790 | 0.824 | 0.818 | 0.802 | 0.745 | 0.819 | 0.788 | 0.706 | 0.744 | 0.517 | 0.340 |

NaN/Inf logits: none

| condition | baselines fell (of 10) | healthy at identity | undetermined | PBE retains | verdict |
|---|---|---|---|---|---|
| gauss_s005 | 0 | 10 | 0 | True | REJECT |
| gauss_s01 | 1 | 10 | 0 | False | REJECT |
| gauss_s02 | 2 | 10 | 0 | False | REJECT |
| gauss_s04 | 3 | 10 | 0 | False | REJECT |
| poisson_p30 | 1 | 10 | 0 | False | REJECT |
| poisson_p10 | 1 | 10 | 0 | False | REJECT |
| poisson_p3 | 4 | 10 | 0 | False | REJECT |
| poisson_p1 | 5 | 10 | 0 | False | REJECT |
| sp_5 | 0 | 10 | 0 | False | REJECT |
| sp_20 | 2 | 10 | 0 | False | REJECT |
| sp_50 | 5 | 10 | 0 | False | REJECT |
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
| occl_25 | 1 | 10 | 0 | True | REJECT |
| occl_50 | 2 | 10 | 0 | True | REJECT |

Package mechanical verdict (order C_CANDIDATE > OBSERVATION > HOLD_INCOMPLETE > REJECT): REJECT
Conditions C_CANDIDATE: none
Conditions OBSERVATION: none
Conditions HOLD_INCOMPLETE: none
(Cross-dataset rule: a condition is a C lead only if C_CANDIDATE on >= 4 of the 6 dataset packages 9501-9506 incl. >= 1 skin and >= 1 polyp; evaluated by the aggregator.)

Integrity: degradation_determinism_check=True alias_groups_after=210
