# SUMMARY (exploratory acquisition-degradation collapse search; machine-written)

dataset=PH2 n_val_full=40 n_used=40 (strided subset) models=11 batch=1 CPU val split k=1 only; package_status=COMPLETE
chance IoU (on the used images): all-bg=0.0000 all-fg=0.2305 max=0.2305
baseline "fell" = identity > chance+0.05 and condition <= chance+0.05; PBE retains = identity > chance+0.10 and condition >= chance+0.10; C_CANDIDATE = >= 8 of 10 baselines fell and PBE retains; OBSERVATION = >= 8 fell and PBE does not retain

mean val IoU (E = error; - = not run)

| model | identity | gauss_s005 | gauss_s01 | gauss_s02 | gauss_s04 | poisson_p30 | poisson_p10 | poisson_p3 | poisson_p1 | sp_5 | sp_20 | sp_50 | jpeg_q50 | jpeg_q20 | jpeg_q10 | jpeg_q5 | gblur_s1 | gblur_s2 | gblur_s4 | gblur_s8 | mblur_l9 | mblur_l21 | mblur_l41 | dsup_16 | occl_25 | occl_50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pbe | 0.890 | 0.818 | 0.298 | 0.234 | 0.234 | 0.308 | 0.278 | 0.258 | 0.255 | 0.342 | 0.235 | 0.234 | 0.890 | 0.883 | 0.875 | 0.821 | 0.863 | 0.844 | 0.847 | 0.811 | 0.879 | 0.864 | 0.831 | 0.819 | 0.428 | 0.051 |
| maresunet | 0.890 | 0.890 | 0.887 | 0.857 | 0.664 | 0.881 | 0.858 | 0.702 | 0.208 | 0.875 | 0.762 | 0.322 | 0.890 | 0.885 | 0.883 | 0.850 | 0.887 | 0.876 | 0.867 | 0.848 | 0.882 | 0.870 | 0.826 | 0.848 | 0.378 | 0.013 |
| utanet | 0.867 | 0.694 | 0.623 | 0.562 | 0.319 | 0.632 | 0.546 | 0.309 | 0.069 | 0.460 | 0.467 | 0.017 | 0.869 | 0.867 | 0.852 | 0.789 | 0.865 | 0.839 | 0.810 | 0.722 | 0.855 | 0.825 | 0.781 | 0.765 | 0.415 | 0.018 |
| mk_unet | 0.889 | 0.870 | 0.822 | 0.749 | 0.415 | 0.792 | 0.732 | 0.433 | 0.365 | 0.850 | 0.780 | 0.605 | 0.883 | 0.873 | 0.861 | 0.822 | 0.884 | 0.872 | 0.854 | 0.818 | 0.883 | 0.865 | 0.824 | 0.821 | 0.455 | 0.045 |
| mhorunet | 0.884 | 0.880 | 0.855 | 0.755 | 0.379 | 0.832 | 0.707 | 0.368 | 0.306 | 0.787 | 0.610 | 0.312 | 0.884 | 0.881 | 0.850 | 0.795 | 0.883 | 0.882 | 0.879 | 0.856 | 0.882 | 0.873 | 0.847 | 0.859 | 0.627 | 0.421 |
| tinyunet | 0.892 | 0.000 | 0.001 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.004 | 0.012 | 0.890 | 0.881 | 0.873 | 0.823 | 0.884 | 0.859 | 0.807 | 0.687 | 0.872 | 0.826 | 0.716 | 0.695 | 0.428 | 0.037 |
| ucmnet | 0.892 | 0.873 | 0.794 | 0.579 | 0.344 | 0.808 | 0.676 | 0.397 | 0.352 | 0.703 | 0.384 | 0.313 | 0.891 | 0.888 | 0.874 | 0.794 | 0.891 | 0.890 | 0.883 | 0.850 | 0.891 | 0.880 | 0.849 | 0.857 | 0.502 | 0.336 |
| lbunet | 0.879 | 0.868 | 0.772 | 0.437 | 0.307 | 0.585 | 0.392 | 0.335 | 0.267 | 0.726 | 0.355 | 0.278 | 0.874 | 0.873 | 0.849 | 0.786 | 0.880 | 0.879 | 0.873 | 0.849 | 0.878 | 0.870 | 0.845 | 0.851 | 0.506 | 0.297 |
| cmunext | 0.897 | 0.870 | 0.514 | 0.015 | 0.003 | 0.239 | 0.009 | 0.010 | 0.014 | 0.093 | 0.006 | 0.023 | 0.893 | 0.890 | 0.876 | 0.827 | 0.856 | 0.836 | 0.797 | 0.763 | 0.881 | 0.856 | 0.804 | 0.762 | 0.436 | 0.034 |
| egeunet | 0.859 | 0.789 | 0.656 | 0.369 | 0.097 | 0.489 | 0.127 | 0.000 | 0.000 | 0.656 | 0.587 | 0.399 | 0.853 | 0.851 | 0.842 | 0.830 | 0.834 | 0.834 | 0.836 | 0.801 | 0.848 | 0.845 | 0.835 | 0.816 | 0.331 | 0.001 |
| malunet | 0.874 | 0.871 | 0.844 | 0.711 | 0.464 | 0.842 | 0.745 | 0.516 | 0.276 | 0.807 | 0.652 | 0.419 | 0.875 | 0.873 | 0.865 | 0.837 | 0.871 | 0.867 | 0.857 | 0.828 | 0.868 | 0.855 | 0.836 | 0.838 | 0.436 | 0.439 |

NaN/Inf logits: none

| condition | baselines fell (of 10) | healthy at identity | undetermined | PBE retains | verdict |
|---|---|---|---|---|---|
| gauss_s005 | 1 | 10 | 0 | True | REJECT |
| gauss_s01 | 1 | 10 | 0 | False | REJECT |
| gauss_s02 | 2 | 10 | 0 | False | REJECT |
| gauss_s04 | 3 | 10 | 0 | False | REJECT |
| poisson_p30 | 2 | 10 | 0 | False | REJECT |
| poisson_p10 | 3 | 10 | 0 | False | REJECT |
| poisson_p3 | 3 | 10 | 0 | False | REJECT |
| poisson_p1 | 7 | 10 | 0 | False | REJECT |
| sp_5 | 2 | 10 | 0 | True | REJECT |
| sp_20 | 2 | 10 | 0 | False | REJECT |
| sp_50 | 4 | 10 | 0 | False | REJECT |
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
| occl_50 | 6 | 10 | 0 | False | REJECT |

Package mechanical verdict (order C_CANDIDATE > OBSERVATION > HOLD_INCOMPLETE > REJECT): REJECT
Conditions C_CANDIDATE: none
Conditions OBSERVATION: none
Conditions HOLD_INCOMPLETE: none
(Cross-dataset rule: a condition is a C lead only if C_CANDIDATE on >= 4 of the 6 dataset packages 9501-9506 incl. >= 1 skin and >= 1 polyp; evaluated by the aggregator.)

Integrity: degradation_determinism_check=True alias_groups_after=210
