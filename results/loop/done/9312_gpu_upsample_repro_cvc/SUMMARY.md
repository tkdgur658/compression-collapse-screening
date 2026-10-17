# SUMMARY (exploratory input-resolution accuracy; machine-written)

dataset=cvc split_k=1 n_val=61; chance IoU all-bg=0.0000 all-fg=0.0778 max=0.0778; package_status=COMPLETE_EXPLORATORY_NO_CONFIRMATION
Input: 256x256 validation image bilinearly upsampled to R (align_corners False), probability map area-downsampled back to 256, threshold 0.5.

| model | R=256 | R=512 | R=768 | R=1024 | R=1536 | R=2048 |
|---|---|---|---|---|---|---|
| pbe | 0.8469 | 0.5062 | 0.3261 | 0.2325 | 0.1724 | 0.1774 |
| maresunet | 0.8159 | 0.6454 | 0.4039 | 0.2435 | 0.1227 | 0.0777 |
| utanet | 0.8114 | 0.4326 | 0.2582 | 0.1770 | 0.1218 | 0.1203 |
| mk_unet | 0.8460 | 0.4786 | 0.2550 | 0.1311 | 0.0688 | 0.0596 |
| mhorunet | 0.7076 | 0.4753 | 0.2961 | 0.2234 | 0.1529 | 0.1258 |
| tinyunet | 0.8158 | 0.5010 | 0.2716 | 0.1607 | 0.1136 | 0.1123 |
| ucmnet | 0.6588 | 0.4940 | 0.2904 | 0.2215 | 0.1733 | 0.1552 |
| lbunet | 0.6424 | 0.5184 | 0.4001 | 0.3396 | 0.2836 | 0.2613 |
| cmunext | 0.7226 | 0.3307 | 0.1579 | 0.0843 | 0.0747 | 0.0804 |
| egeunet | 0.5799 | 0.3441 | 0.2352 | 0.2050 | 0.1867 | 0.1754 |
| malunet | 0.5722 | 0.3809 | 0.2894 | 0.2376 | 0.1756 | 0.1607 |

R=512: baselines collapsed (IoU <= chance+0.05) 0/10 (ok 10, unknown 0); pbe holds (> chance+0.10)
R=1024: baselines collapsed (IoU <= chance+0.05) 1/10 (ok 9, unknown 0); pbe holds (> chance+0.10)
R=2048: baselines collapsed (IoU <= chance+0.05) 6/10 (ok 4, unknown 0); pbe collapses (> chance+0.10)
Mechanical verdict (PREREG accuracy rule, SECONDARY candidate): REJECT_BASELINES_DO_NOT_COLLAPSE
