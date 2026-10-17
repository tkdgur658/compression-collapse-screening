# SUMMARY (exploratory input-resolution accuracy; machine-written)

dataset=PH2 split_k=1 n_val=40; chance IoU all-bg=0.0000 all-fg=0.2305 max=0.2305; package_status=COMPLETE_EXPLORATORY_NO_CONFIRMATION
Input: 256x256 validation image bilinearly upsampled to R (align_corners False), probability map area-downsampled back to 256, threshold 0.5.

| model | R=256 | R=512 | R=768 | R=1024 | R=1536 | R=2048 |
|---|---|---|---|---|---|---|
| pbe | 0.8898 | 0.8159 | 0.7088 | 0.6644 | 0.6310 | 0.6150 |
| maresunet | 0.8902 | 0.8723 | 0.7709 | 0.7191 | 0.6705 | 0.6493 |
| utanet | 0.8672 | 0.8222 | 0.7177 | 0.6500 | 0.6043 | 0.5808 |
| mk_unet | 0.8889 | 0.8391 | 0.7239 | 0.6747 | 0.6351 | 0.6174 |
| mhorunet | 0.8837 | 0.8138 | 0.7483 | 0.6919 | 0.6395 | 0.6244 |
| tinyunet | 0.8921 | 0.8271 | 0.6761 | 0.5323 | 0.3551 | 0.2615 |
| ucmnet | 0.8919 | 0.7676 | 0.6593 | 0.6163 | 0.5777 | 0.5587 |
| lbunet | 0.8787 | 0.8505 | 0.7957 | 0.7486 | 0.6968 | 0.6755 |
| cmunext | 0.8968 | 0.8094 | 0.7337 | 0.6582 | 0.5374 | 0.4409 |
| egeunet | 0.8586 | 0.7894 | 0.7592 | 0.7303 | 0.6832 | 0.6489 |
| malunet | 0.8742 | 0.7235 | 0.7028 | 0.6693 | 0.5844 | 0.5340 |

R=256 identity check vs recorded reference IoUs: {"cmunext": {"max_abs_diff": 0.0, "mean_abs_diff": 0.0, "n_compared": 40}, "egeunet": {"max_abs_diff": 0.0, "mean_abs_diff": 0.0, "n_compared": 40}, "lbunet": {"max_abs_diff": 0.0, "mean_abs_diff": 0.0, "n_compared": 40}, "malunet": {"max_abs_diff": 0.0, "mean_abs_diff": 0.0, "n_compared": 40}, "maresunet": {"max_abs_diff": 0.0, "mean_abs_diff": 0.0, "n_compared": 40}, "mhorunet": {"max_abs_diff": 0.0, "mean_abs_diff": 0.0, "n_compared": 40}, "mk_unet": {"max_abs_diff": 0.0, "mean_abs_diff": 0.0, "n_compared": 40}, "pbe": {"max_abs_diff": 0.0, "mean_abs_diff": 0.0, "n_compared": 40}, "tinyunet": {"max_abs_diff": 0.0, "mean_abs_diff": 0.0, "n_compared": 40}, "ucmnet": {"max_abs_diff": 0.0, "mean_abs_diff": 0.0, "n_compared": 40}, "utanet": {"max_abs_diff": 0.0, "mean_abs_diff": 0.0, "n_compared": 40}}

R=512: baselines collapsed (IoU <= chance+0.05) 0/10 (ok 10, unknown 0); pbe holds (> chance+0.10)
R=1024: baselines collapsed (IoU <= chance+0.05) 0/10 (ok 10, unknown 0); pbe holds (> chance+0.10)
R=2048: baselines collapsed (IoU <= chance+0.05) 1/10 (ok 9, unknown 0); pbe holds (> chance+0.10)
Mechanical verdict (PREREG accuracy rule, SECONDARY candidate): REJECT_BASELINES_DO_NOT_COLLAPSE
