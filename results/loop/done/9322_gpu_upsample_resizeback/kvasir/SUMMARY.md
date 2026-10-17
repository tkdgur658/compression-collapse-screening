# SUMMARY (exploratory input-resolution accuracy; machine-written)

dataset=kvasir split_k=1 n_val=100; chance IoU all-bg=0.0000 all-fg=0.1391 max=0.1391; package_status=COMPLETE_EXPLORATORY_NO_CONFIRMATION
Input: 256x256 validation image bilinearly upsampled to R (align_corners False), probability map area-downsampled back to 256, threshold 0.5.

| model | R=256 | R=512 | R=768 | R=1024 | R=1536 | R=2048 |
|---|---|---|---|---|---|---|
| pbe | 0.8081 | 0.8110 | 0.8105 | 0.8110 | 0.8110 | 0.8110 |
| maresunet | 0.8125 | 0.8131 | 0.8134 | 0.8131 | 0.8131 | 0.8131 |
| utanet | 0.8116 | 0.8166 | 0.8166 | 0.8166 | 0.8166 | 0.8166 |
| mk_unet | 0.8060 | 0.8166 | 0.8159 | 0.8166 | 0.8166 | 0.8166 |
| mhorunet | 0.7439 | 0.7462 | 0.7461 | 0.7462 | 0.7462 | 0.7462 |
| tinyunet | 0.7836 | 0.7789 | 0.7796 | 0.7789 | 0.7789 | 0.7789 |
| ucmnet | 0.7047 | 0.6997 | 0.7001 | 0.6997 | 0.6997 | 0.6997 |
| lbunet | 0.6760 | 0.6721 | 0.6725 | 0.6721 | 0.6721 | 0.6721 |
| cmunext | 0.7840 | 0.7918 | 0.7921 | 0.7918 | 0.7918 | 0.7918 |
| egeunet | 0.6413 | 0.6430 | 0.6431 | 0.6430 | 0.6430 | 0.6430 |
| malunet | 0.6860 | 0.6789 | 0.6796 | 0.6789 | 0.6789 | 0.6789 |

R=512: baselines collapsed (IoU <= chance+0.05) 0/10 (ok 10, unknown 0); pbe holds (> chance+0.10)
R=1024: baselines collapsed (IoU <= chance+0.05) 0/10 (ok 10, unknown 0); pbe holds (> chance+0.10)
R=2048: baselines collapsed (IoU <= chance+0.05) 0/10 (ok 10, unknown 0); pbe holds (> chance+0.10)
Mechanical verdict (PREREG accuracy rule, SECONDARY candidate): REJECT_BASELINES_DO_NOT_COLLAPSE
