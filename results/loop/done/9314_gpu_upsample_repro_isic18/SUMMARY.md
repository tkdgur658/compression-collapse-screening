# SUMMARY (exploratory input-resolution accuracy; machine-written)

dataset=isic18 split_k=1 n_val=538; chance IoU all-bg=0.0000 all-fg=0.2203 max=0.2203; package_status=COMPLETE_EXPLORATORY_NO_CONFIRMATION
Input: 256x256 validation image bilinearly upsampled to R (align_corners False), probability map area-downsampled back to 256, threshold 0.5.

| model | R=256 | R=512 | R=768 | R=1024 | R=1536 | R=2048 |
|---|---|---|---|---|---|---|
| pbe | 0.8334 | 0.5383 | 0.4313 | 0.4075 | 0.3937 | 0.3909 |
| maresunet | 0.8331 | 0.4113 | 0.2607 | 0.2400 | 0.2293 | 0.2257 |
| utanet | 0.8253 | 0.5135 | 0.4319 | 0.4136 | 0.4058 | 0.4072 |
| mk_unet | 0.8295 | 0.6817 | 0.5102 | 0.4239 | 0.3628 | 0.3419 |
| mhorunet | 0.8155 | 0.6707 | 0.5425 | 0.4674 | 0.4023 | 0.3768 |
| tinyunet | 0.8269 | 0.4591 | 0.3857 | 0.3494 | 0.3179 | 0.2999 |
| ucmnet | 0.8285 | 0.4487 | 0.2177 | 0.1584 | 0.1292 | 0.1215 |
| lbunet | 0.8217 | 0.5753 | 0.4217 | 0.3791 | 0.3586 | 0.3591 |
| cmunext | 0.8272 | 0.5225 | 0.4523 | 0.4319 | 0.4145 | 0.4066 |
| egeunet | 0.8166 | 0.5711 | 0.4451 | 0.4028 | 0.3673 | 0.3571 |
| malunet | 0.8180 | 0.6656 | 0.5706 | 0.4804 | 0.3889 | 0.3464 |

R=512: baselines collapsed (IoU <= chance+0.05) 0/10 (ok 10, unknown 0); pbe holds (> chance+0.10)
R=1024: baselines collapsed (IoU <= chance+0.05) 2/10 (ok 8, unknown 0); pbe holds (> chance+0.10)
R=2048: baselines collapsed (IoU <= chance+0.05) 2/10 (ok 8, unknown 0); pbe holds (> chance+0.10)
Mechanical verdict (PREREG accuracy rule, SECONDARY candidate): REJECT_BASELINES_DO_NOT_COLLAPSE
