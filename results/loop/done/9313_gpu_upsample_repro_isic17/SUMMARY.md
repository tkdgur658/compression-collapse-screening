# SUMMARY (exploratory input-resolution accuracy; machine-written)

dataset=isic17 split_k=1 n_val=430; chance IoU all-bg=0.0000 all-fg=0.1978 max=0.1978; package_status=COMPLETE_EXPLORATORY_NO_CONFIRMATION
Input: 256x256 validation image bilinearly upsampled to R (align_corners False), probability map area-downsampled back to 256, threshold 0.5.

| model | R=256 | R=512 | R=768 | R=1024 | R=1536 | R=2048 |
|---|---|---|---|---|---|---|
| pbe | 0.8316 | 0.5067 | 0.3998 | 0.3760 | 0.3721 | 0.3800 |
| maresunet | 0.8288 | 0.5254 | 0.3752 | 0.3296 | 0.3009 | 0.2946 |
| utanet | 0.8212 | 0.5935 | 0.4890 | 0.4568 | 0.4323 | 0.4163 |
| mk_unet | 0.8294 | 0.6260 | 0.4686 | 0.4080 | 0.3698 | 0.3491 |
| mhorunet | 0.8223 | 0.7346 | 0.6616 | 0.6001 | 0.5293 | 0.4892 |
| tinyunet | 0.8284 | 0.4673 | 0.3610 | 0.3219 | 0.2920 | 0.2775 |
| ucmnet | 0.8311 | 0.5078 | 0.2368 | 0.1334 | 0.0755 | 0.0613 |
| lbunet | 0.8267 | 0.6589 | 0.5089 | 0.4505 | 0.4228 | 0.4196 |
| cmunext | 0.8239 | 0.4884 | 0.3861 | 0.3620 | 0.3507 | 0.3522 |
| egeunet | 0.8243 | 0.6297 | 0.4818 | 0.4725 | 0.4727 | 0.4695 |
| malunet | 0.8218 | 0.6989 | 0.6247 | 0.5243 | 0.4159 | 0.3710 |

R=512: baselines collapsed (IoU <= chance+0.05) 0/10 (ok 10, unknown 0); pbe holds (> chance+0.10)
R=1024: baselines collapsed (IoU <= chance+0.05) 1/10 (ok 9, unknown 0); pbe holds (> chance+0.10)
R=2048: baselines collapsed (IoU <= chance+0.05) 1/10 (ok 9, unknown 0); pbe holds (> chance+0.10)
Mechanical verdict (PREREG accuracy rule, SECONDARY candidate): REJECT_BASELINES_DO_NOT_COLLAPSE
