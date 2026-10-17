# SUMMARY (exploratory input-resolution accuracy; machine-written)

dataset=etis split_k=1 n_val=19; chance IoU all-bg=0.0000 all-fg=0.0255 max=0.0255; package_status=COMPLETE_EXPLORATORY_NO_CONFIRMATION
Input: 256x256 validation image bilinearly upsampled to R (align_corners False), probability map area-downsampled back to 256, threshold 0.5.

| model | R=256 | R=512 | R=768 | R=1024 | R=1536 | R=2048 |
|---|---|---|---|---|---|---|
| pbe | 0.7628 | 0.7658 | 0.7671 | 0.7658 | 0.7657 | 0.7657 |
| maresunet | 0.4964 | 0.4962 | 0.4970 | 0.4961 | 0.4962 | 0.4962 |
| utanet | 0.4566 | 0.4478 | 0.4499 | 0.4478 | 0.4478 | 0.4478 |
| mk_unet | 0.7071 | 0.6710 | 0.6722 | 0.6710 | 0.6710 | 0.6710 |
| mhorunet | 0.5584 | 0.5605 | 0.5607 | 0.5606 | 0.5605 | 0.5606 |
| tinyunet | 0.5866 | 0.5999 | 0.5992 | 0.5999 | 0.5999 | 0.5999 |
| ucmnet | 0.4816 | 0.4822 | 0.4824 | 0.4822 | 0.4822 | 0.4822 |
| lbunet | 0.3800 | 0.3708 | 0.3718 | 0.3708 | 0.3708 | 0.3708 |
| cmunext | 0.3922 | 0.3362 | 0.3398 | 0.3362 | 0.3362 | 0.3362 |
| egeunet | 0.2757 | 0.2710 | 0.2717 | 0.2710 | 0.2710 | 0.2710 |
| malunet | 0.3699 | 0.3668 | 0.3674 | 0.3668 | 0.3668 | 0.3668 |

R=512: baselines collapsed (IoU <= chance+0.05) 0/10 (ok 10, unknown 0); pbe holds (> chance+0.10)
R=1024: baselines collapsed (IoU <= chance+0.05) 0/10 (ok 10, unknown 0); pbe holds (> chance+0.10)
R=2048: baselines collapsed (IoU <= chance+0.05) 0/10 (ok 10, unknown 0); pbe holds (> chance+0.10)
Mechanical verdict (PREREG accuracy rule, SECONDARY candidate): REJECT_BASELINES_DO_NOT_COLLAPSE
