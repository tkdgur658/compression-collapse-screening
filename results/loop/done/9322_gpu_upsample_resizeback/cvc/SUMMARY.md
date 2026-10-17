# SUMMARY (exploratory input-resolution accuracy; machine-written)

dataset=cvc split_k=1 n_val=61; chance IoU all-bg=0.0000 all-fg=0.0778 max=0.0778; package_status=COMPLETE_EXPLORATORY_NO_CONFIRMATION
Input: 256x256 validation image bilinearly upsampled to R (align_corners False), probability map area-downsampled back to 256, threshold 0.5.

| model | R=256 | R=512 | R=768 | R=1024 | R=1536 | R=2048 |
|---|---|---|---|---|---|---|
| pbe | 0.8469 | 0.8446 | 0.8451 | 0.8447 | 0.8446 | 0.8446 |
| maresunet | 0.8159 | 0.8147 | 0.8147 | 0.8147 | 0.8147 | 0.8147 |
| utanet | 0.8114 | 0.8093 | 0.8100 | 0.8093 | 0.8093 | 0.8093 |
| mk_unet | 0.8460 | 0.8429 | 0.8435 | 0.8429 | 0.8429 | 0.8429 |
| mhorunet | 0.7076 | 0.7064 | 0.7068 | 0.7064 | 0.7064 | 0.7065 |
| tinyunet | 0.8158 | 0.8083 | 0.8098 | 0.8083 | 0.8083 | 0.8083 |
| ucmnet | 0.6588 | 0.6641 | 0.6639 | 0.6641 | 0.6641 | 0.6641 |
| lbunet | 0.6424 | 0.6406 | 0.6410 | 0.6406 | 0.6406 | 0.6406 |
| cmunext | 0.7226 | 0.7337 | 0.7327 | 0.7337 | 0.7337 | 0.7337 |
| egeunet | 0.5799 | 0.5818 | 0.5813 | 0.5818 | 0.5818 | 0.5818 |
| malunet | 0.5722 | 0.5750 | 0.5754 | 0.5750 | 0.5750 | 0.5750 |

R=512: baselines collapsed (IoU <= chance+0.05) 0/10 (ok 10, unknown 0); pbe holds (> chance+0.10)
R=1024: baselines collapsed (IoU <= chance+0.05) 0/10 (ok 10, unknown 0); pbe holds (> chance+0.10)
R=2048: baselines collapsed (IoU <= chance+0.05) 0/10 (ok 10, unknown 0); pbe holds (> chance+0.10)
Mechanical verdict (PREREG accuracy rule, SECONDARY candidate): REJECT_BASELINES_DO_NOT_COLLAPSE
