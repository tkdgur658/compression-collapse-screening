# SUMMARY (exploratory input-resolution accuracy; machine-written)

dataset=etis split_k=1 n_val=19; chance IoU all-bg=0.0000 all-fg=0.0255 max=0.0255; package_status=COMPLETE_EXPLORATORY_NO_CONFIRMATION
Input: 256x256 validation image bilinearly upsampled to R (align_corners False), probability map area-downsampled back to 256, threshold 0.5.

| model | R=256 | R=512 | R=768 | R=1024 | R=1536 | R=2048 |
|---|---|---|---|---|---|---|
| pbe | 0.7628 | 0.1508 | 0.0502 | 0.0295 | 0.0254 | 0.0269 |
| maresunet | 0.4964 | 0.2220 | 0.1113 | 0.0887 | 0.0810 | 0.0733 |
| utanet | 0.4566 | 0.1681 | 0.0716 | 0.0488 | 0.0335 | 0.0287 |
| mk_unet | 0.7071 | 0.1695 | 0.0275 | 0.0210 | 0.0190 | 0.0140 |
| mhorunet | 0.5584 | 0.2662 | 0.0780 | 0.0282 | 0.0136 | 0.0134 |
| tinyunet | 0.5866 | 0.1563 | 0.0625 | 0.0404 | 0.0291 | 0.0236 |
| ucmnet | 0.4816 | 0.3564 | 0.1747 | 0.1046 | 0.0396 | 0.0387 |
| lbunet | 0.3800 | 0.2913 | 0.1696 | 0.1326 | 0.1029 | 0.0915 |
| cmunext | 0.3922 | 0.0990 | 0.0411 | 0.0340 | 0.0390 | 0.0384 |
| egeunet | 0.2757 | 0.1154 | 0.0888 | 0.0606 | 0.0446 | 0.0460 |
| malunet | 0.3699 | 0.0959 | 0.0429 | 0.0141 | 0.0162 | 0.0136 |

R=512: baselines collapsed (IoU <= chance+0.05) 0/10 (ok 10, unknown 0); pbe holds (> chance+0.10)
R=1024: baselines collapsed (IoU <= chance+0.05) 7/10 (ok 3, unknown 0); pbe collapses (> chance+0.10)
R=2048: baselines collapsed (IoU <= chance+0.05) 9/10 (ok 1, unknown 0); pbe collapses (> chance+0.10)
Mechanical verdict (PREREG accuracy rule, SECONDARY candidate): OBSERVATION_BASELINES_COLLAPSE_PBE_ALSO_COLLAPSES
