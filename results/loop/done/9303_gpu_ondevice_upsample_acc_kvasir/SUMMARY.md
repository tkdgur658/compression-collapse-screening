# SUMMARY (exploratory input-resolution accuracy; machine-written)

dataset=kvasir split_k=1 n_val=100; chance IoU all-bg=0.0000 all-fg=0.1391 max=0.1391; package_status=COMPLETE_EXPLORATORY_NO_CONFIRMATION
Input: 256x256 validation image bilinearly upsampled to R (align_corners False), probability map area-downsampled back to 256, threshold 0.5.

| model | R=256 | R=512 | R=768 | R=1024 | R=1536 | R=2048 |
|---|---|---|---|---|---|---|
| pbe | 0.8081 | 0.5948 | 0.2607 | 0.1066 | 0.0350 | 0.0273 |
| maresunet | 0.8125 | 0.6590 | 0.3606 | 0.0940 | 0.0203 | 0.0195 |
| utanet | 0.8116 | 0.5467 | 0.2866 | 0.1446 | 0.0956 | 0.1184 |
| mk_unet | 0.8060 | 0.6072 | 0.3111 | 0.0684 | 0.0133 | 0.0099 |
| mhorunet | 0.7439 | 0.6405 | 0.5193 | 0.3905 | 0.1874 | 0.1004 |
| tinyunet | 0.7836 | 0.4666 | 0.2338 | 0.1022 | 0.0282 | 0.0245 |
| ucmnet | 0.7047 | 0.6047 | 0.4689 | 0.3586 | 0.2650 | 0.2227 |
| lbunet | 0.6760 | 0.5535 | 0.4025 | 0.2747 | 0.1271 | 0.0718 |
| cmunext | 0.7840 | 0.3845 | 0.2461 | 0.1911 | 0.1808 | 0.1890 |
| egeunet | 0.6413 | 0.4883 | 0.3092 | 0.1892 | 0.1031 | 0.0695 |
| malunet | 0.6860 | 0.5145 | 0.3047 | 0.1701 | 0.0576 | 0.0270 |

R=512: baselines collapsed (IoU <= chance+0.05) 0/10 (ok 10, unknown 0); pbe holds (> chance+0.10)
R=1024: baselines collapsed (IoU <= chance+0.05) 5/10 (ok 5, unknown 0); pbe collapses (> chance+0.10)
R=2048: baselines collapsed (IoU <= chance+0.05) 9/10 (ok 1, unknown 0); pbe collapses (> chance+0.10)
Mechanical verdict (PREREG accuracy rule, SECONDARY candidate): OBSERVATION_BASELINES_COLLAPSE_PBE_ALSO_COLLAPSES
