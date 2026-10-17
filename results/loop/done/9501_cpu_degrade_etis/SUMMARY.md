# SUMMARY (exploratory acquisition-degradation collapse search; machine-written)

dataset=etis n_val_full=19 n_used=19 (strided subset) models=11 batch=1 CPU val split k=1 only; package_status=COMPLETE
chance IoU (on the used images): all-bg=0.0000 all-fg=0.0255 max=0.0255
baseline "fell" = identity > chance+0.05 and condition <= chance+0.05; PBE retains = identity > chance+0.10 and condition >= chance+0.10; C_CANDIDATE = >= 8 of 10 baselines fell and PBE retains; OBSERVATION = >= 8 fell and PBE does not retain

mean val IoU (E = error; - = not run)

| model | identity | gauss_s005 | gauss_s01 | gauss_s02 | gauss_s04 | poisson_p30 | poisson_p10 | poisson_p3 | poisson_p1 | sp_5 | sp_20 | sp_50 | jpeg_q50 | jpeg_q20 | jpeg_q10 | jpeg_q5 | gblur_s1 | gblur_s2 | gblur_s4 | gblur_s8 | mblur_l9 | mblur_l21 | mblur_l41 | dsup_16 | occl_25 | occl_50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pbe | 0.763 | 0.044 | 0.010 | 0.001 | 0.001 | 0.001 | 0.000 | 0.006 | 0.003 | 0.013 | 0.002 | 0.014 | 0.653 | 0.536 | 0.464 | 0.202 | 0.606 | 0.272 | 0.102 | 0.059 | 0.517 | 0.311 | 0.195 | 0.062 | 0.122 | 0.041 |
| maresunet | 0.673 | 0.651 | 0.448 | 0.138 | 0.010 | 0.324 | 0.025 | 0.000 | 0.000 | 0.252 | 0.008 | 0.000 | 0.636 | 0.526 | 0.427 | 0.223 | 0.658 | 0.584 | 0.496 | 0.154 | 0.603 | 0.558 | 0.270 | 0.182 | 0.161 | 0.084 |
| utanet | 0.380 | 0.041 | 0.019 | 0.001 | 0.003 | 0.002 | 0.000 | 0.005 | 0.035 | 0.012 | 0.000 | 0.003 | 0.358 | 0.318 | 0.219 | 0.116 | 0.318 | 0.278 | 0.254 | 0.123 | 0.285 | 0.218 | 0.144 | 0.122 | 0.086 | 0.036 |
| mk_unet | 0.675 | 0.222 | 0.068 | 0.028 | 0.053 | 0.037 | 0.012 | 0.016 | 0.007 | 0.103 | 0.106 | 0.094 | 0.602 | 0.440 | 0.339 | 0.229 | 0.625 | 0.409 | 0.265 | 0.096 | 0.460 | 0.330 | 0.142 | 0.078 | 0.237 | 0.103 |
| mhorunet | 0.598 | 0.578 | 0.394 | 0.083 | 0.000 | 0.290 | 0.096 | 0.004 | 0.004 | 0.103 | 0.000 | 0.000 | 0.568 | 0.500 | 0.448 | 0.330 | 0.598 | 0.566 | 0.477 | 0.317 | 0.570 | 0.493 | 0.298 | 0.313 | 0.112 | 0.020 |
| tinyunet | 0.640 | 0.000 | 0.000 | 0.009 | 0.007 | 0.003 | 0.009 | 0.002 | 0.001 | 0.001 | 0.000 | 0.000 | 0.576 | 0.440 | 0.302 | 0.182 | 0.498 | 0.261 | 0.171 | 0.042 | 0.379 | 0.243 | 0.107 | 0.058 | 0.174 | 0.086 |
| ucmnet | 0.581 | 0.437 | 0.284 | 0.147 | 0.052 | 0.200 | 0.119 | 0.008 | 0.001 | 0.181 | 0.033 | 0.025 | 0.529 | 0.409 | 0.458 | 0.191 | 0.568 | 0.550 | 0.493 | 0.241 | 0.543 | 0.494 | 0.225 | 0.252 | 0.060 | 0.006 |
| lbunet | 0.386 | 0.297 | 0.174 | 0.064 | 0.039 | 0.082 | 0.048 | 0.042 | 0.043 | 0.109 | 0.045 | 0.030 | 0.386 | 0.350 | 0.340 | 0.130 | 0.381 | 0.371 | 0.300 | 0.114 | 0.384 | 0.343 | 0.193 | 0.111 | 0.030 | 0.015 |
| cmunext | 0.309 | 0.047 | 0.075 | 0.003 | 0.000 | 0.040 | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.001 | 0.234 | 0.275 | 0.211 | 0.191 | 0.133 | 0.095 | 0.069 | 0.057 | 0.130 | 0.095 | 0.063 | 0.061 | 0.074 | 0.066 |
| egeunet | 0.290 | 0.232 | 0.162 | 0.178 | 0.042 | 0.120 | 0.144 | 0.000 | 0.000 | 0.172 | 0.074 | 0.000 | 0.286 | 0.301 | 0.241 | 0.226 | 0.287 | 0.294 | 0.271 | 0.174 | 0.308 | 0.262 | 0.161 | 0.180 | 0.056 | 0.012 |
| malunet | 0.383 | 0.325 | 0.189 | 0.015 | 0.000 | 0.118 | 0.025 | 0.000 | 0.000 | 0.031 | 0.000 | 0.000 | 0.368 | 0.360 | 0.351 | 0.314 | 0.334 | 0.318 | 0.237 | 0.135 | 0.304 | 0.244 | 0.168 | 0.146 | 0.060 | 0.027 |

NaN/Inf logits: none

| condition | baselines fell (of 10) | healthy at identity | undetermined | PBE retains | verdict |
|---|---|---|---|---|---|
| gauss_s005 | 3 | 10 | 0 | False | REJECT |
| gauss_s01 | 4 | 10 | 0 | False | REJECT |
| gauss_s02 | 6 | 10 | 0 | False | REJECT |
| gauss_s04 | 10 | 10 | 0 | False | OBSERVATION |
| poisson_p30 | 4 | 10 | 0 | False | REJECT |
| poisson_p10 | 7 | 10 | 0 | False | REJECT |
| poisson_p3 | 10 | 10 | 0 | False | OBSERVATION |
| poisson_p1 | 10 | 10 | 0 | False | OBSERVATION |
| sp_5 | 4 | 10 | 0 | False | REJECT |
| sp_20 | 9 | 10 | 0 | False | OBSERVATION |
| sp_50 | 9 | 10 | 0 | False | OBSERVATION |
| jpeg_q50 | 0 | 10 | 0 | True | REJECT |
| jpeg_q20 | 0 | 10 | 0 | True | REJECT |
| jpeg_q10 | 0 | 10 | 0 | True | REJECT |
| jpeg_q5 | 0 | 10 | 0 | True | REJECT |
| gblur_s1 | 0 | 10 | 0 | True | REJECT |
| gblur_s2 | 0 | 10 | 0 | True | REJECT |
| gblur_s4 | 1 | 10 | 0 | False | REJECT |
| gblur_s8 | 2 | 10 | 0 | False | REJECT |
| mblur_l9 | 0 | 10 | 0 | True | REJECT |
| mblur_l21 | 0 | 10 | 0 | True | REJECT |
| mblur_l41 | 1 | 10 | 0 | True | REJECT |
| dsup_16 | 2 | 10 | 0 | False | REJECT |
| occl_25 | 5 | 10 | 0 | False | REJECT |
| occl_50 | 7 | 10 | 0 | False | REJECT |

Package mechanical verdict (order C_CANDIDATE > OBSERVATION > HOLD_INCOMPLETE > REJECT): OBSERVATION
Conditions C_CANDIDATE: none
Conditions OBSERVATION: ['gauss_s04', 'poisson_p3', 'poisson_p1', 'sp_20', 'sp_50']
Conditions HOLD_INCOMPLETE: none
(Cross-dataset rule: a condition is a C lead only if C_CANDIDATE on >= 4 of the 6 dataset packages 9501-9506 incl. >= 1 skin and >= 1 polyp; evaluated by the aggregator.)

Integrity: degradation_determinism_check=True alias_groups_after=210
