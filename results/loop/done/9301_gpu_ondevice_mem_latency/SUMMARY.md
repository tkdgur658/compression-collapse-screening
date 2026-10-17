# SUMMARY 9301 (exploratory on-device memory feasibility; machine-written)

GPU=NVIDIA GeForce RTX 3090 torch=2.0.1+cu118; peak = torch.cuda.max_memory_allocated (GiB), batch-1, torch.no_grad; OOM(>x) = out of memory at the 14 GiB per-process cap.
3090 peak memory is a PROXY for device memory (allocator, cuDNN/TensorRT workspace and unified-memory differences); a C candidate needs Jetson confirmation. Latency is descriptive (shared GPU).
package_status=INCOMPLETE_SOME_CELLS_NOT_OK

## Peak memory (GiB), fp32
| model | params (inference) | R=256 | R=512 | R=768 | R=1024 | R=1536 | R=2048 |
|---|---|---|---|---|---|---|---|
| pbe | 1,334,337 | 0.164 | 0.525 | 1.162 | 2.056 | 4.610 | 8.185 |
| maresunet | 26,275,065 | 0.241 | 0.271 | 0.364 | 0.565 | 1.139 | 1.942 |
| utanet | 24,075,305 | 0.270 | 0.424 | 0.830 | 1.399 | 3.025 | 5.301 |
| mk_unet | 315,566 | 0.150 | 0.171 | 0.213 | 0.326 | 0.721 | 1.275 |
| mhorunet | 4,962,159 | 0.154 | 0.165 | 0.176 | 0.196 | 0.251 | 0.327 |
| tinyunet | 481,105 | 0.251 | 0.600 | 1.194 | 2.115 | 4.747 | 8.432 |
| ucmnet | 47,436 | 0.143 | 0.173 | 0.223 | 0.292 | 0.497 | 0.877 |
| lbunet | 55,865 | 0.137 | 0.150 | 0.171 | 0.201 | 0.281 | 0.461 |
| cmunext | 144,977 | 0.146 | 0.184 | 0.246 | 0.412 | 0.916 | 1.621 |
| egeunet | 53,374 | 0.141 | 0.153 | 0.168 | 0.193 | 0.377 | 0.663 |
| malunet | 177,943 | 0.136 | 0.144 | 0.157 | 0.174 | 0.225 | 0.306 |

## Latency median of 20 runs (ms, descriptive), fp32
| model | R=256 | R=512 | R=768 | R=1024 | R=1536 | R=2048 |
|---|---|---|---|---|---|---|
| pbe | 8.1 | 26.1 | 38.0 | 95.4 | 216.3 | 379.7 |
| maresunet | 7.5 | 15.8 | 17.0 | 43.5 | 93.7 | 167.4 |
| utanet | 11.2 | 31.0 | 61.7 | 106.6 | 231.9 | 400.1 |
| mk_unet | 7.4 | 20.3 | 27.4 | 38.1 | 171.6 | 302.4 |
| mhorunet | 18.3 | 17.3 | 34.2 | 46.7 | 78.8 | 127.2 |
| tinyunet | 8.0 | 22.1 | 48.5 | 83.4 | 187.8 | 326.9 |
| ucmnet | 3.6 | 7.6 | 12.8 | 22.2 | 49.3 | 87.2 |
| lbunet | 2.8 | 4.7 | 7.0 | 16.5 | 30.6 | 53.0 |
| cmunext | 4.1 | 12.3 | 25.3 | 42.7 | 90.3 | 157.0 |
| egeunet | 6.8 | 7.7 | 11.8 | 16.6 | 33.7 | 56.2 |
| malunet | 4.0 | 7.1 | 8.6 | 13.0 | 26.3 | 44.3 |

## Peak memory (GiB), fp16
| model | params (inference) | R=256 | R=512 | R=768 | R=1024 | R=1536 | R=2048 |
|---|---|---|---|---|---|---|---|
| pbe | 1,334,337 | 0.149 | 0.266 | 0.585 | 1.032 | 2.309 | 4.096 |
| maresunet | 26,275,065 | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR |
| utanet | 24,075,305 | 0.202 | 0.271 | 0.525 | 0.891 | 1.939 | 3.405 |
| mk_unet | 315,566 | 0.142 | 0.165 | 0.205 | 0.261 | 0.419 | 0.641 |
| mhorunet | 4,962,159 | 0.160 | 0.167 | 0.176 | 0.189 | 0.231 | 0.386 |
| tinyunet | 481,105 | 0.192 | 0.366 | 0.656 | 1.062 | 2.378 | 4.220 |
| ucmnet | 47,436 | 0.134 | 0.138 | 0.144 | 0.226 | 0.499 | 0.881 |
| lbunet | 55,865 | 0.135 | 0.140 | 0.150 | 0.163 | 0.199 | 0.250 |
| cmunext | 144,977 | 0.143 | 0.172 | 0.220 | 0.288 | 0.481 | 0.814 |
| egeunet | 53,374 | 0.136 | 0.143 | 0.156 | 0.172 | 0.221 | 0.336 |
| malunet | 177,943 | 0.135 | 0.138 | 0.144 | 0.152 | 0.175 | 0.208 |

## Latency median of 20 runs (ms, descriptive), fp16
| model | R=256 | R=512 | R=768 | R=1024 | R=1536 | R=2048 |
|---|---|---|---|---|---|---|
| pbe | 3.2 | 15.9 | 30.0 | 52.0 | 113.1 | 197.0 |
| maresunet | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR |
| utanet | 10.2 | 13.1 | 26.8 | 68.7 | 148.5 | 258.0 |
| mk_unet | 8.2 | 17.1 | 35.4 | 61.7 | 136.9 | 239.8 |
| mhorunet | 33.5 | 35.0 | 39.5 | 52.2 | 82.5 | 125.7 |
| tinyunet | 7.8 | 17.1 | 39.1 | 68.6 | 149.4 | 262.3 |
| ucmnet | 3.6 | 7.4 | 12.1 | 20.8 | 42.8 | 71.9 |
| lbunet | 3.9 | 7.0 | 8.6 | 13.3 | 29.4 | 47.6 |
| cmunext | 4.3 | 11.6 | 21.1 | 35.0 | 74.7 | 129.8 |
| egeunet | 4.9 | 5.4 | 11.1 | 15.7 | 26.3 | 43.9 |
| malunet | 4.0 | 6.8 | 8.2 | 12.4 | 24.7 | 39.2 |

fp16 mode per model (half = model.half(); autocast fallback): {"cmunext": "half", "egeunet": "half", "lbunet": "half", "malunet": "half", "maresunet": "none_works", "mhorunet": "autocast", "mk_unet": "half", "pbe": "half", "tinyunet": "half", "ucmnet": "half", "utanet": "half"}

analysis_fp32_primary (budget 4.0 GiB):
  R=512: baselines exceeding 0/10 (fit 10, unknown 0); pbe fits
  R=1024: baselines exceeding 0/10 (fit 10, unknown 0); pbe fits
  R=2048: baselines exceeding 2/10 (fit 8, unknown 0); pbe exceeds
  verdict: REJECT_BASELINES_DO_NOT_EXCEED_BUDGET

analysis_fp16_secondary (budget 4.0 GiB):
  R=512: baselines exceeding 0/10 (fit 9, unknown 1); pbe fits
  R=1024: baselines exceeding 0/10 (fit 9, unknown 1); pbe fits
  R=2048: baselines exceeding 1/10 (fit 8, unknown 1); pbe exceeds
  verdict: REJECT_BASELINES_DO_NOT_EXCEED_BUDGET

analysis_fp32_2gib_descriptive (budget 2.0 GiB):
  R=512: baselines exceeding 0/10 (fit 10, unknown 0); pbe fits
  R=1024: baselines exceeding 1/10 (fit 9, unknown 0); pbe exceeds
  R=2048: baselines exceeding 2/10 (fit 8, unknown 0); pbe exceeds
  verdict: REJECT_BASELINES_DO_NOT_EXCEED_BUDGET

Mechanical verdict (PREREG primary rule, FP32, 4 GiB): REJECT_BASELINES_DO_NOT_EXCEED_BUDGET
Non-OK, non-OOM cells: maresunet/fp16/R256=ERROR, maresunet/fp16/R512=ERROR, maresunet/fp16/R768=ERROR, maresunet/fp16/R1024=ERROR, maresunet/fp16/R1536=ERROR, maresunet/fp16/R2048=ERROR
