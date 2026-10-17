# Jetson Nano re-measurement under AMENDMENT 1 (clocks fixed by the user)

TensorRT 8.2.1 FP16, fused graph, ONNX opset 11, batch 1, 256x256, warm-up 10 + 50 timed (GPU compute time, --noDataTransfers).

| model | status | median (ms) | IQR (ms) | min | max |
|---|---|---|---|---|---|
| PBE-Net | measured (a1) | 108.56 | 0.49 | 107.96 | 109.13 |
| UTANet | measured (a1) | 98.53 | 0.59 | 97.35 | 100.00 |
| MK-UNet | measured (a1) | 41.85 | 0.27 | 41.56 | 42.41 |
| UCMNet | measured (a1) | 19.94 | 0.23 | 19.86 | 30.47 |
| CMUNeXt | measured (a1) | 43.33 | 0.07 | 43.25 | 43.52 |
| MALUNet | measured (a1) | 21.08 | 0.46 | 20.84 | 31.22 |

Not attempted (tooling failures, n.m. as in JETSON_RESULT.md): MAResUNet, MHorUNet (ONNX opset 11), TinyU-Net, LB-UNet, EGE-UNet (TensorRT 8.2.1 parser).
