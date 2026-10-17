# JETSON_RESULT: Jetson Nano latency of the 11 models (PREREG_JETSON, run r1, 2026-10-16)

Pre-registration: `PREREG_JETSON.md` (sha256 6afdf44545f203db7dd82c3c68d3eb02eb28024ca17ba5745e51ccd714fa96d4, not edited). Machine-readable result: `JETSON_RESULT.json`. Raw files: `raw/jetson/` (trtexec build, output-check and timing logs, `exportTimes` json, per-attempt pre-condition and clock snapshots, driver log) and `raw/gpu74_logs/` (export logs).

## Outcome

**No model has a reportable Jetson latency under the frozen pre-registration. All 11 rows are n.m.**

- 6 models (PBE-Net, UTANet, MK-UNet, UCMNet, CMUNeXt, MALUNet): FP16 engine built and ran, but the clock state changed between the snapshot immediately before and immediately after the block in all 3 attempts (1 + 2 re-measurements), so every block was discarded and the model is **unstable** (PREREG study lead decision 2).
- 2 models (MAResUNet, MHorUNet): the fused graph cannot be exported to ONNX **opset 11** (`aten::einsum` exists in ONNX from opset 12, `aten::fft_rfft2` has no opset-11 export). No substitute opset was used.
- 3 models (TinyU-Net, LB-UNet, EGE-UNet): the ONNX graph exported, but the **TensorRT 8.2.1** ONNX parser of JetPack R32.7.6 rejects it (cubic `Resize`; `Pad` with pads computed as a shape tensor; `If` node with outputs of different shape). No graph rewriting or simplifier was used.
- FP32 secondary row: not built. The pre-registered condition (an FP32 engine builds and runs for all 11 models) cannot hold, because 5 models have no ONNX file or no parsable graph. FP16 only.

## Why every block changed its clocks (the cause of "unstable")

Clocks are not fixed (no `jetson_clocks`, decision 2), so the device runs its dynamic frequency governors: GPU `nvhost_podgov` (76.8 to 921.6 MHz, polling 25 ms), CPU `schedutil` (scaling_cur_freq 1224 to 1479 MHz, and scaling_min_freq toggles between 1200000 and 102000 kHz on its own). Before every block the device is idle (pre-condition), so the GPU is at 76.8 MHz. Immediately after the block it is at 921.6 MHz (844.8 MHz once). The before/after rule therefore cannot be met by any block with unfixed clocks. This is a property of the pre-registered combination "clocks not fixed" + "discard if any clock changes", not a disturbance by another process: every pre-condition passed with no other user session, GR3D 0% in all 5 tegrastats samples and 1-min load < 0.5, and GPU min/max/governor and CPU max/governor were identical before and after every block. EMC frequency is not readable without root (tegrastats prints EMC utilisation only; `/sys/kernel/debug/clk` permission denied), recorded as "not readable".

## Decision needed (study lead)

The discarded blocks are kept below as a descriptive record. They are not reportable under the frozen prereg and are not in the manuscript. They are very reproducible across the three attempts (e.g. PBE-Net 108.3 / 108.8 / 108.5 ms). The first warm-up run is 0.2 to 2.3 s (GPU clock ramp and lazy initialisation), and the 10 warm-up runs absorb it. Options for an amended pre-registration (not done here): (a) compare only the clock configuration (governor, min/max) before and after, with the instantaneous frequencies recorded, (b) fix clocks with `jetson_clocks` (rejected in decision 2), (c) keep the column deleted (state of Pass 22). The 5 tooling failures would need a separate decision (opset 12+ export, or graph simplification), which also deviates from the frozen opset 11.

## Procedure as run

1. **Fused graph and ONNX (gpu-server, CPU only).** `<WORK>/jetson/` with copies of `code/pbe_seg`, `f2b_fold_all.py` (sha256 9ad43a00...) and `f2_fuse.py` (a1ef95d3...), imported unchanged. Checkpoints PH2 k1 as in F2b (sha256 identical to the local files). `CUDA_VISIBLE_DEVICES=""`, OMP/MKL/OpenBLAS 2 threads, `nice -n 10`, `python -I -B`, env `Skin_Lesion_Seg` (torch 2.0.1+cu118, Python 3.8.20). No GPU of gpu-server was used and no process on it was touched. Script `scripts/jetson_export.py`: generic BN fold (baselines) or `fuse_pbe` (PBE-Net), equivalence on the first 4 PH2 test images in fp32 and fp64 exactly as in F2b, export `torch.onnx.export(..., opset_version=11)` of `adapter.infer`, input 1x3x256x256, batch 1, single output.
2. **ONNX check (local PC, onnxruntime 1.20.1 CPU, 2 threads).** `scripts/ort_check.py`: ONNX output vs fused PyTorch output on PH2 test image 0. Sign agreement 1.0 for all 9 files, max relative difference at most 8.4e-4 (UCMNet), at most 7.3e-5 for the other 8.
3. **Jetson (`~/lsh_latency_20261016/` only).** `scripts/jet_build.sh`: `trtexec --onnx --fp16 --workspace=1024 --saveEngine --buildOnly` (build not timed, 6 to 20 min per model), then one untimed run with `--loadInputs`/`--exportOutput` on the same image. `scripts/jet_measure.sh` (same for all models): pre-conditions (`who`, `timeout 6 tegrastats --interval 1000`, `/proc/loadavg`, `nvpmodel -q`, 30 s wait if not met), clock snapshot, `trtexec --loadEngine --noDataTransfers --warmUp=0 --duration=0 --iterations=60 --exportTimes`, clock snapshot, compare. trtexec 8.2.1 flags checked with `--help` first. Every timing file had exactly 60 entries. `scripts/jet_analyze.py` drops the first 10 entries and summarises `computeMs` of the remaining 50.
4. No sudo, no credential entry, no `jetson_clocks`, no nvpmodel change, no package install, nothing under /etc, no service change. Nothing was deleted on any machine.

## Pre-condition and clock record (18 blocks, 2026-10-16 20:52-21:03 KST)

- `who`: only `<device-user> :0 2026-08-10 14:49 (:0)` in all 18 checks. This is the persistent GDM desktop session of the device account (idle, open since boot), not another user. Our ssh commands have no tty and do not appear. Prereg item 1 ("no other user session doing work") was read as met. The script would have postponed on any other line.
- tegrastats: 5 samples, GR3D_FREQ 0% in every sample of every check. Load average (1 min) at the accepted check: 0.30 to 0.49. 9 checks in 7 blocks failed on load (0.51 to 0.79, after the previous block) and the block started after 1 to 3 waits of 30 s.
- `nvpmodel -q`: "NV Power Mode: MAXN", mode 0, with "NVPM WARN: fan mode is not set!" and an EMC-cap read error (permission 13), the same in all checks.
- Temperatures before the blocks: 55.5 to 59.5 °C (thermal zone 0).

## Deviations and notes

- **ONNX exporter post-processing.** The python package `onnx` is not installed in any environment on gpu-server or on the local PC, and no install is possible (no network). torch 2.0.1 `torch.onnx.export` calls `onnx_proto_utils._add_onnxscript_fn`, which imports `onnx` only to insert onnx-script functions for nodes of a custom domain. It returns the bytes unchanged otherwise (torch source lines 212-244, read on gpu-server). This function was replaced in the export process by the identity, and every graph was verified to import only the default ONNX domain at version 11 (C++ pretty print, files `onnx_op11/*.pretty.txt`). The ONNX files are therefore the same as with `onnx` installed. Run 1 (output directory named `onnx/`, which shadowed the package) and run 2 (without the replacement) failed and are kept as logs.
- Equivalence values are from torch 2.0.1 on gpu-server and differ slightly from F2b (torch 2.4.1 on the local PC). For example, PBE-Net fp32 is 4.6e-5 here against 6.1e-5 in F2b, with sign agreement 1.0 in both.
- `--workspace=1024` (MiB) was chosen because the prereg does not fix it (trtexec default 16 MiB). It is the same for all models.
- Prereg "equivalence reported in an appendix table row per model": values are in `JETSON_RESULT.json`. No appendix row was added because the manuscript was not edited.
- Rule slip: I stopped the local ssh client that launched the Jetson driver with `kill %1` (job number, my own process), not with an explicit PID. The remote driver was not affected.
- Local scratch file `/tmp/loc.txt` (checkpoint sha list) was written outside the <scratch>. It was not deleted.
- Manuscript copy: not changed. `tools/gen_jetson_col.py` (new) builds the Jetson column, caption footnote and n.m. marks from this json only. Its dry run (preview in the session <scratch>) shows 11 x n.m., and `--apply` refuses when no model is reportable.

## Per-model record

Status, hashes (first 16 hex; full values in the json) and equivalence. Fused vs training = F2b check on 4 PH2 test images. ONNX and engine checks are on 1 image.

| model | status | reason (short) | ONNX sha256 | engine FP16 sha256 | fused vs training fp32 max abs (rel) / fp64 | ONNX vs fused (rel) | engine FP16 vs fused fp32 (rel, sign agree) |
|---|---|---|---|---|---|---|---|
| PBE-Net | n.m. | unstable (clock rule) | 604f7944927763f6 | 79b617126c78f18a | 4.6e-05 (5.5e-07) / 1.3e-13 | 9.2e-07 | 1.0e-03, 0.99998 |
| MAResUNet | n.m. | ONNX opset 11 export: aten::einsum | - | - | 1.8e-05 (9.5e-07) / 2.5e-14 | - | - |
| UTANet | n.m. | unstable (clock rule) | b8ae343cae5ebf0e | f11dc9f3eedffe57 | 9.1e-06 (1.3e-06) / 9.3e-15 | 8.5e-07 | 6.2e-03, 0.99992 |
| MK-UNet | n.m. | unstable (clock rule) | 33b9ffc210f17ccd | 901b63dbae7168b5 | 7.2e-06 (5.8e-07) / 1.2e-14 | 4.0e-07 | 7.2e-04, 1.00000 |
| MHorUNet | n.m. | ONNX opset 11 export: aten::fft_rfft2 | - | - | 0.0e+00 (0.0e+00) / 0.0e+00 | - | - |
| TinyU-Net | n.m. | TensorRT parser: cubic Resize | 5d27a1bdfc4846d7 | - | 1.6e-05 (1.5e-06) / 3.8e-14 | 1.6e-06 | - |
| UCMNet | n.m. | unstable (clock rule) | 35efbf7bbcdf05d2 | 0aa896139e4cc600 | 0.0e+00 (0.0e+00) / 0.0e+00 | 8.4e-04 | 5.8e-03, 0.99997 |
| LB-UNet | n.m. | TensorRT parser: Pad shape tensor | 770068da2148a841 | - | 7.6e-06 (5.4e-07) / 1.2e-14 | 7.3e-05 | - |
| CMUNeXt | n.m. | unstable (clock rule) | f697012ad8964065 | cc2188da8ff41ef9 | 2.3e-05 (2.9e-06) / 3.8e-14 | 1.7e-06 | 8.9e-03, 0.99998 |
| EGE-UNet | n.m. | TensorRT parser: If node | 8201f70c8d59f3af | - | 0.0e+00 (0.0e+00) / 0.0e+00 | 4.9e-05 | - |
| MALUNet | n.m. | unstable (clock rule) | 5ff1e7bee989b24e | 70b246002586c314 | 0.0e+00 (0.0e+00) / 0.0e+00 | 2.7e-05 | 7.7e-04, 0.99998 |

## Clock snapshots and DISCARDED blocks (descriptive only, not reportable, not in the manuscript)

Statistics of the 50 timed runs (entries 11-60) of each discarded block, GPU compute time.

| model | attempt | GPU cur_freq before -> after (MHz) | CPU0 cur before -> after (MHz) | changed keys | DISCARDED block: median [IQR] ms | min-max ms | first warm-up run ms |
|---|---|---|---|---|---|---|---|
| PBE-Net | a1 | 76.8 -> 921.6 | 1224 -> 1479 | 9 (cpu0_cur,cpu0_min,cpu1_cur,cpu1_min,cpu2_cur,cpu2_min,cpu3_cur,cpu3_min,gpu_cur) | 108.29 [0.48] | 107.94-109.79 | 1751.5 |
| PBE-Net | a2 | 76.8 -> 921.6 | 1224 -> 1479 | 9 (cpu0_cur,cpu0_min,cpu1_cur,cpu1_min,cpu2_cur,cpu2_min,cpu3_cur,cpu3_min,gpu_cur) | 108.76 [0.56] | 108.18-109.22 | 1743.6 |
| PBE-Net | a3 | 76.8 -> 921.6 | 1326 -> 1479 | 7 (cpu0_cur,cpu1_cur,cpu2_cur,cpu2_min,cpu3_cur,cpu3_min,gpu_cur) | 108.45 [0.49] | 108.19-109.54 | 1744.3 |
| UTANet | a1 | 76.8 -> 921.6 | 1428 -> 1479 | 9 (cpu0_cur,cpu0_min,cpu1_cur,cpu1_min,cpu2_cur,cpu2_min,cpu3_cur,cpu3_min,gpu_cur) | 98.41 [0.73] | 97.52-100.30 | 215.2 |
| UTANet | a2 | 76.8 -> 921.6 | 1224 -> 1479 | 9 (cpu0_cur,cpu0_min,cpu1_cur,cpu1_min,cpu2_cur,cpu2_min,cpu3_cur,cpu3_min,gpu_cur) | 98.40 [0.59] | 97.58-99.16 | 180.6 |
| UTANet | a3 | 76.8 -> 921.6 | 1224 -> 1479 | 5 (cpu0_cur,cpu1_cur,cpu2_cur,cpu3_cur,gpu_cur) | 98.44 [0.57] | 97.59-99.15 | 256.6 |
| MK-UNet | a1 | 76.8 -> 921.6 | 1224 -> 1479 | 9 (cpu0_cur,cpu0_min,cpu1_cur,cpu1_min,cpu2_cur,cpu2_min,cpu3_cur,cpu3_min,gpu_cur) | 41.84 [0.25] | 41.58-42.78 | 1714.9 |
| MK-UNet | a2 | 76.8 -> 921.6 | 1224 -> 1479 | 9 (cpu0_cur,cpu0_min,cpu1_cur,cpu1_min,cpu2_cur,cpu2_min,cpu3_cur,cpu3_min,gpu_cur) | 41.80 [0.18] | 41.50-43.13 | 1708.4 |
| MK-UNet | a3 | 76.8 -> 921.6 | 1224 -> 1479 | 5 (cpu0_cur,cpu1_cur,cpu2_cur,cpu3_cur,gpu_cur) | 41.82 [0.25] | 41.51-42.32 | 1731.1 |
| UCMNet | a1 | 76.8 -> 921.6 | 1224 -> 1479 | 5 (cpu0_cur,cpu1_cur,cpu2_cur,cpu3_cur,gpu_cur) | 20.15 [0.67] | 19.86-31.84 | 2315.8 |
| UCMNet | a2 | 76.8 -> 921.6 | 1479 -> 1479 | 8 (cpu0_min,cpu1_cur,cpu1_min,cpu2_cur,cpu2_min,cpu3_cur,cpu3_min,gpu_cur) | 19.88 [0.02] | 19.84-20.03 | 1968.7 |
| UCMNet | a3 | 76.8 -> 921.6 | 1224 -> 1479 | 9 (cpu0_cur,cpu0_min,cpu1_cur,cpu1_min,cpu2_cur,cpu2_min,cpu3_cur,cpu3_min,gpu_cur) | 20.06 [9.89] | 19.88-30.52 | 2205.3 |
| CMUNeXt | a1 | 76.8 -> 921.6 | 1479 -> 1479 | 1 (gpu_cur) | 43.31 [0.07] | 43.20-43.63 | 2015.8 |
| CMUNeXt | a2 | 76.8 -> 921.6 | 1428 -> 1479 | 9 (cpu0_cur,cpu0_min,cpu1_cur,cpu1_min,cpu2_cur,cpu2_min,cpu3_cur,cpu3_min,gpu_cur) | 43.29 [0.08] | 43.18-43.50 | 1860.0 |
| CMUNeXt | a3 | 76.8 -> 921.6 | 1326 -> 1479 | 5 (cpu0_cur,cpu1_cur,cpu2_cur,cpu3_cur,gpu_cur) | 43.29 [0.07] | 43.13-43.53 | 1582.7 |
| MALUNet | a1 | 76.8 -> 921.6 | 1224 -> 1479 | 5 (cpu0_cur,cpu1_cur,cpu2_cur,cpu3_cur,gpu_cur) | 20.58 [0.18] | 20.27-21.17 | 2148.8 |
| MALUNet | a2 | 76.8 -> 844.8 | 1326 -> 1479 | 5 (cpu0_cur,cpu1_cur,cpu2_cur,cpu3_cur,gpu_cur) | 20.95 [0.24] | 20.63-32.47 | 2136.0 |
| MALUNet | a3 | 76.8 -> 921.6 | 1428 -> 1479 | 9 (cpu0_cur,cpu0_min,cpu1_cur,cpu1_min,cpu2_cur,cpu2_min,cpu3_cur,cpu3_min,gpu_cur) | 20.84 [0.15] | 20.53-21.95 | 2222.8 |

## File hashes (sha256)

```
ed3e719e8efb9a1b2581dd5cbe8db621ca1ac503ca70b8b6d905c163107e31dc *JETSON_RESULT.json
f697012ad896406546cf947e431d53433c99c3272452ecab176714f9eeac2d2b *onnx_op11/cmunext_fused_op11.onnx
8201f70c8d59f3af8d9c1c23faa68ba82c05f44482bc0bb0180d7a81946329a5 *onnx_op11/egeunet_fused_op11.onnx
770068da2148a841a8765f2065f43fc02128cd76cdcb1115ad35f9b32229c2e3 *onnx_op11/lbunet_fused_op11.onnx
5ff1e7bee989b24e587e2387ffd6acb02d5ab6ebbcf8f36046a9a82430c504fd *onnx_op11/malunet_fused_op11.onnx
33b9ffc210f17ccde1ef09661317a199c15bb1b681decf001e9c19346bfcb4b0 *onnx_op11/mk_unet_fused_op11.onnx
604f7944927763f6994f270afe6bb3c1c0e2cb699ad10de71b0dbb0caaf1c1c9 *onnx_op11/pbe_fused_op11.onnx
5d27a1bdfc4846d7abdffbbf6d6c26c6e0d57f80233069a20291c749c02df404 *onnx_op11/tinyunet_fused_op11.onnx
35efbf7bbcdf05d201904a34be82322f12f1dfdeaf07166c2972cbb6ee3ad016 *onnx_op11/ucmnet_fused_op11.onnx
b8ae343cae5ebf0ee6cad4f846e3a174cdc47f8124246e0dff3537c50afc48a5 *onnx_op11/utanet_fused_op11.onnx
b1d372b39cacbc2be4a49044074f171f31a6f2194b95c20133c9fa883cf14cd3 *scripts/jet_analyze.py
aa9eb9a4f04ed0cd559622bcfd664e7897692ca982c934b474685d74cebe9493 *scripts/jet_build.sh
ea1d769c07ce65fc2cb08bd111e8499bd9f055830f10fc430591056851c657a4 *scripts/jet_measure.sh
c13dde86d0d104281355489faf996ac2a9d6a774fe8071b77761f8c7daefb836 *scripts/jet_tables_md.py
e18a3f8b6e59171adc4ce9ef3b6fe2fd4586dbc53b16695ec3ba2cf1fc7d0bba *scripts/jetson_export.py
1f22b7c3708c5c54e92fdb395b80f4e32033bfc024fba999780a3601c2888bdd *scripts/ort_check.py
cc2188da8ff41ef98566a484baea507d627c644d6c54d959c97ff9804680bbd8  engines/cmunext_fp16.engine
70b246002586c3145bbba0c206e08fae1421ca42dcbdc68470626d0050195f4f  engines/malunet_fp16.engine
901b63dbae7168b565fd3a03676d97b9dcba56344806ecd3f5e7d3a8acd4b29a  engines/mk_unet_fp16.engine
79b617126c78f18ac58760c0d84369217633e6124ada538d093db0fb9a69c7fd  engines/pbe_fp16.engine
0aa896139e4cc6003fd830fb34d739f030f076f6f187dde6cdef60a5f44f61d8  engines/ucmnet_fp16.engine
f11dc9f3eedffe573cbf08e04f232093de560baab00d3204ee764295cfa6c3bf  engines/utanet_fp16.engine
```
