# PREREG_JETSON — Jetson Nano latency of the 11 models (FROZEN 2026-10-16 17:3x after study lead confirmation)

Written 2026-10-16 17:2x KST ([internal session]) from [internal decision] items. No measurement yet.

## Device and access
- Jetson Nano `<device-user>-desktop` (<HOST_IP>), reachable only via gpu-server (`ssh gpu-server` → `ssh <device-user>@<HOST_IP>`, key auth). JetPack R32.7.6, TensorRT present (`/usr/src/tensorrt`), ~2.8 GB RAM available at check time (2026-10-16 15:03).
- Known issue: `python3 -c "import torch"` dies with "Illegal instruction". Allowed fix: per-process environment variable only (e.g. `OPENBLAS_CORETYPE=ARMV8`). No system-wide change (no package install into system Python, no edits under /etc, no service changes).

## Pre-conditions (checked immediately before each model's block; else wait and re-check)
1. No other user session doing work: if another user is logged in (`who`), postpone the measurement.
2. `tegrastats` sample (5 s): no other GPU process, CPU load average < 0.5.
3. Power mode recorded with `nvpmodel -q` (MAXN expected); clocks fixed with `jetson_clocks` only if allowed without system-wide persistent change — otherwise record the current clock state and do not change it (decision: record only, no `jetson_clocks`).

## Graph and runtime (pre-selected)
- Graph: deployment graph = fused (BatchNorm folded, RepConv merged) for every model, built on gpu-server with the F2/F2b fusion code, equivalence checked before export.
- Runtime: TensorRT FP16 engine (primary) built from ONNX (opset 11) per model at input 1×3×256×256, batch 1. FP32 TensorRT as a secondary row only if engine build time and memory allow for all 11 models; otherwise FP16 only (stated).
- Same script for all 11 models; engine build is excluded from timing.

## Timing
- Warm-up 10 inferences, then 50 timed inferences with device synchronization; report median and IQR (ms) per model; also record min/max.
- Timing covers device-side inference only (input already on device); host-to-device copy reported separately if measured.

## Output
- `iso/s4/jetson/JETSON_RESULT.md` + json (per model: median, IQR, min, max, power mode, clock state, TensorRT version, precision, ONNX/engine sha256).
- Manuscript: Table 2 Jetson column restored from this file only.

## study lead decisions (2026-10-16 17:3x, [id]) — frozen
- TensorRT FP16 is the primary row. FP32 is a secondary row only if an FP32 engine builds and runs for all 11 models; otherwise FP16 only, with the reason recorded.
- `jetson_clocks` is not used (system-wide change, affects other users). Clock state (GPU/CPU/EMC frequencies from `tegrastats` or `/sys/kernel/debug/clk` read-only) is recorded immediately before and after each model block; if any clock changes during a block, that block is discarded and re-measured (max 2 re-measurements per model, then reported as unstable).
- The postpone-if-another-user-is-logged-in rule stays.
- Equivalence of the fused graph is reported in an appendix table row per model.
