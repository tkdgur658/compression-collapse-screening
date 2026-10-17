# PREREG_JETSON — AMENDMENT 1 (frozen 2026-10-16, before the re-measurement)

Parent: PREREG_JETSON.md (frozen sha 6afdf445…, not edited). First run: JETSON_RESULT.json (sha ed3e719e…): 0 of 11 reportable; 6 models unstable because the GPU/CPU governors changed the clock during every block (clocks not fixed), 3 rejected by TensorRT 8.2.1, 2 without ONNX opset 11 export. study lead note D12: the frozen before/after clock rule was structurally impossible on a DVFS device with unfixed clocks.

## Change (one item, approved by [internal decision] and by the user)
- **Clocks are fixed with `jetson_clocks`, run by the user** (sudo, on the device) before the re-measurement, after `sudo jetson_clocks --store`. The user restores them with `sudo jetson_clocks --restore` after the measurement. This replaces the frozen decision "jetson_clocks is not used". Reason for the original ban: a system-wide change affecting other users; the user takes this action and its restore personally.
- Everything else stays as frozen: who/tegrastats/load pre-conditions (postpone if another user is logged in), nvpmodel recorded, same ONNX files and TensorRT FP16 engines (sha as in JETSON_RESULT.json), warm-up 10 + 50 timed runs, median + IQR + min/max, clock state recorded immediately before and after each block, discard the block if any clock changes, max 2 re-measurements per model then "unstable". FP32 row only if all 11 models build (not possible: 5 tooling failures), so FP16 only.
- Scope: the 6 models whose engines built (PBE-Net, UTANet, MK-UNet, UCMNet, CMUNeXt, MALUNet). The 5 tooling failures stay n.m. (no opset change, no graph rewriting).

## Disclosure
- The discarded block medians of the first run were seen before this amendment (e.g. PBE-Net 108.3 / 108.8 / 108.5 ms). The amendment changes only the clock state, not the timing procedure, and is reported as a deviation from the frozen pre-registration in the manuscript note.
- Result once: the re-measurement is run once; its values replace "n.m." for the models that pass the frozen block rule; the others stay n.m. with the reason. No further changes after seeing the values.
