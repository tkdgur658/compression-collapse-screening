# pbe_seg

Clean, reproducible code for "Predictive Boundary Estimator with Band-Limited Cross-Covariance
Matching for On-Device Medical Image Segmentation" (JBHI resubmission).

```
main.py               single entry: rows x datasets x splits (plan by default, --execute to run)
PROTOCOL.md           pre-registered protocol v1
configs/rows.json     table rows (model + config overrides), suites: main, cumulative
common/               shared data pipeline, losses, evaluator, trainer, provenance
models/<name>/        one package per model: adapter.py, src/, manifest.json (+ README, fairness.json)
models/registry.py    the only list of registered models
tools/verify_legacy.py  bit-exact check of PBE-Net against the legacy implementation
tools/deploy.sh       deploy committed HEAD to the GPU server
```

## Usage (on gpu-server, GPU 1 only)
```bash
cd <ARCHIVE>/code
PY=python
$PY main.py --suite main                                    # print the plan
CUDA_VISIBLE_DEVICES=1 $PY main.py --suite main --stage train --execute \
    --run-root ../runs --cache-root ../cache --slot 0 --nslot 2
```
Runs land in `../runs/<suite>/<row>/<dataset>/k<k>/results.json` with the deployed commit,
a sha256 over all sources, the environment, the config hash and per-image test IoU.

## Relation to the legacy repository
The legacy repository `gpu-server:<LEGACY_REPO>` is untouched and
remains the provenance of every pre-v1 result. PBE-Net here reproduces the legacy
`Phase57_LapSkipSup` + DW-Sep bottleneck exactly (same state_dict, identical outputs and losses
on a legacy checkpoint; `tools/verify_legacy.py`). Baseline sources are byte-identical copies of the
legacy `models/*.py` (sha256 in each manifest). Protocol v1 deliberately differs from the legacy
training scripts (see PROTOCOL.md), so v1 numbers are new results, not re-labelled old ones.
