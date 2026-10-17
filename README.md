# compression-collapse-screening

Code, split lists, scored outputs and pre-registration documents for the manuscript

> **Label-Free Screening of Collapse Across 329 Compressed Medical Segmentation Checkpoints: A PBE-Net Case Study**

The repository reproduces the measurement of the manuscript: 11 segmentation models (PBE-Net and ten baselines) trained under protocol v1 on six public datasets with five splits each, compressed in the deployed (fused) graph by simulated post-training quantization and one-shot pruning, and screened without labels by the agreement between fp32 and compressed outputs on eight unlabeled images. Every compression fact was measured by fake quantization on a GPU. The Jetson Nano scripts measure FP16 latency only.

No image, no mask and no original dataset file is included. The split lists contain image file names only.

## Layout

| Folder | Content |
|---|---|
| `experiments/qfix/A/code/` | Training and evaluation framework of protocol v1 (deployed commit e0e2f07): `main.py`, `common/` (data pipeline, losses, metrics, trainer, provenance), `models/` (PBE-Net in `models/pbe/`, adapters and manifests of the ten baselines), `configs/rows.json`, `PROTOCOL.md` |
| `experiments/qfix/` | QFIX stage A attribution (`A/qfix_a.py`) and compression-aware training C1 (`C/run_C1.sh`, `C/rec/`, `C/code/common/`) |
| `experiments/qdeploy/` | F3: fused deployment graph (RepConv fused, BN folded), settings D (w4 per-channel, a6 per-tensor, l50 layer-wise) and O, retention labels, output agreement, controls, decision (`qdeploy.py`, `qdeploy_analyze.py`, `run_qdeploy.sh`, `run_decide.sh`), checkpoint list `ITEMS.json`, split pairs `FOLD_PAIRS.json`, checkpoint manifest `MANIFEST_330.sha256` |
| `experiments/qdeploy_w8a8/` | 8-bit addendum D_w8a8 |
| `experiments/qdeploy_tau/` | Post hoc threshold (Youden, split 1) and robustness analyses: `tau_youden.py`, `q_robust.py`, `agree_cell_auc.py` |
| `experiments/sgr/` | Repair test of rule R2 (B0 to B5, SGR, ablations, decision) |
| `experiments/theoryq/` | Weight-only prediction test (Propositions 1 to 3, predictors Z and Mstat) |
| `experiments/abl/tools/` | F4 one-factor ablation of PBE-Net |
| `experiments/s4mem/scripts/` | Requirement T / premise M on FIVES (G2 three models, F5 PBE-Net), tiled inference under memory budgets |
| `experiments/jetson/` | ONNX export of the fused graphs and the TensorRT FP16 build and latency scripts of the Jetson Nano |
| `analysis/f5/` | F5 paired bootstrap, summary and synthetic checks |
| `manuscript_tools/tools/` | Generators of every numeric table and of Fig. 1 of the manuscript (read the result files under `results/`, check their sha256, write into `manuscript_tools/tables`, `text`, `figures`) |
| `manuscript_tools/F2_efficiency/` | Parameters, FLOPs and fusion equivalence of the deployed graphs (Table 2) |
| `results/` | Result files read by the generators, the 330 scored per-checkpoint records of F3 (`results/qdeploy/records_p32/`), and the 330 protocol-v1 result files of Table 1 (`results/table1_runs/`, per-image test IoU and test file names) |
| `splits/` | Image file names of every split (`<dataset>.tsv`: dataset, split k, subset, file name) and the FIVES train/val split |
| `prereg/` | The 19 pre-registration documents (see below) |

## Manuscript tables and the files that produce them

| Manuscript item | Measurement code | Result file | Generator |
|---|---|---|---|
| Table 1 (main comparison), Table A (60 comparisons), duplicate-image audit | `experiments/qfix/A/code/main.py --suite main` | `results/table1_runs/` | `gen_table1.py`, `sig_table1.py`, `gen_table1_pvalues.py`, `gen_dup_tables.py`, `dup_audit_server.py` |
| Table 2 (efficiency) | `manuscript_tools/F2_efficiency/f2_flops.py`, `f2_fuse.py`, `f2b_fold_all.py`; RTX 3090 latency and memory: `results/loop/done/9301_*` | `F2_efficiency/*_out*.json` | hand-copied from the JSON files |
| Jetson Nano column, Table tab:app_jetson | `experiments/jetson/` | `results/jetson/JETSON_RESULT*.json` | `gen_jetson_a1.py` |
| Fig. 1 (collapse maps), Table tab:collapse_deploy, 198-cell table | `experiments/qdeploy/` | `results/qdeploy/result/RESULT_QDEPLOY.json`, `records_p32/` | `draw_fig1_collapse_map.py`, `gen_cell_table_d.py` |
| 8-bit row | `experiments/qdeploy_w8a8/` | `results/qdeploy/RESULT_W8A8.json` | text |
| Threshold and robustness tables (tab:app_tau, tab:app_qr_*) | `experiments/qdeploy_tau/` | `results/qdeploy/tau/*.json` | `gen_tau_table.py`, `gen_q_robust_tables.py` |
| Weight-only prediction (tab:app_theoryq) | `experiments/theoryq/` | `results/theoryq/RESULT_THEORYQ.json`, `results/qdeploy/tau/AGREE_CELL_AUC.json` | `gen_theoryq_table.py` |
| Repair test (tab:app_sgr) | `experiments/sgr/` | `results/sgr/RESULT_SGR.json` | `gen_sgr_table.py` |
| Ablation (tab:app_abl) | `experiments/abl/tools/` | `results/abl/result/abl_result.json` | `gen_abl_table.py` |
| Premise M on FIVES (tab:tiling_fives) | `experiments/s4mem/scripts/`, `analysis/f5/` | `results/f5/F5_BOOT.json` | `gen_tab_tiling.py` |
| Exploratory split-1 compression, memory and noise tables (appendix) | earlier exploratory jobs (scripts not included) | `results/loop/done/*/SUMMARY.md` | `gen_appendix_tables.py` |
| PBE-Net configuration table | `manuscript_tools/reference/pbe_seg_98f478d/` (later commit of the PBE-Net files, the table source) | | `gen_pbe_config_table.py` |
| Pre-registration table | | `prereg/SHA256SUMS_original.txt` | `gen_prereg_table.py` |
| Compression-aware training C1 to C3 (tab:collapse_fix) | `experiments/qfix/` | not included | text |

Check of this copy (2026-10-18): all 17 table generators listed above and `draw_fig1_collapse_map.py` were run on this folder with `python -s`. 17 generated tables and `appendix_generated.tex` match the manuscript tables line for line after removing `%` comment lines. `tables/tb_main.tex` has the same numbers, and only its caption wording and one column head differ, since both were edited by hand after generation. `gen_table1.py` writes the table and then stops at a legacy comparison against `tools/tb_main_before.tex.txt` (row "PBE-Net" was named "Proposed" there), which does not affect the table.

## Configuration

Paths are relative to the repository root, and every absolute server path of the original scripts was replaced by one of the variables below (see `config.env.example`).

| Variable | Default | Meaning |
|---|---|---|
| `CCS_WORK` | `experiments/` | root of the experiment folders (the scripts refer to `${CCS_WORK}/qdeploy`, `${CCS_WORK}/qfix/A/code`, ...) |
| `CCS_ARCHIVE` | `archive/` | trained runs and caches: `runs/main/<model>/<dataset>/k<k>/checkpoint.pth`, `results.json`, `cache/v1/` |
| `CCS_DATA` | `datasets/` | dataset folders `<folder>/{train,val,test}_<k>/{images,masks}`, folder names `PH2`, `isic2017`, `isic2018`, `kvasir_seg`, `cvc_clinicdb`, `etis` |
| `CCS_RESULTS` | `results/` | result files read by the generators |
| `CCS_CHECKPOINTS` | `checkpoints/` | split-1 checkpoints of the efficiency scripts: `pbe_reference_k1/<dataset>/checkpoint.pth`, `baselines_k1/<model>/<dataset>/k1/checkpoint.pth` |
| `CCS_PYTHON` | `python` | interpreter used by the shell scripts |
| `JETSON_USER` | `jetson` | login name checked by the Jetson measurement script |

The ten PBE-Net runs of protocol v1 that were completed later are read from `${CCS_WORK}/pbe_v1_fill/runs/pbe/<dataset>/k<k>/`.

## Reproduction

Environment: Python 3 with PyTorch, torchvision, NumPy, SciPy, Pillow, OpenCV, thop and ptflops 0.7.5 (not vendored here). The generators need only NumPy, SciPy and Matplotlib.

```bash
# 0) datasets: download from the sources below and arrange them by splits/<dataset>.tsv
#    (file names per dataset, split k = 1..5, subset train/val/test) under ${CCS_DATA}
# 1) protocol v1 training (Table 1), one row x dataset x split per call
python experiments/qfix/A/code/main.py --suite main                       # print the plan
python experiments/qfix/A/code/main.py --suite main --stage train --execute \
    --data-root "$CCS_DATA" --cache-root "$CCS_ARCHIVE/cache" --run-root "$CCS_ARCHIVE/runs"
# 2) deployed-graph compression and label-free screen (F3), then the single decision run
bash experiments/qdeploy/run_qdeploy.sh && bash experiments/qdeploy/run_decide.sh
bash experiments/qdeploy_w8a8/run_w8a8.sh && bash experiments/qdeploy_w8a8/run_decide_w8a8.sh
python experiments/qdeploy_tau/tau_youden.py; python experiments/qdeploy_tau/q_robust.py; python experiments/qdeploy_tau/agree_cell_auc.py
# 3) weight-only prediction, repair test, ablation
bash experiments/theoryq/run_theoryq.sh synth; bash experiments/theoryq/run_theoryq.sh run; bash experiments/theoryq/run_theoryq.sh decide
bash experiments/sgr/run_sgr.sh gpu0 && bash experiments/sgr/run_decide.sh
bash experiments/abl/tools/run_abl.sh 1 && bash experiments/abl/tools/run_decide.sh
# 4) premise M on FIVES
bash experiments/s4mem/scripts/f5_queue.sh 0
# 5) tables and Fig. 1 from the result files (no GPU)
cd manuscript_tools && for g in tools/gen_*.py tools/sig_table1.py tools/draw_fig1_collapse_map.py; do python -s "$g"; done
```

The run scripts check `FROZEN_SHA256SUMS` before they start (see Redaction).

## Checkpoints

The 329 checkpoints analysed in setting D (330 trained, one excluded by the pre-registered rule: MALUNet, ISIC 2018, split 4) take 7,005.4 MB (6.52 GiB) in total, with the largest single file at 105.3 MB (MAResUNet). This exceeds the 100 MB per-file limit of a git repository, so the checkpoints are **available on request** from the corresponding author. `experiments/qdeploy/MANIFEST_330.sha256` lists the sha256 of every checkpoint and result file, so a copy received on request can be verified.

## Datasets

| Dataset | Source |
|---|---|
| PH2 | T. Mendonça, P. M. Ferreira, J. S. Marques, A. R. S. Marçal, J. Rozeira, "PH² – A dermoscopic image database for research and benchmarking," Proc. 35th Annu. Int. Conf. IEEE EMBS, pp. 5437–5440, 2013, doi:10.1109/EMBC.2013.6610779 |
| ISIC 2017 | N. C. F. Codella et al., "Skin lesion analysis toward melanoma detection: A challenge at the 2017 International Symposium on Biomedical Imaging (ISBI), hosted by the International Skin Imaging Collaboration (ISIC)," arXiv:1710.05006, 2017 |
| ISIC 2018 | N. Codella et al., "Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the International Skin Imaging Collaboration (ISIC)," arXiv:1902.03368, 2019; P. Tschandl, C. Rosendahl, H. Kittler, "The HAM10000 dataset," Scientific Data 5, 180161, 2018, doi:10.1038/sdata.2018.161 |
| Kvasir-SEG | D. Jha et al., "Kvasir-SEG: A segmented polyp dataset," MultiMedia Modeling, LNCS, pp. 451–462, 2020, doi:10.1007/978-3-030-37734-2_37 |
| CVC-ClinicDB | J. Bernal et al., "WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians," Computerized Medical Imaging and Graphics 43, pp. 99–111, 2015, doi:10.1016/j.compmedimag.2015.02.007 |
| ETIS-Larib | J. Silva, A. Histace, O. Romain, X. Dray, B. Granado, "Toward embedded detection of polyps in WCE images for early diagnosis of colorectal cancer," IJCARS 9(2), pp. 283–293, 2014, doi:10.1007/s11548-013-0926-3 |
| FIVES | K. Jin et al., "FIVES: A fundus image dataset for artificial intelligence based vessel segmentation," Scientific Data 9, 475, 2022, doi:10.1038/s41597-022-01564-3 (official `train/` folder only; `splits/FIVES_train_val_split.json`, seed 20261014) |

The datasets are subject to their own licenses and terms of use. Obtain them from their providers.

## Pre-registration documents- `prereg/SHA256SUMS_original.txt` lists the sha256 of the 19 documents exactly as frozen before each measurement. These hashes are the ones printed in the manuscript (Table tab:app_prereg) and recorded in the frozen run folders. The verbatim originals are not published here because they contain internal host names and network addresses. They are available from the corresponding author on request, and their hashes can be checked against this file.n run folders.

**Note on withheld originals.** The first line of `experiments/{qdeploy,qdeploy_w8a8,sgr,theoryq}/FROZEN_SHA256SUMS` and `manuscript_tools/tools/gen_prereg_table.py` refer to `prereg/original/`, which is not part of this release. To run those checks, place the originals (available on request) in `prereg/original/`, or remove that line; every other line refers to a released file.

- `prereg/redacted/` holds copies of the same 19 documents in which absolute server paths, server and host names, the account name, local workspace paths and internal project-management terms (decision identifiers, session and tool names) are replaced by neutral placeholders such as `<WORK>`, `<ARCHIVE>`, `<DATA>`, `gpu-server`, `[internal decision]`, `study lead`. No sentence, number, rule or prediction was changed.
- **Why the redacted copies do not match the hashes:** a sha256 changes when any byte changes, so every replaced path or name gives a different hash. The redacted copies are a reading aid only, and their own hashes are in `prereg/redacted/SHA256SUMS_redacted.txt`. The evidence that a document was frozen before its measurement is the hash of the original copy.

## Redaction of the code copy

The code was copied from the measurement folders and then edited only to remove absolute paths, host names and internal terms from comments and strings, and to read paths from the variables above. `CODE_REDACTION.tsv` lists, for every file, the sha256 of the measured original and of the released copy (`verbatim` when they are equal). Because of these edits:

- `FROZEN_SHA256SUMS.original` in `experiments/{qdeploy,qdeploy_w8a8,sgr,theoryq}` keeps the hashes frozen before each run (hashes of the originals), and `FROZEN_SHA256SUMS` lists the hashes of the released files that the run scripts check. Ten baseline source files of the theory test are not listed there (see Third-party code).
- Generators that check the sha256 of an input whose released copy was redacted carry the released hash; the original hash is in `CODE_REDACTION.tsv`.
- `results/qdeploy/RECORDS_P32_server_sha.original.txt` keeps the hashes of the 330 original records, and `RECORDS_P32_server_sha.txt` those of the released, path-redacted records.

## Third-party code

The ten baseline networks (MAResUNet, UTANet, MK-UNet, MHorUNet, TinyU-Net, UCMNet, LB-UNet, CMUNeXt, EGE-UNet, MALUNet) were used as byte-identical copies of their public implementations. Their source files are not redistributed here. `THIRD_PARTY.md` lists the expected path, size and sha256 of each file, which the adapters in `experiments/qfix/A/code/models/<model>/` load. ptflops 0.7.5 and thop are installed from PyPI.

## License

Code in this repository: MIT License (`LICENSE`). Split lists, result files and documents are released for the reproduction of the manuscript. Datasets and third-party code keep their own licenses.

## Citation

Please cite the manuscript "Label-Free Screening of Collapse Across 329 Compressed Medical Segmentation Checkpoints: A PBE-Net Case Study" (under review; bibliographic details will be added on publication).
