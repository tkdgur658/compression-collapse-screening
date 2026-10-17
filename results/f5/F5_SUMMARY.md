# F5 summary (PBE-Net under the FIVES T protocol)

Rules: PREREG_F5.md (frozen) sec. 3-4, 7. Generated mechanically by f5_summary.py.
Bootstrap: paired percentile, B = 2000, numpy.random.default_rng(20261016), index matrices all (100 images) then I3 (100 images with thin3_nskel > 0), shared by every model and variant. Threshold 0.8.

## sha256 check (sec. 7)

| file | path | expected (prefix) | computed | status |
|---|---|---|---|---|
| scripts/f5_train.py | <RESULTS>\f5\sha_copies/f5_train.py | 910fb8608f9c1c55 | 910fb8608f9c1c55e0eccc0c5c729f5311b6411f4c9e6bd326dfc21a707358f5 | OK |
| scripts/g1.py | <RESULTS>\f5\sha_copies/g1.py | 1625914b9c55d2db | 1625914b9c55d2dbe3cbeb6afd6f3f920c698029d64e1604736dab6f41ffcc59 | OK |
| scripts/g2a_summary.py | <RESULTS>\f5\sha_copies/g2a_summary.py | 8b91ae05a77f4de9 | 8b91ae05a77f4de9a1b729761cba70cc0908baaa867d9c3d15f49238dbe60afa | OK |
| code_v2/models/pbe/src/net.py | <RESULTS>\f5\sha_copies/net.py | de4dcb9c59d4a49c | de4dcb9c59d4a49c96553bced07621635ed0bde57e23bb6b2bad39d754d33562 | OK |
| code_v2/models/pbe/src/matching.py | <RESULTS>\f5\sha_copies/matching.py | 3af81e50e47e4499 | 3af81e50e47e4499d19447ccbb7e9419302bfa486d21d85d4127f8b38c654ff3 | OK |
| code_v2/models/pbe/adapter.py | <RESULTS>\f5\sha_copies/adapter.py | 23e061bbb1a88cf6 | 23e061bbb1a88cf642ed7ed1dffd1149829499361d9aaeaab4a4d29b46a842e6 | OK |
| code_v2/common/losses.py | <RESULTS>\f5\sha_copies/losses.py | 365a663073c303f8 | 365a663073c303f83688378d854f60c261e7d33261a1c300d2219442ac63fd13 | OK |
| g2a/val_list.txt | <RESULTS>\f5\sha_copies/val_list.txt | 93ddae6ab100eb37 | 93ddae6ab100eb371440e1704c384713d9223e026ce30cbc3d1c4d505dd1caea | OK |
| g2a/split.json | <RESULTS>\f5\sha_copies/split.json | 840d22ff92ff5339 | 840d22ff92ff533927277b7dd7276df16bd02929fc3eb35cc098bbaf25271c3b | OK |
| scripts/f5_boot.py | <RESULTS>\f5\scripts/f5_boot.py | d783b5b8b90d4a44f30c24279a0c9934535e0c3e5baf28fef6024a5f8f81d0d1 | d783b5b8b90d4a44f30c24279a0c9934535e0c3e5baf28fef6024a5f8f81d0d1 | OK |
| scripts/f5_summary.py | <RESULTS>\f5\scripts/f5_summary.py | b9c6a16eb44f82418d45a0235dfd4ecfb642a465f4a12a90dbb37599901ae255 | b9c6a16eb44f82418d45a0235dfd4ecfb642a465f4a12a90dbb37599901ae255 | OK |
| scripts/f5_queue.sh | <RESULTS>\f5\sha_copies/f5_queue.sh | 851b62c66e32d6b6469d314e0ee19c93be51292b22bf940b67e424c3f3732902 | 851b62c66e32d6b6469d314e0ee19c93be51292b22bf940b67e424c3f3732902 | OK |
| PREREG_F5.md | <RESULTS>\f5\sha_copies/PREREG_F5.md | 42dc239842e778bfd2329fd88584cbc176356639f833eb9664873382dcfd4c80 | 42dc239842e778bfd2329fd88584cbc176356639f833eb9664873382dcfd4c80 | OK |

Validity: valid (all sec. 7 sha256 match)

Jobs complete: YES  (markers: {'native': 'DONE', 'rs256': 'DONE', 'rs512': 'DONE', 'rs1024': 'DONE'}; ALL_DONE: ALL_DONE 2026-10-16 10:33:11 done=4 fail=0)
Dropped variants (S2/O5): none
Missing results.json: none

Limitations (sec. 4.4): 1 dataset (FIVES), 1 seed, 100 train-internal validation images, the same validation images used for checkpoint selection (optimism, identical for every variant); the official FIVES test set was not used. Patch-512 rows use weights trained on 256 crops (G2 B3 limitation).

## 4.1 PBE-Net (primary)

### PBE-Net

Absolute values (medians over images, paired percentile 95% CI) precede ratios. thin3 medians over I3; other medians over all images.

Full-resolution reference and tiled fp32 (T candidates):

| variant | peak MiB (alloc, batch 1) | exec 2 GiB | exec 0.5 GiB | thin3 clDice [CI] | clDice [CI] | Dice [CI] | thin5 clDice [CI] | ratio r [CI] | T class (CI rule) | point-only (G2 B6) | seam grid | band [CI] | interior [CI] | seam relative s [CI] | seam break (<0.8) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full_fp32 | 8391 | no | no | 0.454 [0.422, 0.468] | 0.932 [0.923, 0.938] | 0.910 [0.902, 0.914] | 0.568 [0.533, 0.602] | 1.000 [1.000, 1.000] | — (not a T target) | — | — | — | — | — | — |  |
| full_amp | 4231 | no | no | 0.455 [0.419, 0.468] | 0.932 [0.923, 0.938] | 0.910 [0.902, 0.914] | 0.569 [0.531, 0.602] | 1.002 [0.989, 1.006] | — (not a T target) | — | — | — | — | — | — |  |
| tile_p256_o0_fp32 | 210 | yes | yes | 0.490 [0.464, 0.508] | 0.914 [0.901, 0.925] | 0.913 [0.903, 0.919] | 0.538 [0.505, 0.571] | 1.080 [1.036, 1.151] | preserved | preserved (point) | 256 | 0.844 [0.834, 0.854] | 0.926 [0.916, 0.936] | 0.923 [0.912, 0.931] | no |  |
| tile_p256_o50_fp32 | 210 | yes | yes | 0.441 [0.418, 0.470] | 0.934 [0.928, 0.942] | 0.928 [0.922, 0.933] | 0.558 [0.531, 0.605] | 0.972 [0.935, 1.033] | preserved | preserved (point) | 128 | 0.925 [0.918, 0.929] | 0.939 [0.932, 0.945] | 1.002 [0.996, 1.007] | no |  |
| tile_p512_o0_fp32 | 581 | yes | no | 0.453 [0.432, 0.476] | 0.926 [0.913, 0.934] | 0.915 [0.904, 0.918] | 0.548 [0.512, 0.576] | 0.998 [0.964, 1.063] | — (not a T target) | — | 512 | 0.848 [0.838, 0.859] | 0.932 [0.921, 0.939] | 0.923 [0.913, 0.932] | no | weights trained on 256 crops |
| tile_p512_o50_fp32 | 581 | yes | no | 0.429 [0.404, 0.453] | 0.938 [0.926, 0.943] | 0.926 [0.917, 0.930] | 0.557 [0.524, 0.595] | 0.945 [0.919, 1.003] | — (not a T target) | — | 256 | 0.927 [0.920, 0.933] | 0.940 [0.929, 0.945] | 0.998 [0.993, 1.007] | no | weights trained on 256 crops |

AMP rows (auxiliary, descriptive; excluded from T, O3):

| variant | peak MiB (alloc, batch 1) | exec 2 GiB | exec 0.5 GiB | thin3 clDice [CI] | clDice [CI] | Dice [CI] | thin5 clDice [CI] | ratio r [CI] | T class (CI rule) | point-only (G2 B6) | seam grid | band [CI] | interior [CI] | seam relative s [CI] | seam break (<0.8) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tile_p256_o0_amp | 199 | yes | yes | 0.488 [0.463, 0.509] | 0.913 [0.901, 0.925] | 0.913 [0.902, 0.919] | 0.537 [0.507, 0.570] | 1.075 [1.033, 1.151] | — (not a T target) | — | 256 | 0.845 [0.833, 0.853] | 0.926 [0.916, 0.936] | 0.924 [0.913, 0.932] | no |  |
| tile_p256_o50_amp | 199 | yes | yes | 0.444 [0.420, 0.472] | 0.934 [0.927, 0.942] | 0.928 [0.922, 0.933] | 0.558 [0.530, 0.603] | 0.978 [0.937, 1.037] | — (not a T target) | — | 128 | 0.925 [0.918, 0.928] | 0.939 [0.932, 0.945] | 1.002 [0.996, 1.007] | no |  |
| tile_p512_o0_amp | 320 | yes | yes | 0.451 [0.431, 0.477] | 0.927 [0.913, 0.934] | 0.915 [0.904, 0.918] | 0.548 [0.511, 0.576] | 0.995 [0.966, 1.058] | — (not a T target) | — | 512 | 0.846 [0.837, 0.858] | 0.932 [0.920, 0.939] | 0.922 [0.912, 0.932] | no | weights trained on 256 crops |
| tile_p512_o50_amp | 320 | yes | yes | 0.428 [0.405, 0.453] | 0.938 [0.926, 0.943] | 0.926 [0.917, 0.930] | 0.560 [0.527, 0.594] | 0.944 [0.921, 1.005] | — (not a T target) | — | 256 | 0.927 [0.920, 0.933] | 0.941 [0.929, 0.945] | 0.997 [0.993, 1.007] | no | weights trained on 256 crops |

Block "resize, not tiled" (not a T target, O8):

| variant | peak MiB (alloc, batch 1) | exec 2 GiB | exec 0.5 GiB | thin3 clDice [CI] | clDice [CI] | Dice [CI] | thin5 clDice [CI] | ratio r [CI] | T class (CI rule) | point-only (G2 B6) | seam grid | band [CI] | interior [CI] | seam relative s [CI] | seam break (<0.8) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rs256_fp32 | 177 | yes | yes | 0.304 [0.281, 0.325] | 0.820 [0.806, 0.829] | 0.802 [0.796, 0.806] | 0.201 [0.184, 0.227] | 0.671 [0.614, 0.742] | — (not a T target) | — | — | — | — | — | — |  |
| rs512_fp32 | 545 | yes | no | 0.368 [0.344, 0.400] | 0.916 [0.907, 0.921] | 0.893 [0.884, 0.897] | 0.401 [0.366, 0.421] | 0.810 [0.761, 0.912] | — (not a T target) | — | — | — | — | — | — |  |
| rs1024_fp32 | 2113 | no | no | 0.482 [0.464, 0.493] | 0.936 [0.927, 0.941] | 0.915 [0.905, 0.920] | 0.576 [0.560, 0.610] | 1.062 [1.021, 1.125] | — (not a T target) | — | — | — | — | — | — |  |

E (fp32 tiled, peak ≤ 512 MiB) = ['tile_p256_o0_fp32', 'tile_p256_o50_fp32']

**T statement (PBE-Net):** Tile inference of PBE-Net within 0.5 GiB keeps thin-vessel clDice at ≥ 0.8 of full-resolution inference (all executable tiled variants, CI lower bounds ≥ 0.8).

Overlap attribution (sec. 4.3; fp32; T statements do not use overlap):

- PBE-Net, patch 256: overlap raises seam continuity (Δseam = +0.080 [+0.070, +0.091]).
- PBE-Net, patch 256: whole-image ratio is higher at 0% than at 50% overlap (Δratio = -0.108 [-0.159, -0.052]), not attributed to overlap.
- PBE-Net, patch 512: overlap raises seam continuity (Δseam = +0.075 [+0.067, +0.088]).
- PBE-Net, patch 512: whole-image ratio is higher at 0% than at 50% overlap (Δratio = -0.054 [-0.097, -0.015]), not attributed to overlap.

## 4.2 Same rule, recomputed for the three G2 models (no retraining, no re-inference)

Manuscript match (Table tab:tiling_fives, 21 rows x peak, thin3 clDice, Dice, ratio, seam, budgets; source 4f_analysis_tiling.tex): **PASS**

### UCMNet

Absolute values (medians over images, paired percentile 95% CI) precede ratios. thin3 medians over I3; other medians over all images.

Full-resolution reference and tiled fp32 (T candidates):

| variant | peak MiB (alloc, batch 1) | exec 2 GiB | exec 0.5 GiB | thin3 clDice [CI] | clDice [CI] | Dice [CI] | thin5 clDice [CI] | ratio r [CI] | T class (CI rule) | point-only (G2 B6) | seam grid | band [CI] | interior [CI] | seam relative s [CI] | seam break (<0.8) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full_fp32 | 907 | yes | no | 0.526 [0.507, 0.542] | 0.830 [0.795, 0.854] | 0.829 [0.804, 0.845] | 0.599 [0.584, 0.628] | 1.000 [1.000, 1.000] | — (not a T target) | — | — | — | — | — | — |  |
| full_amp | 1731 | yes | no | 0.523 [0.508, 0.540] | 0.830 [0.795, 0.854] | 0.829 [0.805, 0.845] | 0.599 [0.585, 0.626] | 0.995 [0.988, 1.010] | — (not a T target) | — | — | — | — | — | — |  |
| tile_p256_o0_fp32 | 188 | yes | yes | 0.415 [0.401, 0.445] | 0.880 [0.860, 0.889] | 0.871 [0.858, 0.880] | 0.445 [0.434, 0.472] | 0.790 [0.755, 0.848] | indeterminate | not preserved (point) | 256 | 0.782 [0.770, 0.796] | 0.895 [0.869, 0.905] | 0.886 [0.870, 0.917] | no |  |
| tile_p256_o50_fp32 | 188 | yes | yes | 0.379 [0.365, 0.413] | 0.907 [0.889, 0.917] | 0.897 [0.883, 0.904] | 0.461 [0.440, 0.479] | 0.721 [0.691, 0.778] | not preserved | not preserved (point) | 128 | 0.894 [0.885, 0.900] | 0.910 [0.895, 0.922] | 0.988 [0.972, 1.010] | no |  |
| tile_p512_o0_fp32 | 221 | yes | yes | 0.497 [0.478, 0.515] | 0.881 [0.865, 0.890] | 0.870 [0.860, 0.879] | 0.548 [0.535, 0.575] | 0.946 [0.904, 0.983] | preserved | preserved (point) | 512 | 0.766 [0.753, 0.786] | 0.886 [0.873, 0.899] | 0.897 [0.873, 0.918] | no | weights trained on 256 crops |
| tile_p512_o50_fp32 | 221 | yes | yes | 0.513 [0.479, 0.530] | 0.895 [0.883, 0.912] | 0.890 [0.878, 0.898] | 0.576 [0.559, 0.601] | 0.976 [0.912, 1.014] | preserved | preserved (point) | 256 | 0.889 [0.876, 0.903] | 0.898 [0.881, 0.912] | 1.005 [0.985, 1.033] | no | weights trained on 256 crops |

AMP rows (auxiliary, descriptive; excluded from T, O3):

| variant | peak MiB (alloc, batch 1) | exec 2 GiB | exec 0.5 GiB | thin3 clDice [CI] | clDice [CI] | Dice [CI] | thin5 clDice [CI] | ratio r [CI] | T class (CI rule) | point-only (G2 B6) | seam grid | band [CI] | interior [CI] | seam relative s [CI] | seam break (<0.8) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tile_p256_o0_amp | 180 | yes | yes | 0.418 [0.397, 0.446] | 0.880 [0.860, 0.889] | 0.870 [0.858, 0.880] | 0.444 [0.434, 0.469] | 0.795 [0.753, 0.852] | — (not a T target) | — | 256 | 0.782 [0.771, 0.796] | 0.895 [0.869, 0.905] | 0.886 [0.870, 0.916] | no |  |
| tile_p256_o50_amp | 180 | yes | yes | 0.379 [0.364, 0.416] | 0.907 [0.889, 0.917] | 0.897 [0.883, 0.904] | 0.462 [0.441, 0.479] | 0.720 [0.689, 0.781] | — (not a T target) | — | 128 | 0.894 [0.885, 0.899] | 0.910 [0.895, 0.922] | 0.987 [0.972, 1.010] | no |  |
| tile_p512_o0_amp | 189 | yes | yes | 0.496 [0.479, 0.510] | 0.880 [0.865, 0.891] | 0.870 [0.860, 0.879] | 0.549 [0.536, 0.575] | 0.944 [0.904, 0.983] | — (not a T target) | — | 512 | 0.768 [0.754, 0.783] | 0.886 [0.872, 0.899] | 0.898 [0.872, 0.919] | no | weights trained on 256 crops |
| tile_p512_o50_amp | 189 | yes | yes | 0.516 [0.482, 0.528] | 0.895 [0.883, 0.912] | 0.890 [0.878, 0.898] | 0.575 [0.559, 0.599] | 0.981 [0.917, 1.016] | — (not a T target) | — | 256 | 0.890 [0.876, 0.902] | 0.898 [0.881, 0.912] | 1.005 [0.984, 1.032] | no | weights trained on 256 crops |

Block "resize, not tiled" (not a T target, O8):

| variant | peak MiB (alloc, batch 1) | exec 2 GiB | exec 0.5 GiB | thin3 clDice [CI] | clDice [CI] | Dice [CI] | thin5 clDice [CI] | ratio r [CI] | T class (CI rule) | point-only (G2 B6) | seam grid | band [CI] | interior [CI] | seam relative s [CI] | seam break (<0.8) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rs256_fp32 | 155 | yes | yes | 0.259 [0.228, 0.292] | 0.784 [0.771, 0.795] | 0.782 [0.773, 0.789] | 0.167 [0.152, 0.199] | 0.492 [0.436, 0.557] | — (not a T target) | — | — | — | — | — | — |  |
| rs512_fp32 | 186 | yes | yes | 0.385 [0.359, 0.417] | 0.860 [0.842, 0.869] | 0.830 [0.823, 0.841] | 0.358 [0.321, 0.394] | 0.731 [0.680, 0.797] | — (not a T target) | — | — | — | — | — | — |  |
| rs1024_fp32 | 308 | yes | yes | 0.468 [0.451, 0.483] | 0.861 [0.844, 0.875] | 0.846 [0.827, 0.855] | 0.504 [0.477, 0.521] | 0.890 [0.848, 0.928] | — (not a T target) | — | — | — | — | — | — |  |

E (fp32 tiled, peak ≤ 512 MiB) = ['tile_p256_o0_fp32', 'tile_p256_o50_fp32', 'tile_p512_o0_fp32', 'tile_p512_o50_fp32']

**T statement (UCMNet):** Tile inference of UCMNet within 0.5 GiB preserves at tile_p512_o0_fp32, tile_p512_o50_fp32, except tile_p256_o0_fp32 (indeterminate, r = 0.790 [0.755, 0.848]), tile_p256_o50_fp32 (not preserved, r = 0.721 [0.691, 0.778]).

Overlap attribution (sec. 4.3; fp32; T statements do not use overlap):

- UCMNet, patch 256: overlap raises seam continuity (Δseam = +0.102 [+0.078, +0.119]).
- UCMNet, patch 256: whole-image ratio is higher at 0% than at 50% overlap (Δratio = -0.069 [-0.121, -0.022]), not attributed to overlap.
- UCMNet, patch 512: overlap raises seam continuity (Δseam = +0.108 [+0.090, +0.138]).
- UCMNet, patch 512: whole-image ratio does not differ detectably between 0% and 50% overlap (Δratio = +0.031 [-0.023, +0.068]).

### UTANet

Absolute values (medians over images, paired percentile 95% CI) precede ratios. thin3 medians over I3; other medians over all images.

Full-resolution reference and tiled fp32 (T candidates):

| variant | peak MiB (alloc, batch 1) | exec 2 GiB | exec 0.5 GiB | thin3 clDice [CI] | clDice [CI] | Dice [CI] | thin5 clDice [CI] | ratio r [CI] | T class (CI rule) | point-only (G2 B6) | seam grid | band [CI] | interior [CI] | seam relative s [CI] | seam break (<0.8) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full_fp32 | 5422 | no | no | 0.463 [0.441, 0.487] | 0.938 [0.927, 0.945] | 0.929 [0.924, 0.938] | 0.585 [0.558, 0.620] | 1.000 [1.000, 1.000] | — (not a T target) | — | — | — | — | — | — |  |
| full_amp | 3598 | no | no | 0.460 [0.446, 0.486] | 0.938 [0.927, 0.944] | 0.929 [0.924, 0.938] | 0.584 [0.560, 0.621] | 0.994 [0.990, 1.008] | — (not a T target) | — | — | — | — | — | — |  |
| tile_p256_o0_fp32 | 302 | yes | yes | 0.514 [0.493, 0.527] | 0.922 [0.905, 0.931] | 0.917 [0.905, 0.925] | 0.566 [0.523, 0.609] | 1.110 [1.053, 1.153] | preserved | preserved (point) | 256 | 0.860 [0.848, 0.874] | 0.930 [0.917, 0.939] | 0.936 [0.925, 0.949] | no |  |
| tile_p256_o50_fp32 | 302 | yes | yes | 0.456 [0.424, 0.492] | 0.937 [0.926, 0.943] | 0.929 [0.923, 0.937] | 0.580 [0.552, 0.621] | 0.985 [0.947, 1.027] | preserved | preserved (point) | 128 | 0.927 [0.916, 0.933] | 0.939 [0.934, 0.945] | 1.003 [0.997, 1.005] | no |  |
| tile_p512_o0_fp32 | 461 | yes | yes | 0.493 [0.474, 0.511] | 0.930 [0.918, 0.938] | 0.924 [0.916, 0.934] | 0.586 [0.549, 0.610] | 1.065 [1.030, 1.102] | preserved | preserved (point) | 512 | 0.868 [0.851, 0.881] | 0.934 [0.925, 0.942] | 0.942 [0.924, 0.955] | no | weights trained on 256 crops |
| tile_p512_o50_fp32 | 461 | yes | yes | 0.461 [0.438, 0.487] | 0.938 [0.927, 0.944] | 0.929 [0.924, 0.938] | 0.585 [0.557, 0.619] | 0.995 [0.980, 1.005] | preserved | preserved (point) | 256 | 0.929 [0.919, 0.935] | 0.940 [0.931, 0.945] | 1.000 [0.999, 1.000] | no | weights trained on 256 crops |

AMP rows (auxiliary, descriptive; excluded from T, O3):

| variant | peak MiB (alloc, batch 1) | exec 2 GiB | exec 0.5 GiB | thin3 clDice [CI] | clDice [CI] | Dice [CI] | thin5 clDice [CI] | ratio r [CI] | T class (CI rule) | point-only (G2 B6) | seam grid | band [CI] | interior [CI] | seam relative s [CI] | seam break (<0.8) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tile_p256_o0_amp | 326 | yes | yes | 0.515 [0.491, 0.526] | 0.922 [0.905, 0.931] | 0.918 [0.905, 0.925] | 0.566 [0.523, 0.610] | 1.111 [1.050, 1.151] | — (not a T target) | — | 256 | 0.860 [0.848, 0.874] | 0.930 [0.917, 0.939] | 0.936 [0.925, 0.949] | no |  |
| tile_p256_o50_amp | 326 | yes | yes | 0.455 [0.428, 0.491] | 0.937 [0.926, 0.943] | 0.929 [0.923, 0.937] | 0.581 [0.552, 0.620] | 0.982 [0.948, 1.026] | — (not a T target) | — | 128 | 0.927 [0.916, 0.933] | 0.939 [0.934, 0.945] | 1.003 [0.997, 1.005] | no |  |
| tile_p512_o0_amp | 406 | yes | yes | 0.496 [0.474, 0.511] | 0.930 [0.918, 0.938] | 0.924 [0.916, 0.934] | 0.586 [0.549, 0.611] | 1.071 [1.032, 1.104] | — (not a T target) | — | 512 | 0.868 [0.851, 0.881] | 0.934 [0.924, 0.942] | 0.942 [0.924, 0.955] | no | weights trained on 256 crops |
| tile_p512_o50_amp | 406 | yes | yes | 0.458 [0.441, 0.487] | 0.938 [0.927, 0.944] | 0.929 [0.924, 0.938] | 0.584 [0.559, 0.620] | 0.989 [0.977, 1.010] | — (not a T target) | — | 256 | 0.930 [0.919, 0.935] | 0.940 [0.931, 0.945] | 1.000 [0.999, 1.001] | no | weights trained on 256 crops |

Block "resize, not tiled" (not a T target, O8):

| variant | peak MiB (alloc, batch 1) | exec 2 GiB | exec 0.5 GiB | thin3 clDice [CI] | clDice [CI] | Dice [CI] | thin5 clDice [CI] | ratio r [CI] | T class (CI rule) | point-only (G2 B6) | seam grid | band [CI] | interior [CI] | seam relative s [CI] | seam break (<0.8) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rs256_fp32 | 268 | yes | yes | 0.365 [0.349, 0.387] | 0.923 [0.915, 0.927] | 0.869 [0.867, 0.874] | 0.411 [0.394, 0.425] | 0.788 [0.736, 0.853] | — (not a T target) | — | — | — | — | — | — |  |
| rs512_fp32 | 427 | yes | yes | 0.460 [0.421, 0.480] | 0.948 [0.941, 0.954] | 0.925 [0.919, 0.928] | 0.599 [0.584, 0.629] | 0.993 [0.923, 1.045] | — (not a T target) | — | — | — | — | — | — |  |
| rs1024_fp32 | 1425 | yes | no | 0.502 [0.482, 0.518] | 0.949 [0.940, 0.955] | 0.939 [0.932, 0.943] | 0.650 [0.629, 0.671] | 1.084 [1.037, 1.125] | — (not a T target) | — | — | — | — | — | — |  |

E (fp32 tiled, peak ≤ 512 MiB) = ['tile_p256_o0_fp32', 'tile_p256_o50_fp32', 'tile_p512_o0_fp32', 'tile_p512_o50_fp32']

**T statement (UTANet):** Tile inference of UTANet within 0.5 GiB keeps thin-vessel clDice at ≥ 0.8 of full-resolution inference (all executable tiled variants, CI lower bounds ≥ 0.8).

Overlap attribution (sec. 4.3; fp32; T statements do not use overlap):

- UTANet, patch 256: overlap raises seam continuity (Δseam = +0.068 [+0.051, +0.078]).
- UTANet, patch 256: whole-image ratio is higher at 0% than at 50% overlap (Δratio = -0.125 [-0.176, -0.058]), not attributed to overlap.
- UTANet, patch 512: overlap raises seam continuity (Δseam = +0.058 [+0.045, +0.076]).
- UTANet, patch 512: whole-image ratio is higher at 0% than at 50% overlap (Δratio = -0.070 [-0.108, -0.031]), not attributed to overlap.

### MAResUNet

Absolute values (medians over images, paired percentile 95% CI) precede ratios. thin3 medians over I3; other medians over all images.

Full-resolution reference and tiled fp32 (T candidates):

| variant | peak MiB (alloc, batch 1) | exec 2 GiB | exec 0.5 GiB | thin3 clDice [CI] | clDice [CI] | Dice [CI] | thin5 clDice [CI] | ratio r [CI] | T class (CI rule) | point-only (G2 B6) | seam grid | band [CI] | interior [CI] | seam relative s [CI] | seam break (<0.8) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full_fp32 | 1999 | yes | no | 0.395 [0.373, 0.407] | 0.932 [0.924, 0.938] | 0.926 [0.917, 0.931] | 0.502 [0.490, 0.524] | 1.000 [1.000, 1.000] | — (not a T target) | — | — | — | — | — | — |  |
| full_amp | 1520 | yes | no | non-finite 100/100 | — | — | — | — | — | — | — | — | — | — | — |  |
| tile_p256_o0_fp32 | 290 | yes | yes | 0.479 [0.457, 0.509] | 0.912 [0.904, 0.925] | 0.908 [0.899, 0.917] | 0.531 [0.514, 0.550] | 1.214 [1.159, 1.297] | preserved | preserved (point) | 256 | 0.844 [0.830, 0.858] | 0.922 [0.913, 0.932] | 0.929 [0.917, 0.940] | no |  |
| tile_p256_o50_fp32 | 290 | yes | yes | 0.437 [0.410, 0.453] | 0.930 [0.924, 0.939] | 0.924 [0.917, 0.933] | 0.563 [0.531, 0.578] | 1.107 [1.060, 1.141] | preserved | preserved (point) | 128 | 0.920 [0.908, 0.928] | 0.935 [0.927, 0.940] | 1.001 [0.997, 1.009] | no |  |
| tile_p512_o0_fp32 | 325 | yes | yes | 0.440 [0.419, 0.458] | 0.926 [0.914, 0.934] | 0.920 [0.911, 0.928] | 0.513 [0.505, 0.542] | 1.114 [1.079, 1.167] | preserved | preserved (point) | 512 | 0.849 [0.828, 0.870] | 0.929 [0.919, 0.938] | 0.926 [0.906, 0.943] | no | weights trained on 256 crops |
| tile_p512_o50_fp32 | 325 | yes | yes | 0.405 [0.386, 0.426] | 0.931 [0.925, 0.940] | 0.928 [0.919, 0.934] | 0.526 [0.503, 0.550] | 1.026 [0.998, 1.064] | preserved | preserved (point) | 256 | 0.919 [0.909, 0.928] | 0.934 [0.927, 0.941] | 0.998 [0.993, 1.004] | no | weights trained on 256 crops |

AMP rows (auxiliary, descriptive; excluded from T, O3):

| variant | peak MiB (alloc, batch 1) | exec 2 GiB | exec 0.5 GiB | thin3 clDice [CI] | clDice [CI] | Dice [CI] | thin5 clDice [CI] | ratio r [CI] | T class (CI rule) | point-only (G2 B6) | seam grid | band [CI] | interior [CI] | seam relative s [CI] | seam break (<0.8) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tile_p256_o0_amp | 335 | yes | yes | non-finite 100/100 | — | — | — | — | — | — | — | — | — | — | — |  |
| tile_p256_o50_amp | 335 | yes | yes | non-finite 100/100 | — | — | — | — | — | — | — | — | — | — | — |  |
| tile_p512_o0_amp | 360 | yes | yes | non-finite 100/100 | — | — | — | — | — | — | — | — | — | — | — | weights trained on 256 crops |
| tile_p512_o50_amp | 360 | yes | yes | non-finite 100/100 | — | — | — | — | — | — | — | — | — | — | — | weights trained on 256 crops |

Block "resize, not tiled" (not a T target, O8):

| variant | peak MiB (alloc, batch 1) | exec 2 GiB | exec 0.5 GiB | thin3 clDice [CI] | clDice [CI] | Dice [CI] | thin5 clDice [CI] | ratio r [CI] | T class (CI rule) | point-only (G2 B6) | seam grid | band [CI] | interior [CI] | seam relative s [CI] | seam break (<0.8) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rs256_fp32 | 257 | yes | yes | 0.285 [0.260, 0.317] | 0.879 [0.868, 0.884] | 0.841 [0.839, 0.849] | 0.269 [0.233, 0.295] | 0.721 [0.660, 0.811] | — (not a T target) | — | — | — | — | — | — |  |
| rs512_fp32 | 288 | yes | yes | 0.403 [0.388, 0.424] | 0.932 [0.930, 0.940] | 0.910 [0.903, 0.914] | 0.507 [0.492, 0.544] | 1.022 [0.976, 1.080] | — (not a T target) | — | — | — | — | — | — |  |
| rs1024_fp32 | 588 | yes | no | 0.480 [0.467, 0.507] | 0.941 [0.935, 0.946] | 0.928 [0.921, 0.935] | 0.612 [0.590, 0.631] | 1.217 [1.182, 1.287] | — (not a T target) | — | — | — | — | — | — |  |

E (fp32 tiled, peak ≤ 512 MiB) = ['tile_p256_o0_fp32', 'tile_p256_o50_fp32', 'tile_p512_o0_fp32', 'tile_p512_o50_fp32']

**T statement (MAResUNet):** Tile inference of MAResUNet within 0.5 GiB keeps thin-vessel clDice at ≥ 0.8 of full-resolution inference (all executable tiled variants, CI lower bounds ≥ 0.8).

Overlap attribution (sec. 4.3; fp32; T statements do not use overlap):

- MAResUNet, patch 256: overlap raises seam continuity (Δseam = +0.073 [+0.061, +0.086]).
- MAResUNet, patch 256: whole-image ratio is higher at 0% than at 50% overlap (Δratio = -0.107 [-0.195, -0.054]), not attributed to overlap.
- MAResUNet, patch 512: overlap raises seam continuity (Δseam = +0.073 [+0.055, +0.092]).
- MAResUNet, patch 512: whole-image ratio is higher at 0% than at 50% overlap (Δratio = -0.089 [-0.134, -0.052]), not attributed to overlap.

Statement under review in the manuscript: "tile inference within 0.5 GiB preserves thin-vessel topology, with UCMNet at 256-pixel patches as the one exception". Replaced or confirmed by the T statements above (sec. 4.2).

