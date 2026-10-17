# F3 QDEPLOY RESULT (decide run once)

## Table (i) L80

| set | all cells | PBE-Net | baselines | checkpoint-level |
|---|---|---|---|---|
| O | 102/198 | 18/18 | 84/180 | 477/987 |
| D | 80/198 | 10/18 | 70/180 | 398/987 |
| S_w4g | 38/66 | 6/6 | 32/60 | 159/329 |
| S_w4c | 9/66 | 1/6 | 8/60 | 46/329 |
| S_a6c | 6/66 | 0/6 | 6/60 | 20/329 |
| S_l50c | 57/66 | 6/6 | 51/60 | 266/329 |

## Table (i) L90

| set | all cells | PBE-Net | baselines | checkpoint-level |
|---|---|---|---|---|
| O | 122/198 | 18/18 | 104/180 | 584/987 |
| D | 107/198 | 12/18 | 95/180 | 490/987 |
| S_w4g | 45/66 | 6/6 | 39/60 | 205/329 |
| S_w4c | 22/66 | 1/6 | 21/60 | 90/329 |
| S_a6c | 6/66 | 0/6 | 6/60 | 26/329 |
| S_l50c | 61/66 | 6/6 | 55/60 | 293/329 |

Footnote: Manuscript "11/18" (PBE-Net): queue10 k1 reference checkpoints (sha differs from the v1 runs), split 1 only, collapse = q IoU <= chance+0.05; not the label used here (5-split mean val-Dice retention < 0.8).

## Branch (ii)

N_PBE(D) = 10/18 -> **survives**: PBE-Net vulnerability remains in the deployment graph; main Q table uses D, O becomes stress setting

## (iii) AUC (5-split mean [min, max]; pooled [95% CI])

- O L80: agree 0.975 [0.956, 0.986] pooled 0.975 [0.968, 0.982]; fg 0.932 [0.917, 0.940] pooled 0.933 [0.917, 0.948]; sqnr 0.894 [0.883, 0.902] pooled 0.894 [0.875, 0.913]
  - rule agree vs fg: agreement higher (diff 0.043, positive in 5/5 splits); agree vs SQNR: agreement higher (diff 0.081, positive in 5/5 splits)
- O L90: agree 0.968 [0.949, 0.981] pooled 0.968 [0.959, 0.976]; fg 0.911 [0.892, 0.930] pooled 0.913 [0.896, 0.930]; sqnr 0.882 [0.869, 0.897] pooled 0.883 [0.861, 0.903]
  - rule agree vs fg: agreement higher (diff 0.057, positive in 5/5 splits); agree vs SQNR: agreement higher (diff 0.086, positive in 5/5 splits)
- D L80: agree 0.973 [0.961, 0.989] pooled 0.973 [0.966, 0.980]; fg 0.929 [0.902, 0.948] pooled 0.929 [0.912, 0.945]; sqnr 0.917 [0.905, 0.930] pooled 0.916 [0.900, 0.932]
  - rule agree vs fg: agreement higher (diff 0.044, positive in 5/5 splits); agree vs SQNR: agreement higher (diff 0.056, positive in 5/5 splits)
- D L90: agree 0.964 [0.953, 0.980] pooled 0.963 [0.954, 0.972]; fg 0.909 [0.899, 0.924] pooled 0.909 [0.891, 0.926]; sqnr 0.902 [0.887, 0.913] pooled 0.901 [0.884, 0.919]
  - rule agree vs fg: agreement higher (diff 0.055, positive in 5/5 splits); agree vs SQNR: agreement higher (diff 0.062, positive in 5/5 splits)

## Exclusions

{
 "malunet__isic18__k4": "old FAILED.txt rule not met"
}
E1 failed (D/S excluded): []

Old FAILED.txt runs: {
 "tinyunet__kvasir__k4": {
  "ckpt_sha_match_local_copy": true,
  "results_sha_match_table1_copy": true,
  "val_iou_reeval": 0.7755086114055297,
  "best_val_iou_stored": 0.7754971648149391,
  "abs_diff": 1.144659059060249e-05,
  "included": true
 },
 "egeunet__kvasir__k2": {
  "ckpt_sha_match_local_copy": true,
  "results_sha_match_table1_copy": true,
  "val_iou_reeval": 0.6105411065215198,
  "best_val_iou_stored": 0.6105406912993864,
  "abs_diff": 4.152221334541295e-07,
  "included": true
 },
 "egeunet__kvasir__k3": {
  "ckpt_sha_match_local_copy": true,
  "results_sha_match_table1_copy": true,
  "val_iou_reeval": 0.5956708912980606,
  "best_val_iou_stored": 0.5956724856603564,
  "abs_diff": 1.594362295742613e-06,
  "included": true
 },
 "utanet__isic17__k3": {
  "ckpt_sha_match_local_copy": true,
  "results_sha_match_table1_copy": true,
  "val_iou_reeval": 0.820971279316487,
  "best_val_iou_stored": 0.8209767668856258,
  "abs_diff": 5.487569138873383e-06,
  "included": true
 },
 "mhorunet__isic18__k3": {
  "ckpt_sha_match_local_copy": true,
  "results_sha_match_table1_copy": true,
  "val_iou_reeval": 0.8190439886692924,
  "best_val_iou_stored": 0.8190431956429054,
  "abs_diff": 7.930263870026266e-07,
  "included": true
 },
 "malunet__isic18__k4": {
  "ckpt_sha_match_local_copy": true,
  "results_sha_match_table1_copy": true,
  "val_iou_reeval": 0.8049240132502973,
  "best_val_iou_stored": 0.8220728361468267,
  "abs_diff": 0.017148822896529414,
  "included": false
 }
}
