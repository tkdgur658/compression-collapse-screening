# RESULT_THEORYQ (decide run once)

labels: {'S_w4g': '38/66', 'D_w4': '9/66', 'D_l50': '55/66', 'O_w4': '25/66'}

- H_T1 dep.Z vs S_w4g: AUC 0.705 [0.581, 0.817], threshold 0.85 -> FAIL; p(AUC<=thr) 0.9955, Holm adj 0.9985 (keep); p(AUC<=0.5) 0.0005; pos/neg 38/28
- H_T3 l50.Mstat vs D_l50: AUC 0.744 [0.576, 0.887], threshold 0.75 -> FAIL; p(AUC<=thr) 0.4993, Holm adj 0.9985 (keep); p(AUC<=0.5) 0.0040; pos/neg 55/11
- H_T2 Zpc = 0: PASS (0/329 ckpt with Zpc > 0, nonfinite 0)

## aux
```
{
 "Zprime_S_w4g": {
  "auc": 0.7053571428571429,
  "ci95": [
   0.581081081081081,
   0.8168269230769231
  ],
  "p_le_0.5": 0.0004997501249375312
 },
 "Zw_S_w4g": {
  "auc": 0.6879699248120301,
  "ci95": [
   0.5634191176470589,
   0.8021616541353384
  ],
  "p_le_0.5": 0.0009995002498750624
 },
 "Zglob_S_w4g": {
  "auc": 0.6795112781954887,
  "ci95": [
   0.5482536764705882,
   0.7950367647058824
  ],
  "p_le_0.5": 0.0014992503748125937
 },
 "Zelem_S_w4g": {
  "auc": 0.6541353383458647,
  "ci95": [
   0.516309412861137,
   0.782851817334576
  ],
  "p_le_0.5": 0.013993003498250875
 },
 "Z_all_targets_S_w4g": {
  "auc": 0.7053571428571429,
  "ci95": [
   0.581081081081081,
   0.8168269230769231
  ],
  "p_le_0.5": 0.0004997501249375312
 },
 "logparams_S_w4g": {
  "auc": 0.6353383458646616,
  "ci95": [
   0.4930555555555556,
   0.7735085945399394
  ],
  "p_le_0.5": 0.030984507746126936
 },
 "ZO_O_w4": {
  "auc": 0.573170731707317,
  "ci95": [
   0.4419512195121951,
   0.6960975609756097
  ],
  "p_le_0.5": 0.13943028485757122
 },
 "E_D_l50": {
  "auc": 0.2727272727272727,
  "ci95": [
   0.13598901098901098,
   0.4281567489114659
  ],
  "p_le_0.5": 0.9980009995002499
 },
 "Msum_D_l50": {
  "auc": 0.30578512396694213,
  "ci95": [
   0.16198347107438016,
   0.46942148760330576
  ],
  "p_le_0.5": 0.9900049975012494
 },
 "logparams_D_l50": {
  "auc": 0.22231404958677686,
  "ci95": [
   0.08181818181818182,
   0.38693957115009747
  ],
  "p_le_0.5": 0.9975012493753124
 },
 "Z_D_w4": {
  "auc": 0.4249512670565302,
  "ci95": [
   0.24193548387096775,
   0.6267056530214425
  ],
  "p_le_0.5": 0.7946026986506747
 },
 "Z_max_agg_S_w4g": {
  "auc": 0.7091165413533834,
  "ci95": [
   0.5862068965517241,
   0.8206730769230769
  ]
 },
 "Mstat_max_agg_D_l50": {
  "auc": 0.684297520661157,
  "ci95": [
   0.49878934624697335,
   0.8519593613933236
  ]
 },
 "ckpt_Z_S_w4g": {
  "point": 0.6862190159082501,
  "ci95": [
   0.5965614782704529,
   0.7648538660747318
  ],
  "n_cells": 66,
  "n_pos": 159,
  "n_neg": 170
 },
 "ckpt_Mstat_D_l50": {
  "point": 0.6183356661152818,
  "ci95": [
   0.46484375,
   0.7521550442898758
  ],
  "n_cells": 66,
  "n_pos": 271,
  "n_neg": 58
 },
 "ckpt_ZO_O_w4": {
  "point": 0.5897008547008546,
  "ci95": [
   0.49269238620126554,
   0.6807695546013303
  ],
  "n_cells": 66,
  "n_pos": 104,
  "n_neg": 225
 },
 "within_model": {
  "Z_S_w4g": {
   "cmunext": 0.0,
   "egeunet": 0.9444444444444444,
   "lbunet": 0.5,
   "malunet": 0.5,
   "maresunet": 0.8888888888888888,
   "mhorunet": 0.25,
   "mk_unet": 0.875,
   "pbe": "-(6/6)",
   "tinyunet": 0.7,
   "ucmnet": 0.5,
   "utanet": "-(6/6)"
  },
  "Mstat_D_l50": {
   "cmunext": "-(6/6)",
   "egeunet": "-(6/6)",
   "lbunet": "-(6/6)",
   "malunet": 0.3333333333333333,
   "maresunet": 0.8,
   "mhorunet": 0.25,
   "mk_unet": 1.0,
   "pbe": "-(6/6)",
   "tinyunet": "-(6/6)",
   "ucmnet": "-(6/6)",
   "utanet": "-(6/6)"
  }
 }
}
```

## dZ
```
{
 "n_risk_cells": 41,
 "n_new_collapse": 13,
 "auc_all_risk": 0.7815934065934066,
 "ci95_all_risk": [
  0.6264367816091954,
  0.9193548387096774
 ],
 "n_risk_folded": 25,
 "n_new_collapse_folded": 13,
 "auc_folded": 0.6955128205128205,
 "ci95_folded": [
  0.4675324675324675,
  0.8782051282051282
 ],
 "n_cells_dZ_gt0": 27,
 "n_cells_dZ_lt0": 2,
 "n_new_with_dZ_gt0": 9,
 "n_risk_nonnew_with_dZ_gt0": 6,
 "new_cells": {
  "cmunext|PH2": {
   "dZ": 0.0,
   "Z_dep": 0.0,
   "Z_train": 0.0
  },
  "cmunext|isic18": {
   "dZ": 0.0,
   "Z_dep": 0.0,
   "Z_train": 0.0
  },
  "cmunext|kvasir": {
   "dZ": 0.025,
   "Z_dep": 0.025,
   "Z_train": 0.0
  },
  "maresunet|isic17": {
   "dZ": 0.0062500000000000056,
   "Z_dep": 0.197265625,
   "Z_train": 0.191015625
  },
  "maresunet|isic18": {
   "dZ": 0.0,
   "Z_dep": 0.463671875,
   "Z_train": 0.463671875
  },
  "mk_unet|kvasir": {
   "dZ": 0.024999999999999994,
   "Z_dep": 0.09791666666666665,
   "Z_train": 0.07291666666666666
  },
  "tinyunet|PH2": {
   "dZ": 0.00625,
   "Z_dep": 0.00625,
   "Z_train": 0.0
  },
  "tinyunet|isic17": {
   "dZ": 0.0,
   "Z_dep": 0.0,
   "Z_train": 0.0
  },
  "utanet|PH2": {
   "dZ": 0.1,
   "Z_dep": 0.1,
   "Z_train": 0.0
  },
  "utanet|cvc": {
   "dZ": 0.059375,
   "Z_dep": 0.059375,
   "Z_train": 0.0
  },
  "utanet|isic17": {
   "dZ": 0.49921875,
   "Z_dep": 0.49921875,
   "Z_train": 0.0
  },
  "utanet|isic18": {
   "dZ": 0.65546875,
   "Z_dep": 0.65546875,
   "Z_train": 0.0
  },
  "utanet|kvasir": {
   "dZ": 0.22890625,
   "Z_dep": 0.22890625,
   "Z_train": 0.0
  }
 }
}
```

## excluded by model
```
{
 "cmunext": {
  "dep_unexecuted": 0,
  "dep_train_only": 0,
  "train_unexecuted": 0,
  "dep_targets": 44,
  "harmless_channels_max": 0,
  "harmless_in_zeroed_max": 0,
  "bn_off_max": 0,
  "autograd_fail": 0
 },
 "egeunet": {
  "dep_unexecuted": 0,
  "dep_train_only": 0,
  "train_unexecuted": 0,
  "dep_targets": 119,
  "harmless_channels_max": 0,
  "harmless_in_zeroed_max": 0,
  "bn_off_max": 0,
  "autograd_fail": 0
 },
 "lbunet": {
  "dep_unexecuted": 0,
  "dep_train_only": 0,
  "train_unexecuted": 0,
  "dep_targets": 104,
  "harmless_channels_max": 0,
  "harmless_in_zeroed_max": 0,
  "bn_off_max": 0,
  "autograd_fail": 0
 },
 "malunet": {
  "dep_unexecuted": 0,
  "dep_train_only": 0,
  "train_unexecuted": 0,
  "dep_targets": 109,
  "harmless_channels_max": 0,
  "harmless_in_zeroed_max": 0,
  "bn_off_max": 0,
  "autograd_fail": 0
 },
 "maresunet": {
  "dep_unexecuted": 0,
  "dep_train_only": 0,
  "train_unexecuted": 0,
  "dep_targets": 79,
  "harmless_channels_max": 0,
  "harmless_in_zeroed_max": 0,
  "bn_off_max": 0,
  "autograd_fail": 0
 },
 "mhorunet": {
  "dep_unexecuted": 0,
  "dep_train_only": 0,
  "train_unexecuted": 0,
  "dep_targets": 449,
  "harmless_channels_max": 0,
  "harmless_in_zeroed_max": 0,
  "bn_off_max": 0,
  "autograd_fail": 0
 },
 "mk_unet": {
  "dep_unexecuted": 0,
  "dep_train_only": 0,
  "train_unexecuted": 0,
  "dep_targets": 86,
  "harmless_channels_max": 0,
  "harmless_in_zeroed_max": 0,
  "bn_off_max": 0,
  "autograd_fail": 0
 },
 "pbe": {
  "dep_unexecuted": 0,
  "dep_train_only": 7,
  "train_unexecuted": 0,
  "dep_targets": 53,
  "harmless_channels_max": 0,
  "harmless_in_zeroed_max": 0,
  "bn_off_max": 0,
  "autograd_fail": 0
 },
 "tinyunet": {
  "dep_unexecuted": 0,
  "dep_train_only": 0,
  "train_unexecuted": 0,
  "dep_targets": 73,
  "harmless_channels_max": 0,
  "harmless_in_zeroed_max": 0,
  "bn_off_max": 0,
  "autograd_fail": 0
 },
 "ucmnet": {
  "dep_unexecuted": 0,
  "dep_train_only": 0,
  "train_unexecuted": 0,
  "dep_targets": 58,
  "harmless_channels_max": 0,
  "harmless_in_zeroed_max": 0,
  "bn_off_max": 0,
  "autograd_fail": 0
 },
 "utanet": {
  "dep_unexecuted": 2,
  "dep_train_only": 0,
  "train_unexecuted": 2,
  "dep_targets": 48,
  "harmless_channels_max": 0,
  "harmless_in_zeroed_max": 0,
  "bn_off_max": 0,
  "autograd_fail": 0
 }
}
```
