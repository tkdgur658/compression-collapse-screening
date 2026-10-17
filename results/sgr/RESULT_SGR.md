# RESULT_SGR (decide, once)

time: 2026-10-17 17:33:24 KST

| hypothesis | verdict | detail |
|---|---|---|
| H_R1 SGR repairs >= 39/55 | FAIL | {"n_repaired": 24, "need": 39, "n_cells": 55} |
| H_R2 SGR > B1 (McNemar, Holm) | FAIL | {"b_SGR_only": 3, "c_B1_only": 5, "n_pairs": 55, "p_one_sided": 0.85546875} |
| H_R4 SGR >= 0.8 B4 (subset, Holm) | FAIL | {"n_SGR": 7, "n_B4": 6, "point_ok": true, "p_boot": 0.0835} |

Holm: {"p": {"H_R2": 0.85546875, "H_R4": 0.0835}, "reject": {"H_R4": false, "H_R2": false}}

Branches: B1 repairs more cells than SGR: SGR discarded, B1 reported as the R2 implementation; H_R1 not met: report "label-free repair under mask / bit-width preservation reaches 24/55 cells"; R2 = partial verification

## Cell counts (main 55)

| arm | repaired |
|---|---|
| B0 | 0/55 |
| B1 | 26/55 |
| B2 | 13/55 |
| B3 | 24/55 |
| SGR | 24/55 |

## Auxiliary (checkpoint level, subset, w4, n)

```
{
 "main": {
  "B0": {
   "cells_repaired": 0,
   "n_cells": 55,
   "ckpt_R_ge_0.8": 15,
   "n_ckpt_with_value": 274
  },
  "B1": {
   "cells_repaired": 26,
   "n_cells": 55,
   "ckpt_R_ge_0.8": 130,
   "n_ckpt_with_value": 274
  },
  "B2": {
   "cells_repaired": 13,
   "n_cells": 55,
   "ckpt_R_ge_0.8": 72,
   "n_ckpt_with_value": 274
  },
  "B3": {
   "cells_repaired": 24,
   "n_cells": 55,
   "ckpt_R_ge_0.8": 119,
   "n_ckpt_with_value": 274
  },
  "SGR": {
   "cells_repaired": 24,
   "n_cells": 55,
   "ckpt_R_ge_0.8": 138,
   "n_ckpt_with_value": 274
  }
 },
 "sub": {
  "B4": {
   "cells_repaired": 6,
   "n_cells": 13,
   "ckpt_R_ge_0.8": 32,
   "n_ckpt_with_value": 65
  },
  "A1": {
   "cells_repaired": 1,
   "n_cells": 13,
   "ckpt_R_ge_0.8": 18,
   "n_ckpt_with_value": 65
  },
  "A2": {
   "cells_repaired": 5,
   "n_cells": 13,
   "ckpt_R_ge_0.8": 31,
   "n_ckpt_with_value": 65
  },
  "A3": {
   "cells_repaired": 1,
   "n_cells": 13,
   "ckpt_R_ge_0.8": 15,
   "n_ckpt_with_value": 65
  },
  "SGR": {
   "cells_repaired": 7,
   "n_cells": 13,
   "ckpt_R_ge_0.8": 49,
   "n_ckpt_with_value": 65
  }
 },
 "w4": {
  "B0w": {
   "cells_repaired": 0,
   "n_cells": 15,
   "ckpt_R_ge_0.8": 9,
   "n_ckpt_with_value": 75
  },
  "B1w": {
   "cells_repaired": 9,
   "n_cells": 15,
   "ckpt_R_ge_0.8": 48,
   "n_ckpt_with_value": 75
  },
  "B3w": {
   "cells_repaired": 1,
   "n_cells": 15,
   "ckpt_R_ge_0.8": 13,
   "n_ckpt_with_value": 75
  },
  "B5": {
   "cells_repaired": 13,
   "n_cells": 15,
   "ckpt_R_ge_0.8": 63,
   "n_ckpt_with_value": 75
  },
  "SGRw": {
   "cells_repaired": 7,
   "n_cells": 15,
   "ckpt_R_ge_0.8": 32,
   "n_ckpt_with_value": 75
  }
 },
 "naux": {
  "SGR_n10": {
   "cells_repaired": 3,
   "n_cells": 6,
   "ckpt_R_ge_0.8": 16,
   "n_ckpt_with_value": 30
  },
  "SGR_n30": {
   "cells_repaired": 3,
   "n_cells": 6,
   "ckpt_R_ge_0.8": 19,
   "n_ckpt_with_value": 30
  },
  "B1_n10": {
   "cells_repaired": 3,
   "n_cells": 6,
   "ckpt_R_ge_0.8": 14,
   "n_ckpt_with_value": 30
  },
  "B1_n30": {
   "cells_repaired": 4,
   "n_cells": 6,
   "ckpt_R_ge_0.8": 15,
   "n_ckpt_with_value": 30
  },
  "SGR": {
   "cells_repaired": 3,
   "n_cells": 6,
   "ckpt_R_ge_0.8": 19,
   "n_ckpt_with_value": 30
  }
 }
}
```

Re-screen flags (tau = 0.738, reported only): {"B0": {"flagged": 251, "n": 274}, "B1": {"flagged": 155, "n": 274}, "B2": {"flagged": 205, "n": 274}, "B3": {"flagged": 151, "n": 274}, "SGR": {"flagged": 148, "n": 274}, "A1": {"flagged": 46, "n": 65}, "A2": {"flagged": 37, "n": 65}, "A3": {"flagged": 50, "n": 65}, "B4": {"flagged": 35, "n": 65}, "B0w": {"flagged": 64, "n": 75}, "B1w": {"flagged": 28, "n": 75}, "B3w": {"flagged": 64, "n": 75}, "B5": {"flagged": 14, "n": 75}, "SGRw": {"flagged": 39, "n": 75}, "B1_n10": {"flagged": 16, "n": 30}, "B1_n30": {"flagged": 14, "n": 30}, "SGR_n10": {"flagged": 14, "n": 30}, "SGR_n30": {"flagged": 14, "n": 30}}

Tuned-parameter fraction (by construction, H_R3 not a hypothesis): {"B1": [0.36170686993302714, 0.5294690853709692], "SGR": [0.0008319212764780658, 0.41168153439098104], "A1": [0.000824524573820069, 0.30859560224295773], "A3": [0.0038475735719470825, 0.2651369680619021], "B4": [0.4897345194597066, 0.5014092457140342], "B1w": [0.9999531265134931, 1.0], "SGRw": [0.0004661169309817415, 0.35746589170164805], "B1_n10": [0.5014084870353188, 0.5014092457140342], "B1_n30": [0.5014084870353188, 0.5014092457140342], "SGR_n10": [0.010635158233826297, 0.3821426756018788], "SGR_n30": [0.010635158233826297, 0.3449302432855037]}

Integrity: {"max_fp32_repro": 0.0, "max_B0_repro": 0.0, "E1_all_pass": true, "state_restored_all": true, "mask_preserved_all": true, "levels_ok_all": true, "exact_count_all": true, "overlap_zero_all": true, "gpu_s_total": 31740.339829206467}

Missing: {}; FAIL: {"main": ["egeunet__isic18__k3"]}
