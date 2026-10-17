# Protocol v1 (pre-registered before any v1 result)

Fixed on 2026-10-03, before any protocol-v1 training run was started. Decisions on the metric,
resize, checkpoint rule and bug fixes were made by the user on 2026-10-03 (see the
resubmission plan, section 6). A change to anything below is an amendment: record what had
been seen at the time, then bump the protocol id.

## Data
- Datasets: PH2, ISIC 2017, ISIC 2018 (dermoscopy); Kvasir-SEG, CVC-ClinicDB, ETIS-Larib (colonoscopy).
- Splits: the fixed legacy folders `{train,val,test}_k`, k = 1..5. Dermoscopy 60/20/20,
  colonoscopy 80/10/10. Split k is trained with seed k.
- Preprocessing: per-image min-max to [0,1]; aspect-ratio-preserving bilinear resize (antialias)
  to 256 with centred zero padding; masks resized bilinearly and binarised at 0.5.
- Augmentation (train only, per sample): horizontal flip p=0.5, vertical flip p=0.5,
  rotation p=0.5 with angle ~ U[0, 360) drawn per sample (image bilinear, mask nearest, fill 0).

## Training (identical for every model)
- AdamW, lr 1e-3, weight decay 1e-2, betas (0.9, 0.999); cosine annealing to 1e-6 over 100 epochs;
  batch 16 (drop last); gradient-norm clip 5.0.
- Loss: BCE + soft Dice on the final output. Models whose published architecture emits five
  deep-supervision side outputs (UCMNet, EGE-UNet, LB-UNet, and the proposed PBE-Net) add them
  with weights 0.1, 0.2, 0.3, 0.4, 0.5. The proposed model adds the train-only matching loss
  (lambda_s = 0.1, centred cosine on 3x3 morphological-gradient bands at skip heads 1-3).
- Determinism: seed k for python/numpy/torch, seeded DataLoader generator and workers,
  cudnn.deterministic = True, benchmark = False.

## Selection and evaluation
- Checkpoint: the epoch with the highest mean per-image IoU on the validation split.
- Test metric: per-image IoU (and Dice) at threshold 0.5 on the 256x256 padded inputs, averaged
  over test images. Pooled (dataset-level) IoU is stored as a secondary metric only.
- Table cells: mean and standard deviation over k = 1..5. Significance: two-sided paired t-test
  over the five splits against the proposed model, Holm correction within each dataset column.

## Rows
`configs/rows.json`, suites `main` (11 models) and `cumulative` (component ablation).
No row is added or removed after its results are seen.
