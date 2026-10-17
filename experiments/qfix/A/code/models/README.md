# Model packages

One folder per model with a number in any table. Each has:
- `adapter.py`: `NAME`, `DEFAULT_CFG`, `build`, `forward_train`, `extra_loss`, `infer`, `is_train_only`
- `src/`: implementation (baselines: byte-identical legacy copies, loaded in isolation by `_vendored.py`)
- `manifest.json`: source lineage and sha256, validation status
- `fairness.json`: matched conditions, intended and unresolved differences

Model-specific losses and defaults live only in the package. Ablations of the proposed model are
configurations of `pbe` (see `configs/rows.json`), not separate packages. Do not edit a vendored
`src/` file in place: add a patched copy and record the diff in the manifest.
