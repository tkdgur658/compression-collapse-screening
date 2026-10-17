"""F4 shared constants: variants, datasets, reference (full v1) run resolution. Read-only on the archive."""
import os as _ccs_os  # path configuration of the released copy (README, Configuration)
CCS_REPO_ROOT = _ccs_os.path.normpath(_ccs_os.path.join(_ccs_os.path.dirname(_ccs_os.path.abspath(__file__)), '../../..'))
CCS_WORK_ROOT = _ccs_os.environ.get('CCS_WORK', _ccs_os.path.join(CCS_REPO_ROOT, 'experiments'))
CCS_ARCHIVE_ROOT = _ccs_os.environ.get('CCS_ARCHIVE', _ccs_os.path.join(CCS_REPO_ROOT, 'archive'))
import json
import os

ARCH = CCS_ARCHIVE_ROOT
FILL = _ccs_os.path.join(CCS_WORK_ROOT, 'pbe_v1_fill/runs/pbe')
BASE = _ccs_os.path.join(CCS_WORK_ROOT, 'abl')
RUNS = os.path.join(BASE, 'rr', 'abl4')
DATASETS = ['cvc', 'isic17', 'isic18']              # prereg section 1, fixed
VARIANTS = {'pbe_idskip': {'skip': 'plain'}, 'pbe_lam0': {'lambda_s': 0},
            'pbe_mbv2': {'block': 'mbv2'}, 'pbe_se': {'gate': 'se'}}
SPLITS = [1, 2, 3, 4, 5]


def ref_dir(model, ds, k):
    """full v1 run of `model` (archive first, else the v1 fill-in with a DONE marker)."""
    a = os.path.join(ARCH, 'runs', 'main', model, ds, f'k{k}')
    r = os.path.join(a, 'results.json')
    if os.path.exists(r) and json.load(open(r)).get('status') == 'complete':
        return a
    if model == 'pbe':
        f = os.path.join(FILL, ds, f'k{k}')
        if os.path.exists(os.path.join(f, 'DONE')) and os.path.exists(os.path.join(f, 'results.json')):
            return f
    raise FileNotFoundError(f'no complete reference run for {model} {ds} k{k}')


def ref_results(model, ds, k):
    return json.load(open(os.path.join(ref_dir(model, ds, k), 'results.json')))


def run_dir(row, ds, k):
    return os.path.join(RUNS, row, ds, f'k{k}')
