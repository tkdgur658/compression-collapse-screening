#!/bin/bash
: "${CCS_WORK:=$(cd "$(dirname "$0")/.." && pwd)}"  # path configuration of the released copy (README)
# re-smoke of the w4 block after excluding train-only tensors from w4 diagnosis/tuning (output smoke2/, smoke/ kept)
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 SGR_SMOKE_DIR=smoke2
P=${CCS_PYTHON:-python}
cd ${CCS_WORK}/sgr
for m in pbe utanet maresunet; do nice -n 10 $P -B sgr_run.py --smoke ${m}__etis__k1 --blocks w4; done
nice -n 10 $P -B sgr_run.py --smoke pbe__etis__k1 --blocks main
echo SMOKE_END
