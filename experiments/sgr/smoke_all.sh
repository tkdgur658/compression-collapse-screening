#!/bin/bash
: "${CCS_WORK:=$(cd "$(dirname "$0")/.." && pwd)}"  # path configuration of the released copy (README)
# CPU smoke of every code path (PREREG_SGR §9): 2 steps, 8 train images, no Dice. Sequential, nice 10, 2 threads.
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
P=${CCS_PYTHON:-python}
cd ${CCS_WORK}/sgr
for m in pbe utanet maresunet; do nice -n 10 $P -B sgr_run.py --smoke ${m}__etis__k1 --blocks main,sub,w4,naux; done
for m in mk_unet mhorunet tinyunet ucmnet lbunet cmunext egeunet malunet; do nice -n 10 $P -B sgr_run.py --smoke ${m}__etis__k1 --blocks main; done
echo SMOKE_END
