#!/bin/bash
: "${CCS_WORK:=$(cd "$(dirname "$0")/.." && pwd)}"  # path configuration of the released copy (README)
set -eu
cd ${CCS_WORK}/theoryq
test ! -e FROZEN_SHA256SUMS
Q=${CCS_WORK}/qdeploy
(
sha256sum PREREG_THEORYQ_FROZEN.md theoryq_core.py theoryq_run.py theoryq_decide.py theoryq_synth.py run_theoryq.sh frozen_src/qfix_a.py frozen_src/qdeploy.py frozen_src/qdeploy_analyze.py
sha256sum $Q/f2_fuse.py $Q/f2b_fold_all.py $Q/FOLD_PAIRS.json $Q/MANIFEST_330.sha256 $Q/ITEMS.json
find ${CCS_WORK}/qfix/A/code/models -name "*.py" -not -path "*__pycache__*" | sort | xargs sha256sum
) > FROZEN_SHA256SUMS
date "+%F %T %Z" > FROZEN_AT
wc -l FROZEN_SHA256SUMS; cat FROZEN_AT; head -9 FROZEN_SHA256SUMS; sha256sum FROZEN_SHA256SUMS
sha256sum -c --quiet FROZEN_SHA256SUMS && echo CHECK_OK
