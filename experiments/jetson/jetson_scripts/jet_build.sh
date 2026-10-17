#!/bin/bash
# PREREG_JETSON: build TensorRT FP16 engines from the fused ONNX files (engine build is not timed), then one untimed
# inference per engine on the PH2 image used for the ONNX check (--loadInputs / --exportOutput) for an output sanity check.
# No sudo, no jetson_clocks, no nvpmodel change, writes only under $D.
set -u
D=$HOME/lsh_latency_20261016
TRT=/usr/src/tensorrt/bin/trtexec
MODELS=${MODELS:-"pbe utanet mk_unet tinyunet ucmnet lbunet cmunext egeunet malunet"}
mkdir -p $D/engines $D/logs $D/out
for m in $MODELS; do
  O=$D/onnx/${m}_fused_op11.onnx; E=$D/engines/${m}_fp16.engine
  echo "$(date '+%F %T %Z') build $m start" >> $D/logs/build_driver.log
  $TRT --onnx=$O --fp16 --workspace=1024 --saveEngine=$E --buildOnly > $D/logs/${m}_fp16_build.log 2>&1
  rc=$?
  echo "$(date '+%F %T %Z') build $m rc=$rc" >> $D/logs/build_driver.log
  if [ $rc -eq 0 ]; then
    $TRT --loadEngine=$E --loadInputs=input:$D/onnx/input_ph2_test1_img0.bin --exportOutput=$D/out/${m}_fp16_output.json \
         --warmUp=0 --duration=0 --iterations=1 > $D/logs/${m}_fp16_outcheck.log 2>&1
    echo "$(date '+%F %T %Z') outcheck $m rc=$?" >> $D/logs/build_driver.log
  fi
done
echo "$(date '+%F %T %Z') BUILD_DONE" >> $D/logs/build_driver.log
