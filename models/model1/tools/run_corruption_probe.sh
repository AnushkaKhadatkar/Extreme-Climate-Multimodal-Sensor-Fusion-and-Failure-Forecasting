#!/usr/bin/env bash
# Sensor-failure probe: evaluate Model 1's best checkpoints (M4 lidar-only, M4b camera without gate, M5 camera with gate)
# on the clear VAL split with one injected failure at a time (tools/eval_corruptions.py), recording AP, ECE and the
# reliability scores. Exploratory only: val, not the test splits; SYNTHETIC failures; one seed per model.
# Runs inside the Model 1 image after the three train_and_eval.sh runs exist (checkpoints under the output volume), e.g.
#   docker run -d --gpus all --shm-size 16g -v stf_kitti_mm:/opt/OpenPCDet/data/kitti_format_mm \
#       -v <repo>/models/model1/output:/opt/OpenPCDet/output -v <repo>:/workspace:ro stf-model1:pinned \
#       bash /workspace/models/model1/tools/run_corruption_probe.sh
# Then: python models/model1/tools/summarize_corruption_probe.py <repo>/models/model1/output/corruption_probe_v1
# The best-on-val epochs (71, 72, 63) are those picked by train_and_eval.sh for the 2026-11-14 runs.
set -u
mkdir -p /opt/OpenPCDet/tools/cfgs/model1
cp /workspace/models/model1/configs/stf_multimodal_dataset.yaml /opt/OpenPCDet/tools/cfgs/dataset_configs/
cp /workspace/models/model1/configs/model1_*.yaml /opt/OpenPCDet/tools/cfgs/model1/
sed -i 's/USE_AMP: True/USE_AMP: False/' /opt/OpenPCDet/tools/cfgs/model1/model1_*.yaml     # same deployed setup as the runs
export PYTHONPATH=/workspace/models/model1:/workspace/models/model1/tools
OUTP=/opt/OpenPCDet/output/corruption_probe_v1; mkdir -p $OUTP
VALINFOS=/opt/OpenPCDet/data/kitti_format_mm/kitti_infos_val.pkl
run_one () {   # label cfgname cfgfile tag epoch cond
  label=$1; cfgname=$2; cfgfile=$3; tag=$4; ep=$5; cond=$6
  out=/opt/OpenPCDet/output/model1/$cfgname/$tag
  ck=$out/ckpt/checkpoint_epoch_$ep.pth
  rm -f $OUTP/rel_${label}_${cond}.jsonl
  if [ "$cond" = "clean" ]; then unset M1_EVAL_CORRUPTION; else export M1_EVAL_CORRUPTION=eval_corruptions:$cond; fi
  M1_RELIABILITY_LOG=$OUTP/rel_${label}_${cond}.jsonl python /workspace/models/model1/tools/openpcdet_entry.py test \
      --cfg_file $cfgfile --extra_tag $tag --ckpt $ck --eval_tag probe_$cond > $OUTP/${label}_${cond}.log 2>&1
  res=$out/eval/epoch_$ep/val/probe_$cond/result.pkl
  python /workspace/models/model1/tools/eval_calibration.py --result $res --infos $VALINFOS --out $OUTP/ece_${label}_${cond}.json > $OUTP/ece_${label}_${cond}.txt 2>&1
  echo "$(date +%T) done $label $cond (eval exit logged in log)"
}
ALL="clean cam_zeros cam_occlusion cam_dark cam_blur lidar_sparse lidar_sector both_dark_sparse"
for c in clean lidar_sparse lidar_sector; do run_one M4 model1_lidar_only cfgs/model1/model1_lidar_only.yaml m4_lidar_only_fp32_seed666 71 $c; done
for c in $ALL; do run_one M4b model1_lidar_rgb_nogate cfgs/model1/model1_lidar_rgb_nogate.yaml m4b_nogate_fp32_seed666 72 $c; done
for c in $ALL; do run_one M5 model1_lidar_rgb cfgs/model1/model1_lidar_rgb.yaml m5_fp32_seed666 63 $c; done
echo PROBE_DONE