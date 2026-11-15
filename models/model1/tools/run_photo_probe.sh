#!/usr/bin/env bash
# Sensor-failure probe v3 (photometric-dropout variants). Evaluates the best-on-val checkpoints of M5p (gate, floor 0.1,
# photometric camera dropout, epoch 61) and M4bp (no gate, same dropout, epoch 59) on clear VAL under all eleven injected
# failures of tools/eval_corruptions.py, and re-evaluates M4b (epoch 72) and M5 (epoch 63) on the three HELD-OUT camera
# failures (cam_haze, cam_motion_blur, cam_color_cast) that probe v1 did not have. Plan: docs/RUN_LOG.md, section
# "M5p and M4bp launched". Exploratory: val, SYNTHETIC failures, one seed per model. Run like run_corruption_probe.sh;
# summarise with tools/summarize_photo_probe.py (merges this folder with corruption_probe_v1).
set -u
mkdir -p /opt/OpenPCDet/tools/cfgs/model1
cp /workspace/models/model1/configs/stf_multimodal_dataset.yaml /opt/OpenPCDet/tools/cfgs/dataset_configs/
cp /workspace/models/model1/configs/model1_*.yaml /opt/OpenPCDet/tools/cfgs/model1/
sed -i 's/USE_AMP: True/USE_AMP: False/' /opt/OpenPCDet/tools/cfgs/model1/model1_*.yaml
export PYTHONPATH=/workspace/models/model1:/workspace/models/model1/tools
OUTP=/opt/OpenPCDet/output/corruption_probe_v3; mkdir -p $OUTP
VALINFOS=/opt/OpenPCDet/data/kitti_format_mm/kitti_infos_val.pkl
run_one () {   # label cfgname tag epoch cond
  label=$1; cfgname=$2; tag=$3; ep=$4; cond=$5
  out=/opt/OpenPCDet/output/model1/$cfgname/$tag
  rm -f $OUTP/rel_${label}_${cond}.jsonl
  if [ "$cond" = "clean" ]; then unset M1_EVAL_CORRUPTION; else export M1_EVAL_CORRUPTION=eval_corruptions:$cond; fi
  M1_RELIABILITY_LOG=$OUTP/rel_${label}_${cond}.jsonl python /workspace/models/model1/tools/openpcdet_entry.py test \
      --cfg_file cfgs/model1/$cfgname.yaml --extra_tag $tag --ckpt $out/ckpt/checkpoint_epoch_$ep.pth --eval_tag probe_$cond > $OUTP/${label}_${cond}.log 2>&1
  python /workspace/models/model1/tools/eval_calibration.py --result $out/eval/epoch_$ep/val/probe_$cond/result.pkl --infos $VALINFOS \
      --out $OUTP/ece_${label}_${cond}.json > $OUTP/ece_${label}_${cond}.txt 2>&1
  echo "$(date +%T) done $label $cond"
}
ALL="clean cam_zeros cam_occlusion cam_dark cam_blur cam_haze cam_motion_blur cam_color_cast lidar_sparse lidar_sector both_dark_sparse"
HELD="cam_haze cam_motion_blur cam_color_cast"
for c in $ALL;  do run_one M5p  model1_lidar_rgb_photo        m5p_floor0.1_photo_fp32_seed666 61 $c; done
for c in $ALL;  do run_one M4bp model1_lidar_rgb_nogate_photo m4bp_nogate_photo_fp32_seed666  59 $c; done
for c in $HELD; do run_one M4b  model1_lidar_rgb_nogate       m4b_nogate_fp32_seed666         72 $c; done
for c in $HELD; do run_one M5   model1_lidar_rgb              m5_fp32_seed666                 63 $c; done
echo PROBE_DONE