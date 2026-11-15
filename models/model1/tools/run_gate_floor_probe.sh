#!/usr/bin/env bash
# Gate-floor probe: does a softer floor on r (FUSION.GATE_R_FLOOR) fix what the sensor-failure probe v1 found,
# namely that the gate drives r to about 0 for a failed lidar and drops the stream entirely? Evaluates the already
# trained M5 checkpoint (best-on-val epoch 63; trained with floor 1e-4) at eval-time floors 0.05, 0.2 and 0.5 under the
# same synthetic failures (tools/eval_corruptions.py) on clear VAL, with no retraining. Run it like
# tools/run_corruption_probe.sh; then tools/summarize_gate_floor_probe.py. Exploratory: eval-time floor only.
set -u
mkdir -p /opt/OpenPCDet/tools/cfgs/model1
cp /workspace/models/model1/configs/stf_multimodal_dataset.yaml /opt/OpenPCDet/tools/cfgs/dataset_configs/
cp /workspace/models/model1/configs/model1_*.yaml /opt/OpenPCDet/tools/cfgs/model1/
sed -i 's/USE_AMP: True/USE_AMP: False/' /opt/OpenPCDet/tools/cfgs/model1/model1_*.yaml
export PYTHONPATH=/workspace/models/model1:/workspace/models/model1/tools
OUTP=/opt/OpenPCDet/output/corruption_probe_v2_floor; mkdir -p $OUTP
VALINFOS=/opt/OpenPCDet/data/kitti_format_mm/kitti_infos_val.pkl
OUT=/opt/OpenPCDet/output/model1/model1_lidar_rgb/m5_fp32_seed666
CK=$OUT/ckpt/checkpoint_epoch_63.pth
for floor in 0.05 0.2 0.5; do
  for cond in clean cam_zeros cam_occlusion cam_dark cam_blur lidar_sparse lidar_sector both_dark_sparse; do
    label=M5f$floor
    rm -f $OUTP/rel_${label}_${cond}.jsonl
    if [ "$cond" = "clean" ]; then unset M1_EVAL_CORRUPTION; else export M1_EVAL_CORRUPTION=eval_corruptions:$cond; fi
    M1_RELIABILITY_LOG=$OUTP/rel_${label}_${cond}.jsonl python /workspace/models/model1/tools/openpcdet_entry.py test \
        --cfg_file cfgs/model1/model1_lidar_rgb.yaml --extra_tag m5_fp32_seed666 --ckpt $CK --eval_tag probe_floor${floor}_$cond \
        --set MODEL.FUSION.GATE_R_FLOOR $floor > $OUTP/${label}_${cond}.log 2>&1
    res=$OUT/eval/epoch_63/val/probe_floor${floor}_$cond/result.pkl
    python /workspace/models/model1/tools/eval_calibration.py --result $res --infos $VALINFOS --out $OUTP/ece_${label}_${cond}.json > $OUTP/ece_${label}_${cond}.txt 2>&1
    echo "$(date +%T) done floor $floor $cond"
  done
done
echo PROBE_DONE