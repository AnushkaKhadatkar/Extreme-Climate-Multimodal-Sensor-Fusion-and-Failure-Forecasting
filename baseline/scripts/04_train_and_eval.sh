#!/usr/bin/env bash
# Train PV-RCNN on STF clear-weather-only, pick the best-on-val checkpoint, then
# evaluate it on all three STF test splits (clear/light-fog/dense-fog),
# reproducing Hahner et al. ICCV 2021 Table 2 PV-RCNN dagger row. RUN ON A CUDA
# GPU BOX, normally inside the baseline/docker image -- see baseline/README.md.
#
# Prerequisites:
#   1. OpenPCDet with spconv + CUDA torch (the baseline/docker image).
#   2. Converted data at <openpcdet_root>/data/kitti_format/ with infos + gt
#      database from 03_create_infos.py.
#   3. configs/pv_rcnn_stf.yaml -> <openpcdet_root>/tools/cfgs/kitti_models/pv_rcnn_stf.yaml
#      configs/kitti_dataset_stf.yaml -> <openpcdet_root>/tools/cfgs/dataset_configs/kitti_dataset_stf.yaml
#
# Everything runs from <openpcdet_root>/tools/: OpenPCDet resolves _BASE_CONFIG_
# and DATA_PATH ('../data/kitti_format') relative to the working directory.
#
# Usage:
#   OPENPCDET_ROOT=/opt/OpenPCDet ./04_train_and_eval.sh
# Env:
#   NUM_GPUS    (default 1)
#   EXTRA_TAG   (default stf_baseline) -- output/kitti_models/pv_rcnn_stf/<EXTRA_TAG>/
#   EPOCHS      (default: config's 80; set only for a smoke test, never for the gate run)
#   SKIP_TRAIN=1  evaluate an existing run without training
#   BEST_CKPT   override automatic best-on-val selection

set -euo pipefail

: "${OPENPCDET_ROOT:?Set OPENPCDET_ROOT to your OpenPCDet checkout path}"
: "${NUM_GPUS:=1}"
: "${EXTRA_TAG:=stf_baseline}"
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="cfgs/kitti_models/pv_rcnn_stf.yaml"
OUT_DIR="$OPENPCDET_ROOT/output/kitti_models/pv_rcnn_stf/$EXTRA_TAG"
LOG_DIR="$OUT_DIR/baseline_logs"
EPOCH_ARGS=()
[ -n "${EPOCHS:-}" ] && EPOCH_ARGS=(--epochs "$EPOCHS")

cd "$OPENPCDET_ROOT/tools"
mkdir -p "$LOG_DIR"

# --fix_random_seed pins OpenPCDet's internal seed (boolean flag, no value).
# cuDNN/spconv kernels are still not bitwise deterministic -- see README.
# --max_ckpt_save_num 80 keeps every epoch so best-on-val selection sees all of
# them (OpenPCDet's default keeps only the last 30).
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
    echo "=== Training PV-RCNN on STF clear (${EPOCHS:-80} epochs) ==="
    TRAIN_ARGS=(--cfg_file "$CFG" --extra_tag "$EXTRA_TAG" --fix_random_seed --max_ckpt_save_num 80 "${EPOCH_ARGS[@]}")
    if [ "$NUM_GPUS" -gt 1 ]; then
        bash scripts/dist_train.sh "$NUM_GPUS" "${TRAIN_ARGS[@]}" 2>&1 | tee "$LOG_DIR/train.log"
    else
        python train.py "${TRAIN_ARGS[@]}" 2>&1 | tee "$LOG_DIR/train.log"
    fi
fi

# Per the paper: "we select the snapshot with the best performance on the clear
# weather validation set and test it on the aforementioned test splits."
if [ -z "${BEST_CKPT:-}" ]; then
    echo "=== Evaluating all checkpoints on clear val to find the best snapshot ==="
    python test.py --cfg_file "$CFG" --extra_tag "$EXTRA_TAG" --eval_all --max_waiting_mins 0 \
        2>&1 | tee "$LOG_DIR/eval_all_val.log"
    python "$SCRIPTS_DIR/07_read_eval_logs.py" best --log "$LOG_DIR/eval_all_val.log" --ckpt-dir "$OUT_DIR/ckpt"
    BEST_CKPT="$(python "$SCRIPTS_DIR/07_read_eval_logs.py" best --log "$LOG_DIR/eval_all_val.log" \
        --ckpt-dir "$OUT_DIR/ckpt" --print-ckpt-only)"
fi
echo "Best-on-val checkpoint: $BEST_CKPT"

# Each test split needs BOTH overrides. KittiDataset reads frames and ground
# truth from INFO_PATH[mode]; DATA_SPLIT only picks the ImageSets file and the
# output folder name. Overriding DATA_SPLIT alone silently evaluates val again
# (BASELINE_TASKS.md R1). --set must be the last argument (argparse REMAINDER).
for split in test_clear test_light_fog test_dense_fog; do
    echo "--- eval on $split ---"
    python test.py --cfg_file "$CFG" --extra_tag "$EXTRA_TAG" --ckpt "$BEST_CKPT" \
        --set DATA_CONFIG.DATA_SPLIT.test "$split" DATA_CONFIG.INFO_PATH.test "kitti_infos_${split}.pkl" \
        2>&1 | tee "$LOG_DIR/eval_${split}.log"
done

# Gate: Car, R40, IoU 0.5 (the 'Car AP_R40@0.70, 0.50, 0.50:' block), 3d AP,
# moderate column of test_clear. Also checks each log evaluated the right
# number of frames, and the moderate > easy inversion on both fog splits.
echo
python "$SCRIPTS_DIR/07_read_eval_logs.py" gate \
    --logs "$LOG_DIR/eval_test_clear.log" "$LOG_DIR/eval_test_light_fog.log" "$LOG_DIR/eval_test_dense_fog.log"
echo "Logs: $LOG_DIR"
