#!/usr/bin/env bash
# Train PV-RCNN on STF clear-weather-only, then evaluate on all three STF test
# splits (clear/light-fog/dense-fog), reproducing Hahner et al. ICCV 2021 Table 2
# PV-RCNN dagger row. RUN ON A CUDA GPU BOX -- see REPRODUCTION_PLAN.md for why
# this cannot run on the machine these scripts were written on (no spconv/CUDA).
#
# Prerequisites on the GPU box:
#   1. An OpenPCDet checkout with spconv + CUDA torch installed (follow OpenPCDet's
#      own INSTALL.md -- not duplicated here, versions change and context.md notes
#      the Docker image with pinned versions is still "planned", not done).
#   2. This repo's baseline/data/kitti_format/ copied to
#      <openpcdet_root>/data/kitti_format/ (or symlinked -- do not train off a
#      mounted drive, see context.md section 8).
#   3. configs/pv_rcnn_stf.yaml -> <openpcdet_root>/tools/cfgs/kitti_models/pv_rcnn_stf.yaml
#      configs/kitti_dataset_stf.yaml -> <openpcdet_root>/tools/cfgs/dataset_configs/kitti_dataset_stf.yaml
#      (must go in these exact stock-mirroring locations -- pv_rcnn_stf.yaml's
#      _BASE_CONFIG_ is the relative path 'cfgs/dataset_configs/kitti_dataset_stf.yaml',
#      resolved from wherever tools/train.py or tools/test.py is run, i.e. tools/.)
#   4. Run 03_create_infos.py once (generates the five kitti_infos_*.pkl + the
#      gt database) before training.
#
# Usage:
#   OPENPCDET_ROOT=/path/to/OpenPCDet ./04_train_and_eval.sh

set -euo pipefail

: "${OPENPCDET_ROOT:?Set OPENPCDET_ROOT to your OpenPCDet checkout path}"
: "${NUM_GPUS:=1}"
CFG="tools/cfgs/kitti_models/pv_rcnn_stf.yaml"
EXTRA_TAG="stf_baseline"

cd "$OPENPCDET_ROOT"

# --fix_random_seed pins OpenPCDet's internal fixed seed (it's a boolean flag,
# not a value you pass -- context.md section 6's "Fix seeds" is satisfied by
# just passing the flag, there is no separate --seed argument in tools/train.py
# as of the version checked 2026-09-22).
echo "=== Training PV-RCNN on STF clear (80 epochs) ==="
if [ "$NUM_GPUS" -gt 1 ]; then
    bash tools/scripts/dist_train.sh "$NUM_GPUS" \
        --cfg_file "$CFG" \
        --extra_tag "$EXTRA_TAG" \
        --fix_random_seed
else
    python tools/train.py \
        --cfg_file "$CFG" \
        --extra_tag "$EXTRA_TAG" \
        --fix_random_seed
fi

# Per the paper: "we select the snapshot with the best performance on the clear
# weather validation set and test it on the aforementioned test splits." This is
# --eval_all (evaluates every saved checkpoint against val, not just the last),
# NOT just taking the epoch-80 checkpoint. Do not skip this step -- taking the
# final-epoch checkpoint instead of the best-on-val one is a protocol deviation
# that context.md's gate does not forgive silently.
echo "=== Evaluating all checkpoints against clear val to find the best snapshot ==="
python tools/test.py \
    --cfg_file "$CFG" \
    --extra_tag "$EXTRA_TAG" \
    --eval_all \
    --set DATA_CONFIG.DATA_SPLIT.test val

echo "=== Best-on-val checkpoint selected. Now evaluating that checkpoint on all three STF test splits ==="
echo "(Read the best epoch number from the eval_all log above / eval/eval_all_default/eval_list_val.txt,"
echo " set BEST_CKPT below, then run the three blocks -- kept separate so each test split's log is distinct.)"

: "${BEST_CKPT:?Set BEST_CKPT to the checkpoint path selected by --eval_all against val}"

for split in test_clear test_light_fog test_dense_fog; do
    echo "--- eval on $split ---"
    python tools/test.py \
        --cfg_file "$CFG" \
        --extra_tag "$EXTRA_TAG" \
        --ckpt "$BEST_CKPT" \
        --set DATA_CONFIG.DATA_SPLIT.test "$split" \
        > "eval_${split}.log" 2>&1
    echo "  log: eval_${split}.log"
done

echo
echo "=== Done. Gate check: grep 'Car AP@0.50' in each eval_*.log, moderate column ==="
echo "Target: clear/moderate = 77.05 +/- 2 (pass band 75.05-79.05). See context.md section 3."
