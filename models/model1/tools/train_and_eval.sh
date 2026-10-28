#!/usr/bin/env bash
# Train Model 1 on STF clear weather, pick the best-on-val checkpoint, evaluate it
# on the three STF test splits, then compute calibration (ECE, D-ECE) and the
# per-split reliability summary. Same protocol as the baseline's
# baseline/scripts/04_train_and_eval.sh, so the numbers are comparable.
#
# Runs inside the Model 1 image (models/model1/docker), normally via
#   powershell -File models/model1/docker/run_model1_docker.ps1 -Detach `
#       -Cmd "bash /workspace/models/model1/tools/train_and_eval.sh"
#
# Env:
#   MODEL1_CFG  (default cfgs/model1/model1_lidar_rgb.yaml; the ablation is cfgs/model1/model1_lidar_only.yaml)
#   EXTRA_TAG   (default model1) -- output/model1/<cfg name>/<EXTRA_TAG>/
#   EPOCHS      (default: the config's 80; set only for smoke tests)
#   SKIP_TRAIN=1  evaluate an existing run
#   BEST_CKPT   override best-on-val selection

set -euo pipefail

: "${OPENPCDET_ROOT:=/opt/OpenPCDet}"
: "${MODEL1_CFG:=cfgs/model1/model1_lidar_rgb.yaml}"
: "${EXTRA_TAG:=model1}"
TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$TOOLS_DIR/../../.." && pwd)"
ENTRY="python $TOOLS_DIR/openpcdet_entry.py"
CFG_NAME="$(basename "$MODEL1_CFG" .yaml)"
OUT_DIR="$OPENPCDET_ROOT/output/model1/$CFG_NAME/$EXTRA_TAG"
LOG_DIR="$OUT_DIR/model1_logs"
DATA_DIR="$OPENPCDET_ROOT/data/kitti_format_mm"
READ_LOGS="$REPO_DIR/baseline/scripts/07_read_eval_logs.py"   # from rohil/baseline-eval-fixes (R3)
EPOCH_ARGS=()
[ -n "${EPOCHS:-}" ] && EPOCH_ARGS=(--epochs "$EPOCHS")
mkdir -p "$LOG_DIR"

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
    echo "=== Training $CFG_NAME on STF clear (${EPOCHS:-80} epochs) ==="
    $ENTRY train --cfg_file "$MODEL1_CFG" --extra_tag "$EXTRA_TAG" --fix_random_seed \
        --max_ckpt_save_num 80 "${EPOCH_ARGS[@]}" 2>&1 | tee "$LOG_DIR/train.log"
fi

if [ -z "${BEST_CKPT:-}" ]; then
    if [ ! -f "$READ_LOGS" ]; then
        echo "ERROR: $READ_LOGS not found. It comes with Rohil's R3 (rohil/baseline-eval-fixes);"
        echo "merge that first, or set BEST_CKPT to the checkpoint to evaluate." >&2
        exit 1
    fi
    echo "=== Evaluating all checkpoints on clear val ==="
    $ENTRY test --cfg_file "$MODEL1_CFG" --extra_tag "$EXTRA_TAG" --eval_all --max_waiting_mins 0 \
        2>&1 | tee "$LOG_DIR/eval_all_val.log"
    BEST_CKPT="$(python "$READ_LOGS" best --log "$LOG_DIR/eval_all_val.log" --ckpt-dir "$OUT_DIR/ckpt" --print-ckpt-only)"
fi
echo "Best-on-val checkpoint: $BEST_CKPT"

# Each test split needs BOTH overrides (BASELINE_TASKS R1): INFO_PATH picks the frames
# and ground truth; DATA_SPLIT alone would silently evaluate val again.
for split in test_clear test_light_fog test_dense_fog; do
    echo "--- eval on $split ---"
    rm -f "$LOG_DIR/reliability_${split}.jsonl"
    M1_RELIABILITY_LOG="$LOG_DIR/reliability_${split}.jsonl" \
    $ENTRY test --cfg_file "$MODEL1_CFG" --extra_tag "$EXTRA_TAG" --ckpt "$BEST_CKPT" --eval_tag "$split" \
        --set DATA_CONFIG.DATA_SPLIT.test "$split" DATA_CONFIG.INFO_PATH.test "kitti_infos_${split}.pkl" \
        2>&1 | tee "$LOG_DIR/eval_${split}.log"
    result="$(ls -t "$OUT_DIR"/eval/epoch_*/"$split"/"$split"/result.pkl | head -n 1)"
    python "$TOOLS_DIR/eval_calibration.py" --result "$result" --infos "$DATA_DIR/kitti_infos_${split}.pkl" \
        --out "$LOG_DIR/ece_${split}.json" | tee "$LOG_DIR/ece_${split}.txt"
done

echo
echo "=== AP (Car 3D, R40, IoU 0.5) and frame counts ==="
python "$READ_LOGS" gate \
    --logs "$LOG_DIR/eval_test_clear.log" "$LOG_DIR/eval_test_light_fog.log" "$LOG_DIR/eval_test_dense_fog.log" || true
echo "(The pass/fail band above is the BASELINE's reproduction gate; for Model 1 read only the AP values.)"

if ls "$LOG_DIR"/reliability_*.jsonl >/dev/null 2>&1 && [ -s "$LOG_DIR/reliability_test_clear.jsonl" ]; then
    echo
    echo "=== Reliability per split ==="
    python "$TOOLS_DIR/summarize_reliability.py" "$LOG_DIR"/reliability_test_{clear,light_fog,dense_fog}.jsonl
fi
echo "Logs: $LOG_DIR"
