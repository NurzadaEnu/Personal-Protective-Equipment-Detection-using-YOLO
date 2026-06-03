#!/usr/bin/env bash
# Run the four ablation configurations (M0..M3) across five seeds (0..4) on CHV.
# Wall-time estimate on 4×Quadro RTX 6000 at 200 epochs / batch 16:
#   - M0  ~ 0.5 h × 5 = 2.5 h
#   - M1  ~ 0.6 h × 5 = 3.0 h
#   - M2  ~ 0.65 h × 5 = 3.3 h
#   - M3  ~ 0.7 h × 5 = 3.5 h
#   ----
#   Total ≈ 12 h
#
# Usage:
#   bash run_ablation.sh
#
# Edit DATA / DEVICE / EPOCHS to taste.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABL_DIR="$HERE/../ablation"
DATA="${DATA:-$HERE/../../datasets/chv/data.yaml}"
DEVICE="${DEVICE:-0,1,2,3}"
EPOCHS="${EPOCHS:-200}"
SEEDS="${SEEDS:-0,1,2,3,4}"
WEIGHTS="${WEIGHTS:-yolo11s.pt}"
PROJECT="${PROJECT:-runs/ablation}"

run_cfg () {
    local cfg="$1" name="$2"
    echo "============================================================"
    echo "=> Training $name with $cfg"
    echo "============================================================"
    python "$HERE/train_multiseed.py" \
        --cfg "$cfg" \
        --data "$DATA" \
        --weights "$WEIGHTS" \
        --epochs "$EPOCHS" \
        --imgsz 640 \
        --batch 16 \
        --device "$DEVICE" \
        --seeds "$SEEDS" \
        --project "$PROJECT" \
        --name "$name"
}

run_cfg "$ABL_DIR/yolo11s_M0_baseline.yaml" "M0_baseline"
run_cfg "$ABL_DIR/yolo11s_M1_R1.yaml"       "M1_R1"
run_cfg "$ABL_DIR/yolo11s_M2_R1R2.yaml"     "M2_R1R2"
run_cfg "$ABL_DIR/yolo11s_M3_R1R2R3.yaml"   "M3_R1R2R3"

echo "============================================================"
echo "All four ablation configurations done."
echo "Now run: python aggregate_results.py --project $PROJECT"
echo "============================================================"
