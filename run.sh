#!/bin/bash
# run.sh — ASL classifier pipeline (MLP on landmarks)
#
# Usage:
#   ./run.sh            → runs full pipeline (collect → dataset → train → infer)
#   ./run.sh collect    → Stage 1 : collect 100 webcam frames per letter
#   ./run.sh dataset    → Stage 2 : extract MediaPipe landmarks → data.pickle
#   ./run.sh train      → Stage 3 : train MLP on landmark features
#   ./run.sh infer      → Stage 4 : run real-time webcam inference

set -e

STAGE=${1:-all}

# ── Stage functions ───────────────────────────────────────────────────────

run_collect() {
    echo ""
    echo "════════════════════════════════════════════"
    echo "  Stage 1 — Collecting webcam data"
    echo "════════════════════════════════════════════"
    python collect_data.py
}

run_dataset() {
    echo ""
    echo "════════════════════════════════════════════"
    echo "  Stage 2 — Extracting landmarks"
    echo "════════════════════════════════════════════"
    python create_dataset.py
}

run_train() {
    echo ""
    echo "════════════════════════════════════════════"
    echo "  Stage 3 — Training MLP"
    echo "════════════════════════════════════════════"
    python train_classifier_mlp.py
}

run_infer() {
    echo ""
    echo "════════════════════════════════════════════"
    echo "  Stage 4 — Running real-time inference"
    echo "════════════════════════════════════════════"
    python inference_classifier_mlp.py
}

# ── Dispatch ──────────────────────────────────────────────────────────────

case $STAGE in
    all)
        run_collect
        run_dataset
        run_train
        run_infer
        ;;
    collect) run_collect ;;
    dataset) run_dataset ;;
    train)   run_train   ;;
    infer)   run_infer   ;;
    *)
        echo ""
        echo "Unknown stage: '$STAGE'"
        echo ""
        echo "Usage: ./run.sh [stage]"
        echo ""
        echo "  collect   Stage 1 — collect 100 webcam frames per letter"
        echo "  dataset   Stage 2 — extract MediaPipe landmarks → data.pickle"
        echo "  train     Stage 3 — train MLP on landmark features"
        echo "  infer     Stage 4 — run real-time webcam inference"
        echo ""
        echo "  (no argument) — run all stages in order"
        exit 1
        ;;
esac

echo ""
echo "Done."
