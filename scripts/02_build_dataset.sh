#!/usr/bin/env bash
# Build (reference list → proposal) dataset records from fetched refs.
set -euo pipefail

REPO=/newcpfs/lxh/agentic-training/proposal_rl
cd "$REPO"

SPLITS=${SPLITS:-"train val test"}
LOG_DIR=runs/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/build_dataset_$(date +%Y%m%d_%H%M%S).log"

echo "[build_dataset] splits='$SPLITS' → $LOG"

conda run -n loongflow_ml --no-capture-output \
  python data/build_dataset.py \
    --config configs/base.yaml \
    --splits $SPLITS \
  2>&1 | tee "$LOG"

echo "[build_dataset] Done."
