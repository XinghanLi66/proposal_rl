#!/usr/bin/env bash
# Build FAISS-free embedding index for val and test splits.
set -euo pipefail

REPO=/newcpfs/lxh/agentic-training/proposal_rl
cd "$REPO"

LOG_DIR=runs/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/build_index_$(date +%Y%m%d_%H%M%S).log"

echo "[build_index] → $LOG"

conda run -n loongflow_ml --no-capture-output \
  python eval/build_index.py \
    --config configs/base.yaml \
    --splits val test \
    --batch-size 256 \
    --from-local \
  2>&1 | tee "$LOG"

echo "[build_index] Done."
