#!/usr/bin/env bash
# Run FAS evaluation on a checkpoint (SFT or GRPO final).
set -euo pipefail

REPO=/newcpfs/lxh/agentic-training/proposal_rl
cd "$REPO"

CHECKPOINT=${CHECKPOINT:-runs/grpo/final}
BASE_MODEL=${BASE_MODEL:-}   # only needed for LoRA checkpoints
SPLIT=${SPLIT:-test}
LIMIT=${LIMIT:-500}
LOG_DIR=runs/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/eval_$(basename "$CHECKPOINT")_$(date +%Y%m%d_%H%M%S).log"

echo "[evaluate] checkpoint=$CHECKPOINT split=$SPLIT limit=$LIMIT → $LOG"

conda run -n loongflow_ml --no-capture-output \
  python eval/evaluate.py \
    --config configs/base.yaml \
    --checkpoint "$CHECKPOINT" \
    ${BASE_MODEL:+--base-model "$BASE_MODEL"} \
    --split "$SPLIT" \
    --limit "$LIMIT" \
    --batch-size 4 \
  2>&1 | tee "$LOG"

echo "[evaluate] Done. See $CHECKPOINT/eval_results/summary.json"
