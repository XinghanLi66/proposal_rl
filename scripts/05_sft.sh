#!/usr/bin/env bash
# SFT training on 8 GPUs with DeepSpeed ZeRO-2.
set -euo pipefail

REPO=/newcpfs/lxh/agentic-training/proposal_rl
cd "$REPO"

NGPU=${NGPU:-8}
LOG_DIR=runs/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/sft_$(date +%Y%m%d_%H%M%S).log"

echo "[sft] ngpu=$NGPU → $LOG"

conda run -n loongflow_ml --no-capture-output \
  torchrun \
    --nproc_per_node "$NGPU" \
    --master_port 29500 \
    train/sft.py \
      --config configs/base.yaml \
      ${RESUME:+--resume} \
  2>&1 | tee "$LOG"

echo "[sft] Done."
