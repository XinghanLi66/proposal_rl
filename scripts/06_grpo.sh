#!/usr/bin/env bash
# GRPO fine-tuning on top of the SFT checkpoint.
set -euo pipefail

REPO=/newcpfs/lxh/agentic-training/proposal_rl
cd "$REPO"

NGPU=${NGPU:-8}
LOG_DIR=runs/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/grpo_$(date +%Y%m%d_%H%M%S).log"

echo "[grpo] ngpu=$NGPU → $LOG"

# TRL 1.0.0 version check fails with transformers>4.55; bypass it
export DISABLE_VERSION_CHECK=1
conda run -n loongflow_ml --no-capture-output \
  torchrun \
    --nproc_per_node "$NGPU" \
    --master_port 29501 \
    train/grpo.py \
      --config configs/base.yaml \
      ${SFT_CHECKPOINT:+--sft-checkpoint "$SFT_CHECKPOINT"} \
      ${RESUME:+--resume} \
  2>&1 | tee "$LOG"

echo "[grpo] Done."
