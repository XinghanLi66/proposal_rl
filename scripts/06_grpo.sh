#!/usr/bin/env bash
# GRPO fine-tuning via verl Ray+FSDP+vLLM (single-process Ray launcher, no torchrun).
set -euo pipefail

REPO=/newcpfs/lxh/agentic-training/proposal_rl
cd "$REPO"

NGPU=${NGPU:-8}
LOG_DIR=runs/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/grpo_$(date +%Y%m%d_%H%M%S).log"

echo "[grpo] ngpu=$NGPU  backend=verl-ray → $LOG"

export DISABLE_VERSION_CHECK=1
NGPU="$NGPU" conda run -n loongflow_ml --no-capture-output \
  python train/rl.py \
    --config configs/base.yaml \
    ${RESUME:+--resume} \
  2>&1 | tee "$LOG"

echo "[grpo] Done."
