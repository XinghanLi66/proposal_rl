#!/bin/bash
# Usage:
#   bash scripts/train.sh sft [config]           — torchrun (verl FSDP SFT)
#   bash scripts/train.sh rl  [config]           — python (verl Ray GRPO)
#   bash scripts/train.sh sft configs/base.yaml --resume

set -euo pipefail

STAGE=${1:?'Usage: train.sh <sft|rl> [config] [--resume]'}
CONFIG=${2:-configs/base.yaml}
EXTRA=${@:3}
NGPU=${NGPU:-8}

LOG_DIR=$(python3 -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG'))['runs_dir'])")/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${STAGE}_$(date +%Y%m%d_%H%M%S).log"

echo "Stage   : $STAGE"
echo "Config  : $CONFIG"
echo "Log     : $LOG"

export DISABLE_VERSION_CHECK=1

if [[ "$STAGE" == "sft" ]]; then
    # verl SFT: torchrun + FSDP
    conda run -n loongflow_ml --no-capture-output \
      torchrun \
        --nproc_per_node="$NGPU" \
        --master_port=29500 \
        train/sft.py \
          --config "$CONFIG" \
          $EXTRA \
      2>&1 | tee "$LOG"
else
    # verl RL (GRPO): Ray single-process launcher
    NGPU="$NGPU" conda run -n loongflow_ml --no-capture-output \
      python train/rl.py \
        --config "$CONFIG" \
        $EXTRA \
      2>&1 | tee "$LOG"
fi
