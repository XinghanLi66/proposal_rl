#!/bin/bash
# Usage:
#   bash scripts/train.sh sft [config]
#   bash scripts/train.sh rl  [config]
#   bash scripts/train.sh sft configs/base.yaml --resume

set -euo pipefail

STAGE=${1:?'Usage: train.sh <sft|rl> [config] [--resume]'}
CONFIG=${2:-configs/base.yaml}
EXTRA=${@:3}

LOG_DIR=$(python3 -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG'))['runs_dir'])")/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${STAGE}_$(date +%Y%m%d_%H%M%S).log"

echo "Stage   : $STAGE"
echo "Config  : $CONFIG"
echo "Log     : $LOG"

export DISABLE_VERSION_CHECK=1

conda run -n loongflow_ml --no-capture-output \
  torchrun \
    --nproc_per_node=8 \
    --master_port=29501 \
    train/${STAGE}.py \
      --config "$CONFIG" \
      $EXTRA \
  2>&1 | tee "$LOG"
