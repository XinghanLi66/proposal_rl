#!/usr/bin/env bash
# Synthesize CoT proposals via the local Claude service.
set -euo pipefail

REPO=/newcpfs/lxh/agentic-training/proposal_rl
cd "$REPO"

# Use the local Claude service (same credentials as run_claude.sh)
export ANTHROPIC_BASE_URL=http://10.39.10.241:10001
export ANTHROPIC_AUTH_TOKEN=123
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1

LIMIT=${LIMIT:-}
LOG_DIR=runs/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/synthesize_cot_$(date +%Y%m%d_%H%M%S).log"

echo "[synthesize_cot] → $LOG"

conda run -n loongflow_ml --no-capture-output \
  python data/synthesize_cot.py \
    --config configs/base.yaml \
    ${LIMIT:+--limit "$LIMIT"} \
  2>&1 | tee "$LOG"

echo "[synthesize_cot] Done."
