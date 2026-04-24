#!/usr/bin/env bash
# Fetch reference lists from Semantic Scholar API.
# Usage: bash scripts/01_fetch_refs.sh [--split train|val|test] [--limit N]
set -euo pipefail

REPO=/newcpfs/lxh/agentic-training/proposal_rl
cd "$REPO"

SPLIT=${SPLIT:-train}
LIMIT=${LIMIT:-8000}   # initial subset; expand with --limit 0 for all
LOG_DIR=runs/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/fetch_${SPLIT}_$(date +%Y%m%d_%H%M%S).log"

echo "[fetch_refs] split=$SPLIT limit=$LIMIT → $LOG"

conda run -n loongflow_ml --no-capture-output \
  python data/fetch_refs.py \
    --config configs/base.yaml \
    --split "$SPLIT" \
    --limit "$LIMIT" \
    ${S2_API_KEY:+--api-key "$S2_API_KEY"} \
  2>&1 | tee "$LOG"

echo "[fetch_refs] Done."
