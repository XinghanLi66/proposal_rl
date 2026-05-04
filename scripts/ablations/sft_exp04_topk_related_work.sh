#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# SFT for exp04 — top_k_related_work prompt strategy
#   Output → runs/sft/top_k_related_work/final
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO=/newcpfs/lxh/agentic-training/proposal_rl
cd "$REPO"

STRATEGY="top_k_related_work"
FINETUNE_MODE=${FINETUNE_MODE:-full}
NGPU=${NGPU:-8}

source "$(dirname "$0")/_sft_lib.sh"
run_sft
