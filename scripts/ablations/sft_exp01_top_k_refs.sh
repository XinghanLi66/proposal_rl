#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# SFT for exp01 / exp06 / exp07 / exp08 — top_k_refs prompt strategy
#   Trains a full fine-tuned SFT checkpoint whose input distribution matches
#   the RL prompt format used by exp01, exp06, exp07, exp08.
#   Output → runs/sft/top_k_refs/final
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO=/newcpfs/lxh/agentic-training/proposal_rl
cd "$REPO"

STRATEGY="top_k_refs"
FINETUNE_MODE=${FINETUNE_MODE:-full}
NGPU=${NGPU:-8}

source "$(dirname "$0")/_sft_lib.sh"
run_sft
