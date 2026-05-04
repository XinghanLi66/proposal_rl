#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# SFT for exp05 — with_research_question prompt strategy
#   Output → runs/sft/with_research_question/final
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO=/newcpfs/lxh/agentic-training/proposal_rl
cd "$REPO"

STRATEGY="with_research_question"
FINETUNE_MODE=${FINETUNE_MODE:-full}
NGPU=${NGPU:-8}

source "$(dirname "$0")/_sft_lib.sh"
run_sft
