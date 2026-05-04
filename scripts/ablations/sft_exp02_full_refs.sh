#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# SFT for exp02 — full_refs prompt strategy
#   This is the original SFT (prompts already stored in train_cot.jsonl).
#   Re-running here produces runs/sft/full_refs/final for a fair comparison.
#   Output → runs/sft/full_refs/final
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO=/newcpfs/lxh/agentic-training/proposal_rl
cd "$REPO"

STRATEGY="full_refs"
FINETUNE_MODE=${FINETUNE_MODE:-full}
NGPU=${NGPU:-8}

source "$(dirname "$0")/_sft_lib.sh"
run_sft
