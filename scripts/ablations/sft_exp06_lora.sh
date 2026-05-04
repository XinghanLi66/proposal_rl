#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# SFT for exp06 — top_k_refs with LoRA fine-tuning
#   Produces a LoRA SFT checkpoint for exp06 (LoRA RL ablation).
#   Output → runs/sft/top_k_refs_lora/final
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO=/newcpfs/lxh/agentic-training/proposal_rl
cd "$REPO"

STRATEGY="top_k_refs"
FINETUNE_MODE="lora"
NGPU=${NGPU:-8}

source "$(dirname "$0")/_sft_lib.sh"
run_sft
