#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Exp 06 — LoRA fine-tuning instead of full fine-tuning
#           (ablation: full-finetune → lora)
#   prompt strategy : top_k_refs
#   finetune mode   : lora
#   reward          : embed-based PRS
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

EXP_NAME="exp06_lora"
EXP_LABEL="lora (top-k / embed-PRS)"
STRATEGY="top_k_refs"
FINETUNE_MODE="lora"
REWARD_TYPE="prs"
EXTRA_OVERRIDES=""

# shellcheck source=_lib.sh
source "$(dirname "$0")/_lib.sh"

setup_experiment
register_dashboard
start_dashboard_refresh
run_hparam_search
run_full_training
