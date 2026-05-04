#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Exp 11 — top_k_related_work, full-FT, PRS reward, with matched SFT
#   prompt strategy : top_k_related_work
#   finetune mode   : full
#   reward          : embed-based PRS
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

EXP_NAME="exp11_topk_rw_sft_rl"
EXP_LABEL="top-k-rw SFT+RL (full-ft / PRS)"
STRATEGY="top_k_related_work"
FINETUNE_MODE="full"
REWARD_TYPE="prs"
EXTRA_OVERRIDES=""

source "$(dirname "$0")/_combined_lib.sh"

setup_combined_experiment
register_dashboard
start_dashboard_refresh

run_cot_synthesis
run_sft_hparam_search
run_sft_full_training
run_rl_hparam_search
run_rl_full_training

finish_experiment
