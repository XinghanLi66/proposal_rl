#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Exp 12 — with_research_question, full-FT, PRS reward, with matched SFT
#   prompt strategy : with_research_question
#   finetune mode   : full
#   reward          : embed-based PRS
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

EXP_NAME="exp12_research_q_sft_rl"
EXP_LABEL="research-q SFT+RL (full-ft / PRS)"
STRATEGY="with_research_question"
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
