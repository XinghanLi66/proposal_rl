#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Exp 09 — top_k_refs, full-FT, PRS reward, with matched SFT
#   Ablates whether aligning SFT prompt format with RL prompt format improves
#   over the original full_refs SFT used by exp01–exp07.
#   prompt strategy : top_k_refs
#   finetune mode   : full
#   reward          : embed-based PRS
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

EXP_NAME="exp09_top_k_refs_sft_rl"
EXP_LABEL="top-k-refs SFT+RL (full-ft / PRS)"
STRATEGY="top_k_refs"
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
