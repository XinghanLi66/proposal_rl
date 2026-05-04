#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Exp 13 — full_refs, full-FT, PRS reward, with matched SFT
#   Mirrors exp02 (RL-only, full-FT, PRS) but adds a freshly-synthesised
#   full_refs CoT SFT stage before RL.
#   prompt strategy : full_refs
#   finetune mode   : full
#   reward          : embed-based PRS
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

EXP_NAME="exp13_full_refs_sft_rl"
EXP_LABEL="full-refs SFT+RL (full-ft / PRS)"
STRATEGY="full_refs"
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
