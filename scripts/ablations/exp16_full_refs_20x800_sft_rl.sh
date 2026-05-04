#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Exp 16 — full_refs 20×800, full-FT, PRS reward, with matched SFT
#   Ablates reference density: 20 refs × 800 abstract chars vs default 40×400.
#   prompt strategy : full_refs  (max_refs=20, abstract_chars=800)
#   finetune mode   : full
#   reward          : embed-based PRS
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

EXP_NAME="exp16_full_refs_20x800_sft_rl"
EXP_LABEL="full-refs 20×800 SFT+RL (full-ft / PRS)"
STRATEGY="full_refs"
FINETUNE_MODE="full"
REWARD_TYPE="prs"
EXTRA_OVERRIDES="prompt_builder.max_refs=20 prompt_builder.abstract_chars=800"

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
