#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Exp 14 — full_refs, LoRA, PRS reward, with matched SFT
#   Mirrors exp13 but uses LoRA for both SFT and RL.
#   prompt strategy : full_refs
#   finetune mode   : lora
#   reward          : embed-based PRS
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

EXP_NAME="exp14_full_refs_lora_sft_rl"
EXP_LABEL="full-refs SFT+RL (LoRA / PRS)"
STRATEGY="full_refs"
FINETUNE_MODE="lora"
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
