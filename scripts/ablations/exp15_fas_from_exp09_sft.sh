#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Exp 15 — top_k_refs, full-FT, FAS reward, RL-only from exp09 SFT ckpt
#   Mirrors exp07 (FAS RL) but starts from the exp09 top_k_refs SFT checkpoint
#   instead of the shared full_refs SFT, isolating prompt-format alignment.
#   prompt strategy : top_k_refs
#   finetune mode   : full
#   reward          : FAS
#   SFT init        : latest exp09 SFT final checkpoint (auto-detected)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

EXP_NAME="exp15_fas_from_exp09_sft"
EXP_LABEL="top-k-refs FAS RL (exp09 SFT init)"
STRATEGY="top_k_refs"
FINETUNE_MODE="full"
REWARD_TYPE="fas"
EXTRA_OVERRIDES=""

# ── Locate exp09 SFT checkpoint (most recent run) ────────────────────────────
_REPO=/newcpfs/lxh/agentic-training/proposal_rl
_EXP09_CKPT=$(ls -d "${_REPO}/runs/exps/exp09_top_k_refs_sft_rl_"*/sft/final \
              2>/dev/null | sort | tail -1)

if [[ -z "$_EXP09_CKPT" ]]; then
    echo "[${EXP_NAME}] ERROR: No exp09 SFT checkpoint found under"
    echo "  ${_REPO}/runs/exps/exp09_top_k_refs_sft_rl_*/sft/final"
    echo "  Please run exp09 first, or set SFT_CKPT_OVERRIDE manually."
    exit 1
fi

export SFT_CKPT_OVERRIDE="$_EXP09_CKPT"
echo "[${EXP_NAME}] Using SFT checkpoint: ${SFT_CKPT_OVERRIDE}"

# ─────────────────────────────────────────────────────────────────────────────
source "$(dirname "$0")/_combined_lib.sh"

setup_combined_experiment
register_dashboard
start_dashboard_refresh

# Skip CoT synthesis + SFT — init from exp09 SFT
run_rl_hparam_search
run_rl_full_training

finish_experiment
