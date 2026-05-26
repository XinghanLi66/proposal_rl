#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Exp 17 — top_k_refs, full-FT, PPL reward, RL-only from exp09 SFT ckpt
#   Ablates reward signal: replaces PRS cosine-similarity with perplexity of
#   the source abstract under the policy model (mean log P(abstract | prompt,
#   proposal)).  Everything else mirrors exp09.
#   prompt strategy : top_k_refs
#   finetune mode   : full
#   reward          : ppl  (mean log-prob of source abstract under policy)
#   SFT init        : latest exp09 SFT final checkpoint (auto-detected)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

EXP_NAME="exp17_top_k_refs_ppl_rl"
EXP_LABEL="top-k-refs PPL RL (exp09 SFT init)"
STRATEGY="top_k_refs"
FINETUNE_MODE="full"
REWARD_TYPE="ppl"
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
