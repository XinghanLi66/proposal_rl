#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Exp 04 — LLM selects top-K refs, then synthesizes related-work from them
#           (ablation: top_k → top_k_related_work)
#   prompt strategy : top_k_related_work  (LLM selects 5, then synthesizes narrative)
#   finetune mode   : full
#   reward          : embed-based PRS
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

EXP_NAME="exp04_topk_related_work"
EXP_LABEL="top-k→related-work (full-ft / embed-PRS)"
STRATEGY="top_k_related_work"
FINETUNE_MODE="full"
REWARD_TYPE="prs"
EXTRA_OVERRIDES="prompt_builder.top_k=5"

# shellcheck source=_lib.sh
source "$(dirname "$0")/_lib.sh"

setup_experiment
register_dashboard
start_dashboard_refresh
run_hparam_search
run_full_training

finish_experiment
