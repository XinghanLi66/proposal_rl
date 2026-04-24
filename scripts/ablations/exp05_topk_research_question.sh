#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Exp 05 — Top-K refs + LLM-generated research question appended
#           (ablation: top_k → with_research_question)
#   prompt strategy : with_research_question  (top-K list + 1-3 sentence RQ)
#   finetune mode   : full
#   reward          : embed-based PRS
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

EXP_NAME="exp05_topk_rq"
EXP_LABEL="top-k+research-q (full-ft / embed-PRS)"
STRATEGY="with_research_question"
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
