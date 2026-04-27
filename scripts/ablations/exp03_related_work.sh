#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Exp 03 — LLM-synthesized related-work narrative from full reference list
#           (ablation: top_k → related_work)
#   prompt strategy : related_work (Claude synthesizes 3-5 paragraphs from 40 refs)
#   finetune mode   : full
#   reward          : embed-based PRS
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

EXP_NAME="exp03_related_work"
EXP_LABEL="related-work/full (full-ft / embed-PRS)"
STRATEGY="related_work"
FINETUNE_MODE="full"
REWARD_TYPE="prs"
EXTRA_OVERRIDES="prompt_builder.max_refs=40"

# shellcheck source=_lib.sh
source "$(dirname "$0")/_lib.sh"

setup_experiment
register_dashboard
start_dashboard_refresh
run_hparam_search
run_full_training

finish_experiment
