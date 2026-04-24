#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Exp 02 — Full reference list (ablation: top_k → full_refs)
#   prompt strategy : full_refs    (all refs, truncated to 40 × 400 chars)
#   finetune mode   : full
#   reward          : embed-based PRS
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

EXP_NAME="exp02_full_refs"
EXP_LABEL="full-refs (full-ft / embed-PRS)"
STRATEGY="full_refs"
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
