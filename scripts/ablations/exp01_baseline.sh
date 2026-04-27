#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Exp 01 — Baseline
#   prompt strategy : top_k_refs   (LLM selects top-5 references)
#   finetune mode   : full
#   reward          : embed-based PRS
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

EXP_NAME="exp01_baseline"
EXP_LABEL="baseline (top-k / full-ft / embed-PRS)"
STRATEGY="top_k_refs"
FINETUNE_MODE="full"
REWARD_TYPE="prs"
EXTRA_OVERRIDES=""

# shellcheck source=_lib.sh
source "$(dirname "$0")/_lib.sh"

setup_experiment
register_dashboard
start_dashboard_refresh
run_hparam_search
run_full_training

finish_experiment
