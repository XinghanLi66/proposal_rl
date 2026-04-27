#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Exp 07 — Embedding-based FAS reward instead of PRS
#           (ablation: embed-PRS → embed-FAS)
#   prompt strategy : top_k_refs
#   finetune mode   : full
#   reward          : embed-based FAS (requires val_index.npz)
#
# Prerequisites: runs/eval/val_index.npz must exist.
#   Build it with: conda run -n loongflow_ml python eval/build_index.py --split val
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

EXP_NAME="exp07_fas"
EXP_LABEL="embed-FAS (top-k / full-ft)"
STRATEGY="top_k_refs"
FINETUNE_MODE="full"
REWARD_TYPE="fas"
EXTRA_OVERRIDES="rl.reward_fas_weight=0.6 rl.reward_format_weight=0.2 rl.reward_antileak_weight=0.2"

# shellcheck source=_lib.sh
source "$(dirname "$0")/_lib.sh"

# Check prerequisite
VAL_INDEX="$REPO/runs/eval/val_index.npz"
if [[ ! -f "$VAL_INDEX" ]]; then
    echo "[exp07] ERROR: val_index.npz not found at $VAL_INDEX"
    echo "[exp07] Build it first:  conda run -n loongflow_ml python eval/build_index.py --split val"
    exit 1
fi

setup_experiment
register_dashboard
start_dashboard_refresh
run_hparam_search
run_full_training

finish_experiment
