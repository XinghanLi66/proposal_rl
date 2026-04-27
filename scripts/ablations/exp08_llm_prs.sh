#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Exp 08 — LLM-judge PRS reward instead of embedding-based PRS
#           (ablation: embed-PRS → llm-judge-PRS)
#
#   This requires a custom reward that calls Claude to judge how well the
#   generated proposal aligns with the source abstract.  The reward module
#   reads fas.strategy=llm_judge to use LLMJudgeFAS for the scoring.
#
#   prompt strategy : top_k_refs
#   finetune mode   : full
#   reward          : llm-judge PRS  (fas.strategy=llm_judge, reward_type=fas)
#
# Note: LLM calls per step × num_generations add API latency. Reduce
#       concurrency if hitting rate limits (fas.judge_topk controls
#       how many abstracts are scored per proposal).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

EXP_NAME="exp08_llm_prs"
EXP_LABEL="llm-judge-PRS (top-k / full-ft)"
STRATEGY="top_k_refs"
FINETUNE_MODE="full"
REWARD_TYPE="fas"   # reuse FAS reward path; LLMJudgeFAS approximates LLM-PRS
# Use llm_judge strategy in FAS; judge against val corpus (closest proxy to PRS)
EXTRA_OVERRIDES="fas.strategy=llm_judge fas.judge_topk=5 rl.reward_fas_weight=0.8 rl.reward_format_weight=0.2 rl.reward_antileak_weight=0.0"

# shellcheck source=_lib.sh
source "$(dirname "$0")/_lib.sh"

# Check prerequisite
VAL_INDEX="$REPO/runs/eval/val_index.npz"
if [[ ! -f "$VAL_INDEX" ]]; then
    echo "[exp08] ERROR: val_index.npz not found at $VAL_INDEX"
    echo "[exp08] Build it first:  conda run -n loongflow_ml python eval/build_index.py --split val"
    exit 1
fi

# LLM judge is slower — reduce group size to keep wall-clock time reasonable
HP_LRS=(2e-6 5e-6)
HP_KLS=(0.02 0.05)

setup_experiment
register_dashboard
start_dashboard_refresh
run_hparam_search
run_full_training

finish_experiment
