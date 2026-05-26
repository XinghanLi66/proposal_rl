#!/bin/bash
# Run end-to-end benchmark sweep for a single experiment checkpoint.
#
# Usage:
#   bash scripts/08_benchmark.sh exp13_full_refs_sft_rl
#   EXP=exp13 TASK=char_lm N=20 bash scripts/08_benchmark.sh
#
# Env overrides:
#   EXP          experiment name prefix (required if no positional arg)
#   TASK         benchmark task name (default: char_lm)
#   N_PROPOSALS  number of proposals (default: 20)
#   WORKERS      parallel worker count (default: 1)

set -euo pipefail

REPO=/newcpfs/lxh/agentic-training/proposal_rl
CONDA_ENV=loongflow_ml

EXP="${1:-${EXP:-}}"
TASK="${TASK:-char_lm}"
N_PROPOSALS="${N_PROPOSALS:-20}"
WORKERS="${WORKERS:-1}"

if [[ -z "$EXP" ]]; then
    echo "Usage: $0 <exp_name_prefix>" >&2
    echo "  e.g. $0 exp13_full_refs_sft_rl" >&2
    exit 1
fi

# Find latest rl/final checkpoint for the experiment
CHECKPOINT=$(find "$REPO/runs/exps" -maxdepth 3 -path "*${EXP}*/rl/final" -type d \
    | sort -t_ -k1,1 | tail -1)

if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: no rl/final checkpoint found for experiment prefix '$EXP'" >&2
    exit 1
fi

echo "Experiment  : $EXP"
echo "Checkpoint  : $CHECKPOINT"
echo "Task        : $TASK"
echo "N proposals : $N_PROPOSALS"
echo "Workers     : $WORKERS"
echo ""

LOG_DIR="$REPO/runs/logs"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/benchmark_${EXP}_$(date +%Y%m%d_%H%M%S).log"

conda run -n "$CONDA_ENV" --no-capture-output \
    python "$REPO/benchmark/run_benchmark.py" \
        --checkpoint "$CHECKPOINT" \
        --task "$TASK" \
        --n-proposals "$N_PROPOSALS" \
        --workers "$WORKERS" \
        --config "$REPO/configs/base.yaml" \
    2>&1 | tee "$LOGFILE"

echo ""
echo "Log: $LOGFILE"
echo ""

# Print comparison table
conda run -n "$CONDA_ENV" --no-capture-output \
    python "$REPO/benchmark/report.py" \
        --task "$TASK" \
        --config "$REPO/configs/base.yaml"
