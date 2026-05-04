#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Shared library for all ablation scripts.
#
# Variables that each script must set before calling setup_experiment:
#   EXP_NAME      — short identifier, e.g. "exp01_baseline"
#   EXP_LABEL     — human-readable tab name in the dashboard
#   STRATEGY      — prompt_builder.strategy value
#   FINETUNE_MODE — "full" or "lora"
#   REWARD_TYPE   — "prs" or "fas"
#   EXTRA_OVERRIDES — (optional) space-separated "key=value" pairs
# ─────────────────────────────────────────────────────────────────────────────

REPO=/newcpfs/lxh/agentic-training/proposal_rl
cd "$REPO"

NGPU=${NGPU:-8}
DASHBOARD_PORT=${DASHBOARD_PORT:-8080}
CONDA_ENV=loongflow_ml

# Initialize conda when running in non-login shells (tmux, SSH exec, etc.)
# where ~/.bashrc / conda init has not been sourced.
if ! command -v conda &>/dev/null; then
    _CONDA_BASE=/newcpfs/lxh/miniconda3
    if [[ -f "$_CONDA_BASE/etc/profile.d/conda.sh" ]]; then
        # shellcheck source=/dev/null
        source "$_CONDA_BASE/etc/profile.d/conda.sh"
    else
        export PATH="$_CONDA_BASE/bin:$PATH"
    fi
fi

# Pick a random free TCP port to avoid EADDRINUSE when 2 experiments
# run concurrently on the same machine.
_free_port() {
    python3 -c "import socket; s=socket.socket(); s.bind(('',0)); \
p=s.getsockname()[1]; s.close(); print(p)"
}

if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
    PY="python"
    TORCHRUN_CMD="torchrun"
else
    PY="conda run -n $CONDA_ENV --no-capture-output python"
    TORCHRUN_CMD="conda run -n $CONDA_ENV --no-capture-output torchrun"
fi
export DISABLE_VERSION_CHECK=1

# Hparam grid: 2 LRs × 2 KL coefficients = 4 mini-runs
HP_LRS=(2e-6 5e-6)
HP_KLS=(0.02 0.05)
# Dataset rows to use during mini-search (faster than full dataset)
HP_LIMIT=512

# ─────────────────────────────────────────────────────────────────────────────
setup_experiment() {
    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    EXP_ID="${EXP_NAME}_${ts}"
    EXP_DIR="$REPO/runs/exps/${EXP_ID}"
    EXP_LOG_DIR="${EXP_DIR}/logs"
    BASE_CFG="${EXP_DIR}/config_base.yaml"

    mkdir -p "$EXP_DIR" "$EXP_LOG_DIR"

    echo "[${EXP_NAME}] EXP_ID   = $EXP_ID"
    echo "[${EXP_NAME}] EXP_DIR  = $EXP_DIR"
    echo "[${EXP_NAME}] strategy = $STRATEGY"
    echo "[${EXP_NAME}] finetune = $FINETUNE_MODE"
    echo "[${EXP_NAME}] reward   = $REWARD_TYPE"

    # Resolve SFT checkpoint: explicit SFT_CHECKPOINT env var wins, then the
    # strategy-matched checkpoint, then the legacy runs/sft/final fallback.
    local _sft_subdir="${STRATEGY}"
    [[ "${FINETUNE_MODE}" == "lora" ]] && _sft_subdir="${STRATEGY}_lora"
    local _default_sft="${REPO}/runs/sft/${_sft_subdir}/final"
    local _legacy_sft="${REPO}/runs/sft/final"
    local _resolved_sft="${SFT_CHECKPOINT:-}"
    if [[ -z "$_resolved_sft" ]]; then
        if [[ -d "$_default_sft" ]]; then
            _resolved_sft="$_default_sft"
        else
            echo "[${EXP_NAME}] WARNING: strategy-matched SFT not found at ${_default_sft}"
            echo "[${EXP_NAME}]          falling back to ${_legacy_sft}"
            _resolved_sft="$_legacy_sft"
        fi
    fi
    echo "[${EXP_NAME}] sft_ckpt = ${_resolved_sft}"

    # Build the experiment's base config (without lr/kl overrides yet)
    # shellcheck disable=SC2206
    local overrides=(
        "prompt_builder.strategy=${STRATEGY}"
        "rl.finetune_mode=${FINETUNE_MODE}"
        "rl.reward_type=${REWARD_TYPE}"
        "rl.output_dir=${EXP_DIR}/rl"
        "rl.sft_checkpoint=${_resolved_sft}"
    )
    if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
        read -r -a extra_arr <<< "$EXTRA_OVERRIDES"
        overrides+=("${extra_arr[@]}")
    fi

    $PY scripts/make_config.py \
        --base configs/base.yaml \
        --out  "$BASE_CFG" \
        --set  "${overrides[@]}"
}

# ─────────────────────────────────────────────────────────────────────────────
register_dashboard() {
    local payload
    payload=$(python3 -c "
import json, sys
print(json.dumps({
    'exp_id':   '${EXP_ID}',
    'name':     '${EXP_LABEL}',
    'runs_dir': '${EXP_DIR}',
    'config': {
        'prompt_builder': {'strategy': '${STRATEGY}'},
        'rl': {
            'finetune_mode': '${FINETUNE_MODE}',
            'reward_type':   '${REWARD_TYPE}',
        }
    }
}))
")
    if curl -s -X POST \
            -H "Content-Type: application/json" \
            -d "$payload" \
            "http://localhost:${DASHBOARD_PORT}/api/experiments" \
            --max-time 5 > /dev/null 2>&1; then
        echo "[${EXP_NAME}] Dashboard tab registered."
    else
        echo "[${EXP_NAME}] Dashboard not reachable — continuing without registration."
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# run_one_mini <lr> <kl>  — 100-step probe; appends result to hparam_search.json
# ─────────────────────────────────────────────────────────────────────────────
run_one_mini() {
    local lr=$1 kl=$2
    local tag="lr${lr}_kl${kl}"
    local mini_out="${EXP_DIR}/hparam_${tag}"
    local mini_cfg="${EXP_DIR}/config_mini_${tag}.yaml"
    local mini_log="${EXP_LOG_DIR}/rl_${EXP_ID}_mini_${tag}.log"

    echo "[${EXP_NAME}] Mini-run: lr=${lr}  kl=${kl}"

    $PY scripts/make_config.py \
        --base "$BASE_CFG" \
        --out  "$mini_cfg" \
        --set  "rl.learning_rate=${lr}" \
               "rl.kl_coeff=${kl}" \
               "rl.num_train_epochs=1" \
               "rl.save_steps=9999" \
               "rl.logging_steps=5" \
               "rl.output_dir=${mini_out}" \
               "rl.limit=${HP_LIMIT}"

    # Run (don't abort on non-zero — log the error and continue)
    local mini_port
    mini_port=$(_free_port)
    $TORCHRUN_CMD \
        --nproc_per_node "$NGPU" \
        --master_port "$mini_port" \
        train/rl.py --config "$mini_cfg" \
        2>&1 | tee "$mini_log" || \
        echo "[${EXP_NAME}] WARNING: mini-run lr=${lr} kl=${kl} exited non-zero"

    # Extract mean reward from log
    local mean_reward
    mean_reward=$($PY scripts/ablations/extract_reward.py "$mini_log" 10)
    echo "[${EXP_NAME}] lr=${lr}  kl=${kl}  mean_reward=${mean_reward}"

    # Append to hparam_search.json
    python3 - <<PYEOF
import json, os
from pathlib import Path
f = Path("${EXP_DIR}/hparam_search.json")
data = {"results": [], "best": None}
if f.exists():
    try: data = json.loads(f.read_text())
    except: pass
data["results"].append({
    "lr": "${lr}", "kl": "${kl}",
    "mean_reward": float("${mean_reward}"), "steps": ${HP_LIMIT}
})
f.write_text(json.dumps(data, indent=2))
PYEOF
}

# ─────────────────────────────────────────────────────────────────────────────
run_hparam_search() {
    echo "[${EXP_NAME}] === Hparam search (${#HP_LRS[@]}×${#HP_KLS[@]} grid) ==="
    for lr in "${HP_LRS[@]}"; do
        for kl in "${HP_KLS[@]}"; do
            run_one_mini "$lr" "$kl"
        done
    done

    # Pick best and write .best_hparams
    $PY scripts/ablations/pick_best_hparam.py "$EXP_DIR"
    # shellcheck disable=SC1090
    source "${EXP_DIR}/.best_hparams"
    echo "[${EXP_NAME}] Best hparams: LR=${BEST_LR}  KL=${BEST_KL}"
}

# ─────────────────────────────────────────────────────────────────────────────
run_full_training() {
    local full_cfg="${EXP_DIR}/config_full.yaml"
    local log="${EXP_LOG_DIR}/rl_${EXP_ID}.log"

    echo "[${EXP_NAME}] === Full training: LR=${BEST_LR}  KL=${BEST_KL} ==="

    $PY scripts/make_config.py \
        --base "$BASE_CFG" \
        --out  "$full_cfg" \
        --set  "rl.learning_rate=${BEST_LR}" \
               "rl.kl_coeff=${BEST_KL}" \
               "rl.output_dir=${EXP_DIR}/rl"

    local full_port
    full_port=$(_free_port)
    $TORCHRUN_CMD \
        --nproc_per_node "$NGPU" \
        --master_port "$full_port" \
        train/rl.py --config "$full_cfg" \
        2>&1 | tee "$log" \
        && echo "[${EXP_NAME}] Training done → ${EXP_DIR}/rl/final" \
        || echo "[${EXP_NAME}] ERROR: full training exited non-zero — check $log"
}

# ─────────────────────────────────────────────────────────────────────────────
# Background dashboard ping (keeps the tab status refreshed during training)
# ─────────────────────────────────────────────────────────────────────────────
start_dashboard_refresh() {
    (
        while true; do
            sleep 30
            curl -s --max-time 3 \
                "http://localhost:${DASHBOARD_PORT}/api/experiment/${EXP_ID}/summary" \
                > /dev/null 2>&1 || true
        done
    ) &
    _REFRESH_PID=$!
    # Kill the background loop when the script exits
    trap 'kill "$_REFRESH_PID" 2>/dev/null || true' EXIT
}

# ─────────────────────────────────────────────────────────────────────────────
# Call at the very end of every exp script.
# Prints a status banner and then drops the tmux/shell to an interactive prompt
# so the session stays alive for debugging even if the experiment crashed.
# ─────────────────────────────────────────────────────────────────────────────
finish_experiment() {
    if [[ -d "${EXP_DIR}/rl/final" ]]; then
        echo ""
        echo "╔══════════════════════════════════════════════════════╗"
        echo "║  [${EXP_NAME}] FINISHED SUCCESSFULLY                "
        echo "║  Output → ${EXP_DIR}/rl/final                       "
        echo "╚══════════════════════════════════════════════════════╝"
    else
        echo ""
        echo "╔══════════════════════════════════════════════════════╗"
        echo "║  [${EXP_NAME}] EXITED (check logs for errors)       "
        echo "║  Log dir → ${EXP_LOG_DIR}                           "
        echo "╚══════════════════════════════════════════════════════╝"
    fi
    # Drop to an interactive shell so the tmux session stays open for debugging
    exec bash
}
