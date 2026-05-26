#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Shared library for combined SFT+RL ablation scripts (exp09–exp16).
#
# Variables each script must set before sourcing:
#   EXP_NAME        — short identifier, e.g. "exp09_top_k_refs_sft_rl"
#   EXP_LABEL       — human-readable tab name for the dashboard
#   STRATEGY        — prompt_builder.strategy
#   FINETUNE_MODE   — "full" (default) or "lora"
#   REWARD_TYPE     — "prs" or "fas"
#   EXTRA_OVERRIDES — (optional) additional "key=value" pairs
#
# Optional control flags:
#   SFT_CKPT_OVERRIDE   — explicit SFT checkpoint path for RL (used by exp15)
#                         exp15 skips SFT by simply not calling the SFT functions
# ─────────────────────────────────────────────────────────────────────────────

REPO=/newcpfs/lxh/agentic-training/proposal_rl
cd "$REPO"

NGPU=${NGPU:-8}
DASHBOARD_PORT=${DASHBOARD_PORT:-8080}
CONDA_ENV=loongflow_ml

# Proxy credentials for Claude API (CoT synthesis + LLM-based prompt builders)
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-http://10.39.10.241:10001}"
export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-123}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-123}"

# Initialize conda when running in non-login shells
if ! command -v conda &>/dev/null; then
    _CONDA_BASE=/newcpfs/lxh/miniconda3
    if [[ -f "$_CONDA_BASE/etc/profile.d/conda.sh" ]]; then
        source "$_CONDA_BASE/etc/profile.d/conda.sh"
    else
        export PATH="$_CONDA_BASE/bin:$PATH"
    fi
fi

if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
    PY="python"
    TORCHRUN_CMD="torchrun"
else
    PY="conda run -n $CONDA_ENV --no-capture-output python"
    TORCHRUN_CMD="conda run -n $CONDA_ENV --no-capture-output torchrun"
fi
export DISABLE_VERSION_CHECK=1
# torchrl registers a broken vLLM plugin (fp32_overrides) that imports vllm.worker.worker
# which was removed in vLLM >=0.7. Disable external plugins to prevent vLLMHttpServer crashes.
export VLLM_PLUGINS=""
# Ray workers all import sympy→gmpy2 concurrently. On glibc 2.28 this causes:
#   _dl_allocate_tls_init: Assertion `listp != NULL` failed!
# SYMPY_GROUND_TYPES=python skips the gmpy2 dlopen entirely.
export SYMPY_GROUND_TYPES="python"
# ProcessGroupNCCL watchdog default timeout (600s) is too short for large SFT steps
# that hold the GIL during CudaEventDestroy. Extend to 30 min to suppress false-positive
# SIGABRT kills on rank 2 after the first optimizer step.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}
# On glibc 2.28, loading libcuda after pthread_create causes TLS slot allocation
# failures (_dl_allocate_tls_init: Assertion `listp != NULL` failed!) when many
# Ray workers dlopen CUDA concurrently. Force-loading libcuda at process start
# via LD_PRELOAD ensures TLS descriptors are set up before any threads spawn.
export LD_PRELOAD="${LD_PRELOAD:+${LD_PRELOAD}:}/usr/lib/x86_64-linux-gnu/libcuda.so.1"
# vLLM v1 multiprocessing executor triggers intermittent "none_dealloc: deallocating None"
# Python GC refcount corruption during collective_rpc weight-sync (update_weights step).
# Setting VLLM_ENABLE_V1_MULTIPROCESSING=0 switches to the uniproc executor, running
# the GPU worker in-process instead of a subprocess, avoiding cross-process IPC teardown.
export VLLM_ENABLE_V1_MULTIPROCESSING=0

_free_port() {
    $PY -c "import socket; s=socket.socket(); s.bind(('',0)); \
p=s.getsockname()[1]; s.close(); print(p)"
}

# ─── Hparam grids ────────────────────────────────────────────────────────────

# SFT 3×3: learning rate × warmup ratio
SFT_HP_LRS=(1e-5 2e-5 5e-5)
SFT_HP_WARMUPS=(0.03 0.05 0.10)
SFT_HP_LIMIT=8192        # training examples per mini-run

# RL 3×3: learning rate × KL coefficient
RL_HP_LRS=(1e-6 2e-6 5e-6)
RL_HP_KLS=(0.01 0.02 0.05)
RL_HP_LIMIT=512          # training examples per mini-run

# ─────────────────────────────────────────────────────────────────────────────
setup_combined_experiment() {
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

    local overrides=(
        "prompt_builder.strategy=${STRATEGY}"
        "sft.finetune_mode=${FINETUNE_MODE}"
        "sft.output_dir=${EXP_DIR}/sft"
        "sft.dataset_file=${EXP_DIR}/train_cot.jsonl"
        "rl.finetune_mode=${FINETUNE_MODE}"
        "rl.reward_type=${REWARD_TYPE}"
        "rl.output_dir=${EXP_DIR}/rl"
        "rl.n_gpus=${NGPU}"
    )
    # verl uses FSDP (not DeepSpeed) — no DS config needed for either SFT or RL.
    # gradient_accumulation is expressed via FSDP micro-batch logic; set to 1 for
    # consistency with how verl computes effective batch size.
    overrides+=(
        "sft.deepspeed_config="
        "sft.gradient_accumulation_steps=1"
        "rl.deepspeed_config="
        "rl.gradient_accumulation_steps=1"
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
    payload=$($PY -c "
import json
print(json.dumps({
    'exp_id':   '${EXP_ID}',
    'name':     '${EXP_LABEL}',
    'runs_dir': '${EXP_DIR}',
    'config': {
        'prompt_builder': {'strategy': '${STRATEGY}'},
        'sft': {'finetune_mode': '${FINETUNE_MODE}'},
        'rl':  {'finetune_mode': '${FINETUNE_MODE}', 'reward_type': '${REWARD_TYPE}'},
    }
}))
")
    curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "http://localhost:${DASHBOARD_PORT}/api/experiments" \
        --max-time 5 > /dev/null 2>&1 \
        && echo "[${EXP_NAME}] Dashboard tab registered." \
        || echo "[${EXP_NAME}] Dashboard not reachable — continuing without registration."
}

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
    trap 'kill "$_REFRESH_PID" 2>/dev/null || true' EXIT
}

# ─────────────────────────────────────────────────────────────────────────────
run_cot_synthesis() {
    local log="${EXP_LOG_DIR}/cot_synthesis_${EXP_ID}.log"
    local out="${EXP_DIR}/train_cot.jsonl"
    echo "[${EXP_NAME}] === CoT synthesis (strategy=${STRATEGY}) ==="
    echo "[${EXP_NAME}] Output → ${out}"

    # Auto-detect the most COMPLETE CoT from a previous run of this experiment
    # (same EXP_NAME prefix, different timestamp).  Pick by highest line count
    # so a partial file from a killed run is never preferred over a complete one.
    local _prev_cot _best_n=0 _f _n
    for _f in "${REPO}/runs/exps/${EXP_NAME}_"*/train_cot.jsonl; do
        [[ "$_f" == "$out" ]] && continue
        [[ -s "$_f" ]] || continue
        _n=$(wc -l < "$_f")
        if (( _n > _best_n )); then
            _best_n=$_n
            _prev_cot=$_f
        fi
    done
    if [[ -n "${_prev_cot:-}" ]]; then
        echo "[${EXP_NAME}] Seeding CoT cache from previous run (${_best_n} records): ${_prev_cot}"
        cp "$_prev_cot" "$out"
    fi

    $PY data/synthesize_cot.py \
        --config "$BASE_CFG" \
        --input  "$REPO/runs/dataset/train.jsonl" \
        --output "$out" \
        --strategy "${STRATEGY}" \
        2>&1 | tee "$log" \
        && echo "[${EXP_NAME}] CoT synthesis done." \
        || { echo "[${EXP_NAME}] ERROR: CoT synthesis failed — check $log"; exit 1; }
}

# ─────────────────────────────────────────────────────────────────────────────
# SFT hparam search helpers
# ─────────────────────────────────────────────────────────────────────────────
_run_one_sft_mini() {
    local lr=$1 warmup=$2
    local tag="lr${lr}_w${warmup}"
    local mini_out="${EXP_DIR}/sft_hparam_${tag}"
    local mini_cfg="${EXP_DIR}/config_sft_mini_${tag}.yaml"
    local mini_log="${EXP_LOG_DIR}/sft_${EXP_ID}_mini_${tag}.log"

    echo "[${EXP_NAME}] SFT mini: lr=${lr}  warmup=${warmup}"

    $PY scripts/make_config.py \
        --base "$BASE_CFG" \
        --out  "$mini_cfg" \
        --set  "sft.learning_rate=${lr}" \
               "sft.warmup_ratio=${warmup}" \
               "sft.num_train_epochs=1" \
               "sft.save_steps=9999" \
               "sft.logging_steps=5" \
               "sft.output_dir=${mini_out}" \
               "sft.limit=${SFT_HP_LIMIT}" \
               "sft.gradient_accumulation_steps=1" \
               "sft.deepspeed_config="

    local port
    port=$(_free_port)
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $TORCHRUN_CMD \
        --nproc_per_node "$NGPU" \
        --master_port "$port" \
        train/sft.py --config "$mini_cfg" \
        2>&1 | tee "$mini_log" || \
        echo "[${EXP_NAME}] WARNING: SFT mini lr=${lr} warmup=${warmup} exited non-zero"

    local mean_loss
    mean_loss=$($PY scripts/ablations/extract_sft_metric.py "$mini_log" 20)
    echo "[${EXP_NAME}] SFT lr=${lr}  warmup=${warmup}  mean_loss=${mean_loss}"

    $PY - <<PYEOF
import json
from pathlib import Path
f = Path("${EXP_DIR}/sft_hparam_search.json")
data = {"results": [], "best": None}
if f.exists():
    try: data = json.loads(f.read_text())
    except: pass
data["results"].append({
    "lr": "${lr}", "warmup": "${warmup}",
    "mean_loss": float("${mean_loss}")
})
f.write_text(json.dumps(data, indent=2))
PYEOF
}

run_sft_hparam_search() {
    echo "[${EXP_NAME}] === SFT hparam search (${#SFT_HP_LRS[@]}×${#SFT_HP_WARMUPS[@]} grid) ==="
    for lr in "${SFT_HP_LRS[@]}"; do
        for warmup in "${SFT_HP_WARMUPS[@]}"; do
            _run_one_sft_mini "$lr" "$warmup"
        done
    done
    $PY scripts/ablations/pick_best_sft_hparam.py "$EXP_DIR"
    source "${EXP_DIR}/.best_sft_hparams"
    echo "[${EXP_NAME}] Best SFT hparams: LR=${BEST_SFT_LR}  warmup=${BEST_SFT_WARMUP}"
}

run_sft_full_training() {
    local full_cfg="${EXP_DIR}/config_sft_full.yaml"
    local log="${EXP_LOG_DIR}/sft_${EXP_ID}_full.log"

    echo "[${EXP_NAME}] === Full SFT: LR=${BEST_SFT_LR}  warmup=${BEST_SFT_WARMUP} ==="

    $PY scripts/make_config.py \
        --base "$BASE_CFG" \
        --out  "$full_cfg" \
        --set  "sft.learning_rate=${BEST_SFT_LR}" \
               "sft.warmup_ratio=${BEST_SFT_WARMUP}"

    local port
    port=$(_free_port)
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $TORCHRUN_CMD \
        --nproc_per_node "$NGPU" \
        --master_port "$port" \
        train/sft.py --config "$full_cfg" --resume \
        2>&1 | tee -a "$log" \
        && echo "[${EXP_NAME}] SFT done → ${EXP_DIR}/sft/final" \
        || { echo "[${EXP_NAME}] ERROR: SFT full training failed — check $log"; exit 1; }
}

# ─────────────────────────────────────────────────────────────────────────────
# RL hparam search helpers (3×3)
# ─────────────────────────────────────────────────────────────────────────────
_run_one_rl_mini() {
    local lr=$1 kl=$2
    local tag="lr${lr}_kl${kl}"
    local mini_out="${EXP_DIR}/rl_hparam_${tag}"
    local mini_cfg="${EXP_DIR}/config_rl_mini_${tag}.yaml"
    local mini_log="${EXP_LOG_DIR}/rl_${EXP_ID}_mini_${tag}.log"

    echo "[${EXP_NAME}] RL mini: lr=${lr}  kl=${kl}"

    $PY scripts/make_config.py \
        --base "$BASE_CFG" \
        --out  "$mini_cfg" \
        --set  "rl.learning_rate=${lr}" \
               "rl.kl_coeff=${kl}" \
               "rl.sft_checkpoint=${_RL_SFT_CKPT}" \
               "rl.num_train_epochs=1" \
               "rl.save_steps=9999" \
               "rl.logging_steps=5" \
               "rl.output_dir=${mini_out}" \
               "rl.limit=${RL_HP_LIMIT}"

    # verl RL uses Ray (single-process launcher — Ray spawns workers internally)
    # expandable_segments is incompatible with vLLM memory pool (pytorch#147851) — must not be set.
    NGPU="$NGPU" PYTORCH_CUDA_ALLOC_CONF="" $PY train/rl.py --config "$mini_cfg" \
        2>&1 | tee "$mini_log" || \
        echo "[${EXP_NAME}] WARNING: RL mini lr=${lr} kl=${kl} exited non-zero"

    local mean_reward
    mean_reward=$($PY scripts/ablations/extract_reward.py "$mini_log" 10)
    echo "[${EXP_NAME}] RL lr=${lr}  kl=${kl}  mean_reward=${mean_reward}"

    $PY - <<PYEOF
import json
from pathlib import Path
f = Path("${EXP_DIR}/rl_hparam_search.json")
data = {"results": [], "best": None}
if f.exists():
    try: data = json.loads(f.read_text())
    except: pass
data["results"].append({
    "lr": "${lr}", "kl": "${kl}",
    "mean_reward": float("${mean_reward}")
})
f.write_text(json.dumps(data, indent=2))
PYEOF
}

run_rl_hparam_search() {
    # Resolve SFT checkpoint for RL
    _RL_SFT_CKPT="${SFT_CKPT_OVERRIDE:-${EXP_DIR}/sft/final}"
    if [[ ! -d "$_RL_SFT_CKPT" ]]; then
        echo "[${EXP_NAME}] ERROR: SFT checkpoint not found at ${_RL_SFT_CKPT}"
        exit 1
    fi
    echo "[${EXP_NAME}] RL using SFT ckpt: ${_RL_SFT_CKPT}"

    echo "[${EXP_NAME}] === RL hparam search (${#RL_HP_LRS[@]}×${#RL_HP_KLS[@]} grid) ==="
    for lr in "${RL_HP_LRS[@]}"; do
        for kl in "${RL_HP_KLS[@]}"; do
            _run_one_rl_mini "$lr" "$kl"
        done
    done

    # Reuse existing pick_best_hparam.py (picks by max mean_reward in hparam_search.json)
    # But we stored results in rl_hparam_search.json, so symlink for the picker
    cp "${EXP_DIR}/rl_hparam_search.json" "${EXP_DIR}/hparam_search.json"
    $PY scripts/ablations/pick_best_hparam.py "$EXP_DIR"
    source "${EXP_DIR}/.best_hparams"
    echo "[${EXP_NAME}] Best RL hparams: LR=${BEST_LR}  KL=${BEST_KL}"
}

run_rl_full_training() {
    local full_cfg="${EXP_DIR}/config_rl_full.yaml"
    local log="${EXP_LOG_DIR}/rl_${EXP_ID}_full.log"

    echo "[${EXP_NAME}] === Full RL: LR=${BEST_LR}  KL=${BEST_KL} ==="

    $PY scripts/make_config.py \
        --base "$BASE_CFG" \
        --out  "$full_cfg" \
        --set  "rl.learning_rate=${BEST_LR}" \
               "rl.kl_coeff=${BEST_KL}" \
               "rl.sft_checkpoint=${_RL_SFT_CKPT}"

    # verl RL uses Ray (single-process launcher)
    # expandable_segments is incompatible with vLLM memory pool (pytorch#147851) — must not be set.
    NGPU="$NGPU" PYTORCH_CUDA_ALLOC_CONF="" \
    $PY train/rl.py --config "$full_cfg" --resume \
        2>&1 | tee "$log" \
        && echo "[${EXP_NAME}] RL done → ${EXP_DIR}/rl/final" \
        || { echo "[${EXP_NAME}] ERROR: RL full training failed — check $log"; exit 1; }
}

# ─────────────────────────────────────────────────────────────────────────────
finish_experiment() {
    local sft_ok=false rl_ok=false
    [[ -d "${EXP_DIR}/sft/final" ]] && sft_ok=true
    [[ -d "${EXP_DIR}/rl/final"  ]] && rl_ok=true

    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    if $rl_ok; then
        echo "║  [${EXP_NAME}] FINISHED SUCCESSFULLY                "
    else
        echo "║  [${EXP_NAME}] EXITED (check logs for errors)       "
    fi
    echo "║  SFT → ${EXP_DIR}/sft/final  (ok=${sft_ok})         "
    echo "║  RL  → ${EXP_DIR}/rl/final   (ok=${rl_ok})          "
    echo "╚══════════════════════════════════════════════════════╝"
    exec bash
}
