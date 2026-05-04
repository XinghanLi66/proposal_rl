#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Shared library for SFT ablation scripts.
#
# Variables each script must set before sourcing:
#   STRATEGY      — prompt_builder.strategy (e.g. top_k_refs, full_refs, ...)
#   FINETUNE_MODE — "full" (default) or "lora"
#   NGPU          — number of GPUs (default 8)
# ─────────────────────────────────────────────────────────────────────────────

# Output dir encodes both strategy and finetune mode so all 6 checkpoints
# can coexist under runs/sft/:
#   full-FT  → runs/sft/{strategy}/final
#   LoRA     → runs/sft/{strategy}_lora/final
if [[ "${FINETUNE_MODE:-full}" == "lora" ]]; then
    _SFT_SUBDIR="${STRATEGY}_lora"
else
    _SFT_SUBDIR="${STRATEGY}"
fi
SFT_OUTPUT_DIR="${REPO}/runs/sft/${_SFT_SUBDIR}"

# Initialize conda in non-login shells
if ! command -v conda &>/dev/null; then
    _CONDA_BASE=/newcpfs/lxh/miniconda3
    if [[ -f "$_CONDA_BASE/etc/profile.d/conda.sh" ]]; then
        source "$_CONDA_BASE/etc/profile.d/conda.sh"
    else
        export PATH="$_CONDA_BASE/bin:$PATH"
    fi
fi

CONDA_ENV=loongflow_ml
if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
    PY="python"
    TORCHRUN_CMD="torchrun"
else
    PY="conda run -n $CONDA_ENV --no-capture-output python"
    TORCHRUN_CMD="conda run -n $CONDA_ENV --no-capture-output torchrun"
fi
export DISABLE_VERSION_CHECK=1

_free_port() {
    python3 -c "import socket; s=socket.socket(); s.bind(('',0)); \
p=s.getsockname()[1]; s.close(); print(p)"
}

# ─────────────────────────────────────────────────────────────────────────────
run_sft() {
    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    local log_dir="${REPO}/runs/logs"
    local log_file="${log_dir}/sft_${_SFT_SUBDIR}_${ts}.log"
    local cfg_file="/tmp/sft_cfg_${_SFT_SUBDIR}.yaml"
    mkdir -p "$log_dir"

    echo "[sft:${_SFT_SUBDIR}] strategy=${STRATEGY}  finetune_mode=${FINETUNE_MODE}  ngpu=${NGPU}"
    echo "[sft:${_SFT_SUBDIR}] output → ${SFT_OUTPUT_DIR}/final"
    echo "[sft:${_SFT_SUBDIR}] log    → ${log_file}"

    # Build a config with strategy and output_dir baked in
    $PY scripts/make_config.py \
        --base configs/base.yaml \
        --out  "$cfg_file" \
        --set  "prompt_builder.strategy=${STRATEGY}" \
               "sft.finetune_mode=${FINETUNE_MODE}" \
               "sft.output_dir=${SFT_OUTPUT_DIR}"

    local port
    port=$(_free_port)
    $TORCHRUN_CMD \
        --nproc_per_node "$NGPU" \
        --master_port "$port" \
        train/sft.py --config "$cfg_file" \
        2>&1 | tee "$log_file" \
        && echo "[sft:${_SFT_SUBDIR}] Done → ${SFT_OUTPUT_DIR}/final" \
        || echo "[sft:${_SFT_SUBDIR}] ERROR — check ${log_file}"
}
