#!/usr/bin/env bash
# =============================================================================
# proposal_rl — master pipeline launcher
#
# Runs the full pipeline in tmux window 1 (as requested):
#   Pane 0 (main):      pipeline steps, sequential
#   Pane 1 (dashboard): rich live dashboard (auto-refreshes)
#   Pane 2 (fetch bg):  background ref fetching while GPU trains
#
# Usage:
#   bash run.sh
#   S2_API_KEY=...  bash run.sh   # optional, speeds up ref fetching
#
# To resume after interruption:
#   RESUME=1 bash run.sh
#
# To run just one stage:
#   bash scripts/05_sft.sh
# =============================================================================
set -euo pipefail

REPO=/newcpfs/lxh/agentic-training/proposal_rl
cd "$REPO"

# Local Claude service
export ANTHROPIC_BASE_URL=http://10.39.10.241:10001
export ANTHROPIC_AUTH_TOKEN=123
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1

TMUX_SESSION=${TMUX_SESSION:-1}   # attach to existing tmux window 1
RESUME=${RESUME:-0}

# ---- Check tmux --------------------------------------------------------------
if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "Creating tmux session '$TMUX_SESSION'..."
  tmux new-session -d -s "$TMUX_SESSION" -x 220 -y 50
fi

# Helper: run a command in the current (main) pane and wait for it
run_step() {
  local name="$1"; shift
  echo ""
  echo "╔══════════════════════════════════════════════════════╗"
  echo "║  STAGE: $name"
  echo "╚══════════════════════════════════════════════════════╝"
  "$@"
}

# ---- Create dashboard pane (right side) ------------------------------------
# Only create if we don't already have split panes
PANE_COUNT=$(tmux list-panes -t "$TMUX_SESSION" 2>/dev/null | wc -l || echo 1)
if [[ "$PANE_COUNT" -lt 2 ]]; then
  tmux split-window -h -t "$TMUX_SESSION" -p 35
  tmux send-keys -t "$TMUX_SESSION".1 \
    "cd $REPO && conda run -n loongflow_ml python monitor/dashboard.py --runs-dir runs" Enter
fi

# ---- STAGE 0: Ensure model is available ------------------------------------
run_step "Model setup" bash -c "
  MODEL_PATH=/newcpfs/user/sujianghao/model/Qwen/Qwen2.5-7B-Instruct
  if [ -d \"\$MODEL_PATH\" ] && [ -f \"\$MODEL_PATH/config.json\" ]; then
    echo 'Found Qwen2.5-7B-Instruct at' \"\$MODEL_PATH\"
    # Patch config to point at local path
    sed -i \"s|model_name_or_path:.*|model_name_or_path: \$MODEL_PATH|\" configs/base.yaml
  else
    echo 'Qwen2.5-7B-Instruct not found locally — will download from HuggingFace during training.'
  fi
"

# ---- STAGE 1: Fetch initial training refs (8K papers) ----------------------
run_step "Fetch refs (initial 8K)" \
  bash -c "SPLIT=train LIMIT=8000 bash scripts/01_fetch_refs.sh"

# Fetch val refs in parallel (smaller, faster)
run_step "Fetch refs (val)" \
  bash -c "SPLIT=val LIMIT=0 bash scripts/01_fetch_refs.sh &
           SPLIT=test LIMIT=0 bash scripts/01_fetch_refs.sh &
           wait"

# ---- STAGE 2: Build base dataset -------------------------------------------
run_step "Build dataset" \
  bash scripts/02_build_dataset.sh

# ---- STAGE 3: Synthesize CoT proposals (Claude API) -----------------------
run_step "Synthesize CoT (Claude API)" \
  bash scripts/03_synthesize_cot.sh

# ---- STAGE 4: Build embedding index for val set ---------------------------
run_step "Build embedding index" \
  bash scripts/04_build_index.sh

# ---- STAGE 5: SFT training (8 GPUs) ----------------------------------------
# Start background fetch of the remaining train refs while GPU trains
echo ""
echo "► Launching background ref fetch (full train split) in pane 2..."
tmux split-window -v -t "$TMUX_SESSION".0 -p 25 2>/dev/null || true
tmux send-keys -t "$TMUX_SESSION".2 \
  "cd $REPO && SPLIT=train LIMIT=0 bash scripts/01_fetch_refs.sh && echo '[bg] Full fetch done!'" Enter

# Now run SFT
${RESUME:+export RESUME=1}
run_step "SFT training (8 GPU)" \
  bash scripts/05_sft.sh

# Evaluate SFT checkpoint
run_step "Evaluate SFT checkpoint" \
  bash -c "CHECKPOINT=runs/sft/final SPLIT=test LIMIT=500 bash scripts/07_evaluate.sh"

# ---- STAGE 6: GRPO training ------------------------------------------------
# Wait for background fetch to write more data, then rebuild dataset
echo ""
echo "► Rebuilding dataset with expanded refs..."
run_step "Rebuild dataset (expanded)" \
  bash scripts/02_build_dataset.sh

# Re-synthesize CoT for new examples only (resumable)
run_step "Synthesize CoT (expanded)" \
  bash scripts/03_synthesize_cot.sh

run_step "GRPO training (8 GPU)" \
  bash scripts/06_grpo.sh

# ---- STAGE 7: Final evaluation ---------------------------------------------
run_step "Final evaluation (GRPO)" \
  bash -c "CHECKPOINT=runs/grpo/final SPLIT=test LIMIT=500 bash scripts/07_evaluate.sh"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Pipeline complete! Results in runs/grpo/final/eval_results/"
echo "╚══════════════════════════════════════════════════════════╝"

# Print final FAS scores
echo ""
echo "=== SFT FAS ==="
cat runs/sft/final/eval_results/summary.json 2>/dev/null || echo "(not found)"
echo ""
echo "=== GRPO FAS ==="
cat runs/grpo/final/eval_results/summary.json 2>/dev/null || echo "(not found)"
