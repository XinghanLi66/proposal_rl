#!/usr/bin/env bash
set -euo pipefail
REPO=/newcpfs/lxh/agentic-training/proposal_rl
cd "$REPO"

# Local Claude service (same as run_claude.sh)
export ANTHROPIC_BASE_URL=http://10.39.10.241:10001
export ANTHROPIC_AUTH_TOKEN=123
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1

log() { echo "[$(date '+%H:%M:%S')] $*"; }
hr()  { echo ""; echo "══════════════════════════════════════════════════════"; echo "  $*"; echo "══════════════════════════════════════════════════════"; }

# ── Step 0: Download / locate model ──────────────────────────────────────────
hr "Step 0: Model setup"
conda run -n loongflow_ml python scripts/00_download_model.py
MODEL_PATH=$(cat runs/model_path.txt)
log "Model ready: $MODEL_PATH"
sed -i "s|model_name_or_path:.*|model_name_or_path: \"$MODEL_PATH\"|" configs/base.yaml

# ── Step 1: Fetch initial 8K training refs ────────────────────────────────────
hr "Step 1: Fetch initial 8K train refs"
SPLIT=train LIMIT=8000 bash scripts/01_fetch_refs.sh

# ── Step 1b: Fetch val+test refs in background ────────────────────────────────
log "Step 1b: Fetching val+test refs (background)"
( SPLIT=val  LIMIT=0 bash scripts/01_fetch_refs.sh > runs/logs/fetch_val_bg.log  2>&1 ; log "[bg] val fetch done"  ) &
( SPLIT=test LIMIT=0 bash scripts/01_fetch_refs.sh > runs/logs/fetch_test_bg.log 2>&1 ; log "[bg] test fetch done" ) &

# ── Step 2: Build base dataset (train only) ───────────────────────────────────
hr "Step 2: Build train dataset"
SPLITS="train" bash scripts/02_build_dataset.sh

# ── Step 3: Synthesize CoT proposals via Claude API ───────────────────────────
hr "Step 3: Synthesize CoT proposals"
bash scripts/03_synthesize_cot.sh

# ── Step 4: Build embedding index (wait for val+test fetch) ──────────────────
hr "Step 4: Build embedding index"
log "Waiting for val/test background fetch..."
wait
SPLITS="val test" bash scripts/02_build_dataset.sh
bash scripts/04_build_index.sh

# ── Step 5: SFT ──────────────────────────────────────────────────────────────
hr "Step 5: SFT training (8 GPU)"
( SPLIT=train LIMIT=0 bash scripts/01_fetch_refs.sh > runs/logs/fetch_train_full_bg.log 2>&1 ; log "[bg] full train fetch done" ) &
FETCH_BG_PID=$!

bash scripts/05_sft.sh

hr "Evaluate SFT checkpoint"
CHECKPOINT=runs/sft/final SPLIT=test LIMIT=500 bash scripts/07_evaluate.sh
log "SFT FAS:"
python3 -c "
import json
d = json.load(open('runs/sft/final/eval_results/summary.json'))
print(f'  FAS={d[\"FAS\"]}  recall@50={d[\"recall_at_k\"]}  format={d[\"format_score\"]}')
" 2>/dev/null || true

# ── Step 6: Expand data + GRPO ────────────────────────────────────────────────
hr "Step 6: Rebuild with expanded data + GRPO"
kill "$FETCH_BG_PID" 2>/dev/null || true

SPLITS="train" bash scripts/02_build_dataset.sh
bash scripts/03_synthesize_cot.sh   # resume-safe, skips already-done IDs

bash scripts/06_grpo.sh

hr "Evaluate GRPO checkpoint"
CHECKPOINT=runs/grpo/final SPLIT=test LIMIT=500 bash scripts/07_evaluate.sh

hr "PIPELINE COMPLETE"
echo ""
echo "=== SFT ===" && python3 -c "import json; d=json.load(open('runs/sft/final/eval_results/summary.json')); [print(f'  {k}: {v}') for k,v in d.items()]" 2>/dev/null || true
echo ""
echo "=== GRPO ===" && python3 -c "import json; d=json.load(open('runs/grpo/final/eval_results/summary.json')); [print(f'  {k}: {v}') for k,v in d.items()]" 2>/dev/null || true
