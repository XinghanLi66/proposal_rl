#!/usr/bin/env bash
# Resume pipeline from Step 3 (train.jsonl already built with 6041 examples)
set -euo pipefail

REPO=/newcpfs/lxh/agentic-training/proposal_rl
cd "$REPO"

export ANTHROPIC_BASE_URL=http://10.39.10.241:10001
export ANTHROPIC_AUTH_TOKEN=123
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
hr()   { echo ""; echo "══════════════════════════════════════════════════════"; echo "  $*"; echo "══════════════════════════════════════════════════════"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

mkdir -p runs/logs

# Verify train.jsonl is ready
TRAIN_N=$(wc -l < runs/dataset/train.jsonl)
log "train.jsonl ready: $TRAIN_N examples"
[[ "$TRAIN_N" -gt 1000 ]] || die "train.jsonl too small ($TRAIN_N lines)"

# ── Step A: Build val/test eval datasets (background, needs ~9 min full scan) ──
hr "Step A: Build val/test eval datasets (background)"
log "Launching val/test build in background..."
(
  SPLITS="val test" bash scripts/02_build_dataset.sh \
    > runs/logs/build_dataset_valtest_$(date +%Y%m%d_%H%M%S).log 2>&1
  log "[bg] val/test dataset build DONE"
) &
VALTEST_BUILD_PID=$!

# ── Step B: Synthesize CoT proposals via local Claude service ──────────────────
hr "Step B: Synthesize CoT proposals (local Claude)"
# train_cot.jsonl may be a stale copy of train.jsonl (no cot_proposal fields).
# load_done() reads arxiv_ids from it → all IDs appear "done" → synthesis skips everything.
# Delete it so synthesis starts clean.
if [[ -f runs/dataset/train_cot.jsonl ]]; then
  EXISTING_COT=$(python3 -c "
import json
count = 0
with open('runs/dataset/train_cot.jsonl') as f:
    for line in f:
        try:
            d = json.loads(line)
            if d.get('cot_proposal'):
                count += 1
        except Exception:
            pass
print(count)
" 2>/dev/null || echo 0)
  if [[ "$EXISTING_COT" -gt 0 ]]; then
    log "Found $EXISTING_COT already-synthesized records in train_cot.jsonl — will resume from there"
  else
    log "train_cot.jsonl exists but has no cot_proposal fields — removing stale copy"
    rm -f runs/dataset/train_cot.jsonl
  fi
fi
log "Starting CoT synthesis for $TRAIN_N train examples..."
bash scripts/03_synthesize_cot.sh

COT_N=$(wc -l < runs/dataset/train_cot.jsonl 2>/dev/null || echo 0)
log "train_cot.jsonl: $COT_N examples"
[[ "$COT_N" -gt 1000 ]] || die "train_cot.jsonl too small ($COT_N lines) — CoT synthesis failed?"

# ── Step C: Wait for val/test build, then build indexes ───────────────────────
hr "Step C: Wait for val/test datasets, then build FAS indexes"
log "Waiting for background val/test build (PID=$VALTEST_BUILD_PID)..."
wait "$VALTEST_BUILD_PID" || log "WARNING: val/test build process ended with error"

VAL_N=$(wc -l < runs/dataset/val.jsonl  2>/dev/null || echo 0)
TEST_N=$(wc -l < runs/dataset/test.jsonl 2>/dev/null || echo 0)
log "val.jsonl: $VAL_N | test.jsonl: $TEST_N"

log "Building FAS indexes from local arxiv metadata (ALL papers in val/test months)..."
bash scripts/04_build_index.sh

# ── Step D: SFT ───────────────────────────────────────────────────────────────
hr "Step D: SFT training (8 GPU)"
# Start expanding train refs in background while GPU trains
(
  SPLIT=train LIMIT=0 bash scripts/01_fetch_refs.sh \
    > runs/logs/fetch_train_full_$(date +%Y%m%d_%H%M%S).log 2>&1
  log "[bg] Full train fetch done"
) &
FETCH_BG_PID=$!

bash scripts/05_sft.sh

# ── Step D eval ───────────────────────────────────────────────────────────────
hr "Evaluate SFT checkpoint"
CHECKPOINT=runs/sft/final SPLIT=test LIMIT=500 bash scripts/07_evaluate.sh
log "SFT results:"
python3 -c "
import json
try:
    d = json.load(open('runs/sft/final/eval_results/summary.json'))
    print(f'  FAS={d[\"FAS\"]}  recall@50={d[\"recall_at_k\"]}  mean_sim={d[\"mean_similarity\"]}  format={d[\"format_score\"]}')
except Exception as e:
    print(f'  (could not read: {e})')
"

# ── Step E: GRPO ──────────────────────────────────────────────────────────────
hr "Step E: Rebuild train data + GRPO training"
kill "$FETCH_BG_PID" 2>/dev/null || true

SPLITS="train" bash scripts/02_build_dataset.sh
# Note: skip synthesize_cot.sh here — GRPO uses train.jsonl (not train_cot.jsonl),
# and re-running CoT synthesis on expanded data would delay GRPO by many hours.

bash scripts/06_grpo.sh

# ── Step E eval ───────────────────────────────────────────────────────────────
hr "Evaluate GRPO checkpoint"
CHECKPOINT=runs/grpo/final SPLIT=test LIMIT=500 bash scripts/07_evaluate.sh

hr "PIPELINE COMPLETE"
echo ""
echo "=== SFT ===" && python3 -c "
import json
d = json.load(open('runs/sft/final/eval_results/summary.json'))
[print(f'  {k}: {v}') for k, v in d.items()]
" 2>/dev/null || true
echo ""
echo "=== GRPO ===" && python3 -c "
import json
d = json.load(open('runs/grpo/final/eval_results/summary.json'))
[print(f'  {k}: {v}') for k, v in d.items()]
" 2>/dev/null || true
