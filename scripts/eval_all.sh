#!/usr/bin/env bash
# Evaluate all finished experiment checkpoints sequentially.
# Writes eval_results/summary.json + per_example.jsonl inside each checkpoint dir.
#
# Usage:
#   bash scripts/eval_all.sh                   # all exps with rl/final
#   EXPS="exp01 exp02" bash scripts/eval_all.sh  # specific exps
set -euo pipefail

REPO=/newcpfs/lxh/agentic-training/proposal_rl
cd "$REPO"

SPLIT=${SPLIT:-test}
LIMIT=${LIMIT:-500}
BATCH=${BATCH:-4}

# Collect checkpoints: either from EXPS env var or all rl/final dirs
checkpoints=()
if [[ -n "${EXPS:-}" ]]; then
    for prefix in $EXPS; do
        dir=$(ls -d "$REPO/runs/exps/${prefix}_"*/rl/final 2>/dev/null | tail -1 || true)
        if [[ -n "$dir" ]]; then
            checkpoints+=("$dir")
        else
            echo "[eval_all] WARNING: no rl/final found for $prefix"
        fi
    done
else
    while IFS= read -r -d '' d; do
        checkpoints+=("$d")
    done < <(find "$REPO/runs/exps" -path "*/rl/final" -type d -print0 2>/dev/null | sort -z)
fi

if [[ ${#checkpoints[@]} -eq 0 ]]; then
    echo "[eval_all] No checkpoints found."
    exit 1
fi

echo "[eval_all] Evaluating ${#checkpoints[@]} checkpoint(s)  split=$SPLIT  limit=$LIMIT"
echo ""

ok=0; fail=0
for cp in "${checkpoints[@]}"; do
    exp=$(basename "$(dirname "$(dirname "$cp")")")
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " $exp"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if /newcpfs/lxh/miniconda3/bin/conda run -n loongflow_ml --no-capture-output \
        python eval/evaluate.py \
            --config configs/base.yaml \
            --checkpoint "$cp" \
            --split "$SPLIT" \
            --limit "$LIMIT" \
            --batch-size "$BATCH" \
        2>&1; then
        echo "[eval_all] ✓ $exp done"
        ok=$((ok+1))
    else
        echo "[eval_all] ✗ $exp FAILED"
        fail=$((fail+1))
    fi
    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[eval_all] Done: $ok succeeded, $fail failed"
echo ""
echo "Summary table:"
/newcpfs/lxh/miniconda3/bin/conda run -n loongflow_ml python eval/compare.py 2>/dev/null || true
