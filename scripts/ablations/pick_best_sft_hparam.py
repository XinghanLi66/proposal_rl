#!/usr/bin/env python3
"""
Read sft_hparam_search.json, mark the best entry (lowest mean_loss), and
write a sourceable shell snippet (.best_sft_hparams) alongside it.

Usage: python pick_best_sft_hparam.py <exp_dir>
"""
import json
import sys
from pathlib import Path

exp_dir = Path(sys.argv[1])
result_file = exp_dir / "sft_hparam_search.json"

if not result_file.exists():
    print("ERROR: sft_hparam_search.json not found", file=sys.stderr)
    sys.exit(1)

data = json.loads(result_file.read_text())
results = data.get("results", [])
if not results:
    print("ERROR: no results in sft_hparam_search.json", file=sys.stderr)
    sys.exit(1)

# Lower loss = better
best = min(results, key=lambda r: float(r.get("mean_loss", 9999.0)))
data["best"] = best
result_file.write_text(json.dumps(data, indent=2))

snippet = f"BEST_SFT_LR={best['lr']}\nBEST_SFT_WARMUP={best['warmup']}\n"
(exp_dir / ".best_sft_hparams").write_text(snippet)

print(f"[sft-hparam] Best → lr={best['lr']}  warmup={best['warmup']}  loss={best['mean_loss']:.4f}")
