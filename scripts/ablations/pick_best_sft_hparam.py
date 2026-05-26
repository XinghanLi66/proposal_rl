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

# Filter out invalid runs: crash sentinel (≥9999), collapsed (≤0.01), or exploded (≥10)
# A healthy SFT run should produce mean cross-entropy loss in the 0.1–8 range
valid = [r for r in results if 0.01 < float(r.get("mean_loss", 9999.0)) < 10.0]
if not valid:
    print("WARNING: no valid results (0.01<loss<10); falling back to all results below 9999", file=sys.stderr)
    valid = [r for r in results if float(r.get("mean_loss", 9999.0)) < 9999.0]
if not valid:
    print("WARNING: all results are crash sentinels; using all results", file=sys.stderr)
    valid = results

# Lower loss = better
best = min(valid, key=lambda r: float(r.get("mean_loss", 9999.0)))
data["best"] = best
result_file.write_text(json.dumps(data, indent=2))

snippet = f"BEST_SFT_LR={best['lr']}\nBEST_SFT_WARMUP={best['warmup']}\n"
(exp_dir / ".best_sft_hparams").write_text(snippet)

print(f"[sft-hparam] Best → lr={best['lr']}  warmup={best['warmup']}  loss={best['mean_loss']:.4f}")
