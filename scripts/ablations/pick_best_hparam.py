#!/usr/bin/env python3
"""
Read hparam_search.json, mark the best entry, and write a sourceable shell
snippet (.best_hparams) alongside it.

Usage: python pick_best_hparam.py <exp_dir>
"""
import json
import sys
from pathlib import Path

exp_dir = Path(sys.argv[1])
result_file = exp_dir / "hparam_search.json"

if not result_file.exists():
    print("ERROR: hparam_search.json not found", file=sys.stderr)
    sys.exit(1)

data = json.loads(result_file.read_text())
results = data.get("results", [])
if not results:
    print("ERROR: no results in hparam_search.json", file=sys.stderr)
    sys.exit(1)

best = max(results, key=lambda r: float(r.get("mean_reward", 0)))
data["best"] = best
result_file.write_text(json.dumps(data, indent=2))

snippet = f"BEST_LR={best['lr']}\nBEST_KL={best['kl']}\n"
(exp_dir / ".best_hparams").write_text(snippet)

print(f"[hparam] Best → lr={best['lr']}  kl={best['kl']}  reward={best['mean_reward']:.4f}")
