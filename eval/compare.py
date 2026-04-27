#!/usr/bin/env python3
"""
Print a comparison table of eval results across all experiment checkpoints.

Usage:
    python eval/compare.py                          # all exps with eval_results
    python eval/compare.py --exps exp01 exp02 exp07 # filter by prefix
    python eval/compare.py --sort FAS               # sort by metric
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


METRICS = ["FAS", "recall_at_k", "PRS", "format_score", "leakage_score_mean"]
HEADERS  = ["exp",        "FAS",   "recall@K", "PRS",   "fmt",   "leakage", "n",    "strategy", "checkpoint"]


def load_summaries(runs_dir: Path, exp_prefixes: list[str]) -> list[dict]:
    rows = []
    for summary_path in sorted(runs_dir.glob("exps/*/rl/final/eval_results/summary.json")):
        exp_dir  = summary_path.parents[3]   # exps/EXP_ID
        exp_id   = exp_dir.name

        if exp_prefixes and not any(exp_id.startswith(p) for p in exp_prefixes):
            continue

        try:
            s = json.loads(summary_path.read_text())
        except Exception:
            continue

        # Short label: exp01_baseline_20260425 → "exp01_baseline"
        parts = exp_id.rsplit("_", 2)
        label = parts[0] if len(parts) == 3 else exp_id

        rows.append({
            "exp":       label,
            "FAS":       s.get("FAS"),
            "recall@K":  s.get("recall_at_k"),
            "PRS":       s.get("PRS"),
            "fmt":       s.get("format_score"),
            "leakage":   s.get("leakage_score_mean"),
            "n":         s.get("n_examples"),
            "strategy":  s.get("fas_strategy", "embedding"),
            "checkpoint": str(summary_path.parents[1]),   # .../rl/final
        })
    return rows


def fmt_cell(v) -> str:
    if v is None:
        return "  –  "
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def print_table(rows: list[dict], sort_by: str | None) -> None:
    if not rows:
        print("No eval results found.")
        return

    if sort_by and sort_by in rows[0]:
        rows = sorted(rows, key=lambda r: (r[sort_by] is None, -(r[sort_by] or 0)))

    col_keys = ["exp", "FAS", "recall@K", "PRS", "fmt", "leakage", "n", "strategy"]
    col_w    = {k: max(len(k), max(len(fmt_cell(r[k])) for r in rows)) for k in col_keys}

    def row_str(r):
        return "  ".join(fmt_cell(r[k]).ljust(col_w[k]) for k in col_keys)

    header = "  ".join(k.ljust(col_w[k]) for k in col_keys)
    sep    = "  ".join("-" * col_w[k] for k in col_keys)
    print(header)
    print(sep)
    for r in rows:
        print(row_str(r))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exps", nargs="*", default=[],
                        help="Filter by exp prefix, e.g. exp01 exp02")
    parser.add_argument("--sort", default="FAS",
                        help="Sort by this metric column (default: FAS)")
    parser.add_argument("--runs-dir", default=None,
                        help="Override runs directory")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    runs_dir = Path(args.runs_dir) if args.runs_dir else repo / "runs"

    rows = load_summaries(runs_dir, args.exps)
    print_table(rows, args.sort)


if __name__ == "__main__":
    main()
