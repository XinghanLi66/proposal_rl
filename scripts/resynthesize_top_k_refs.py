#!/usr/bin/env python3
"""
Parallel pre-population of the TopKRefs index cache (top_k_{k}_refs.jsonl).

For each record in runs/dataset/train.jsonl, calls the LLM to select the
top-k most important reference indices. Results are written to the new
top_k_{k}_refs.jsonl cache (independent of the legacy top_k_{k}.jsonl which
had duplicates and was built when abstracts were truncated to 400 chars).

The final prompt built from these indices already uses full abstracts
(abstract_chars=None in TopKRefsBuilder.build()).

Usage:
    python scripts/resynthesize_top_k_refs.py [--workers N] [--k K] [--config PATH]

Defaults:
    --workers 8
    --k       5
    --config  configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("ANTHROPIC_BASE_URL", "http://10.39.10.241:10001")
os.environ.setdefault("ANTHROPIC_AUTH_TOKEN", "123")
os.environ.setdefault("ANTHROPIC_API_KEY", "123")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_records(dataset_path: Path) -> list[dict]:
    records = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def select_one(builder, record: dict) -> tuple[str, bool, str]:
    """Returns (arxiv_id, was_cached, status_msg)."""
    arxiv_id = record.get("arxiv_id", "?")
    t0 = time.time()
    try:
        builder.build(record)
        elapsed = time.time() - t0
        cached = elapsed < 0.05
        return arxiv_id, cached, f"ok ({elapsed:.1f}s)"
    except Exception as exc:
        return arxiv_id, False, f"ERROR: {exc}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--k",       type=int, default=5)
    parser.add_argument("--config",  default="configs/base.yaml")
    args = parser.parse_args()

    import yaml
    config_path = Path(_REPO_ROOT) / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    runs_dir = cfg.get("runs_dir", str(Path(_REPO_ROOT) / "runs"))
    dataset_path = Path(runs_dir) / "dataset" / "train.jsonl"
    cache_dir    = Path(runs_dir) / "dataset" / "prompt_cache"
    new_cache    = cache_dir / f"top_k_{args.k}_refs.jsonl"

    log.info("Dataset:    %s", dataset_path)
    log.info("New cache:  %s", new_cache)
    log.info("Workers:    %d", args.workers)
    log.info("k:          %d", args.k)

    records = load_records(dataset_path)
    log.info("Loaded %d records", len(records))

    already_done: set[str] = set()
    if new_cache.exists():
        with open(new_cache) as f:
            for line in f:
                try:
                    already_done.add(json.loads(line)["key"])
                except Exception:
                    pass
    log.info("Already in new cache: %d / %d", len(already_done), len(records))

    pending = [r for r in records if r.get("arxiv_id", "") not in already_done]
    log.info("To select: %d", len(pending))
    if not pending:
        log.info("Nothing to do — all records are cached.")
        return

    pb_cfg = cfg.get("prompt_builder", {}).copy()
    pb_cfg.setdefault("runs_dir", runs_dir)
    pb_cfg["top_k"] = args.k
    pb_cfg["strategy"] = "top_k_refs"

    from train.prompt_builder import TopKRefsBuilder

    builders = [TopKRefsBuilder(pb_cfg) for _ in range(args.workers)]

    done = errors = 0
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(select_one, builders[i % args.workers], rec): rec
            for i, rec in enumerate(pending)
        }
        for fut in as_completed(futs):
            arxiv_id, was_cached, msg = fut.result()
            done += 1
            if "ERROR" in msg:
                errors += 1
                log.warning("[%d/%d] %s — %s", done, len(pending), arxiv_id, msg)
            elif done % 100 == 0 or done <= 5:
                elapsed = time.time() - t_start
                rate = done / elapsed
                eta = (len(pending) - done) / rate if rate > 0 else 0
                log.info(
                    "[%d/%d] %s — %s  (%.1f/s, ETA %.0fm)",
                    done, len(pending), arxiv_id, msg, rate, eta / 60,
                )

    elapsed_total = time.time() - t_start
    log.info(
        "Done. %d selected, %d errors, %.1fs total (%.2f/s)",
        done - errors, errors, elapsed_total, done / elapsed_total,
    )


if __name__ == "__main__":
    main()
