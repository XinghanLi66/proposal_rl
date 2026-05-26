#!/usr/bin/env python3
"""
Parallel re-synthesis of RelatedWorkBuilder prompts (all-refs version).

Reads every record from runs/dataset/train.jsonl, calls RelatedWorkBuilder.build()
for each one, and skips records already in related_work_annotated.jsonl.

Usage:
    python scripts/resynthesize_related_work.py [--workers N] [--config PATH]

Defaults:
    --workers 8
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


def build_one(builder, record: dict) -> tuple[str, str]:
    arxiv_id = record.get("arxiv_id", "?")
    t0 = time.time()
    try:
        builder.build(record)
        elapsed = time.time() - t0
        return arxiv_id, f"ok ({elapsed:.1f}s)"
    except Exception as exc:
        return arxiv_id, f"ERROR: {exc}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--config",  default="configs/base.yaml")
    args = parser.parse_args()

    import yaml
    config_path = Path(_REPO_ROOT) / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    runs_dir     = cfg.get("runs_dir", str(Path(_REPO_ROOT) / "runs"))
    dataset_path = Path(runs_dir) / "dataset" / "train.jsonl"
    cache_dir    = Path(runs_dir) / "dataset" / "prompt_cache"
    annotated_f  = cache_dir / "related_work_annotated.jsonl"

    log.info("Dataset:   %s", dataset_path)
    log.info("Cache:     %s", annotated_f)
    log.info("Workers:   %d", args.workers)

    records = load_records(dataset_path)
    log.info("Loaded %d records", len(records))

    already_done: set[str] = set()
    if annotated_f.exists():
        with open(annotated_f) as f:
            for line in f:
                try:
                    already_done.add(json.loads(line)["key"])
                except Exception:
                    pass
    log.info("Already cached: %d / %d", len(already_done), len(records))

    pending = [r for r in records if r.get("arxiv_id", "") not in already_done]
    log.info("To synthesize: %d", len(pending))
    if not pending:
        log.info("Nothing to do — all records are cached.")
        return

    pb_cfg = cfg.get("prompt_builder", {}).copy()
    pb_cfg.setdefault("runs_dir", runs_dir)
    pb_cfg["strategy"] = "related_work"

    from train.prompt_builder import RelatedWorkBuilder
    builders = [RelatedWorkBuilder(pb_cfg) for _ in range(args.workers)]

    done = errors = 0
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(build_one, builders[i % args.workers], rec): rec
            for i, rec in enumerate(pending)
        }
        for fut in as_completed(futs):
            arxiv_id, msg = fut.result()
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
        "Done. %d synthesized, %d errors, %.1fs total (%.2f/s)",
        done - errors, errors, elapsed_total, done / elapsed_total,
    )


if __name__ == "__main__":
    main()
