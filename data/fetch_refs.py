#!/usr/bin/env python3
"""
Fetch reference lists for arXiv papers via the Semantic Scholar API.

For each paper in our local arxiv store we call:
  GET /graph/v1/paper/ARXIV:{id}/references?fields=title,year,externalIds,abstract

Results are saved as JSONL, one line per paper:
  {"arxiv_id": "...", "refs": [{"arxiv_id": "...", "title": "...", "abstract": "...", "year": ...}, ...]}

Resumable: already-fetched IDs are skipped on restart.
Runs at ~0.9 req/s to stay under the S2 free-tier rate limit.
Pass --workers N to increase effective throughput (use multiple IPs / with API key).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import time
from pathlib import Path

import aiohttp
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TARGET_CATEGORIES = {"cs.LG", "cs.AI", "cs.CL", "cs.CV", "cs.IR", "cs.NE", "stat.ML"}
S2_URL = "https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_id}/references"
S2_FIELDS = "title,year,externalIds,abstract"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def iter_arxiv_papers(root: Path, months: list[str]) -> list[dict]:
    """Iterate over local arxiv metadata matching the given months and categories.

    Uses the known directory structure papers/{YYYY}/{MM}/{arxiv_id}/metadata.json
    to scan only the relevant month subdirectories — avoids a full rglob over 300K files.
    """
    papers = []
    for month_str in months:          # e.g. "2025-04"
        year_str, mon_str = month_str.split("-")
        month_dir = root / year_str / mon_str
        if not month_dir.is_dir():
            log.warning(f"Month dir not found: {month_dir}")
            continue
        count = 0
        for paper_dir in month_dir.iterdir():
            meta_path = paper_dir / "metadata.json"
            if not meta_path.exists():
                continue
            try:
                d = json.loads(meta_path.read_text())
            except Exception:
                continue
            cats = set(d.get("categories", []))
            if not cats.intersection(TARGET_CATEGORIES):
                continue
            papers.append(d)
            count += 1
        log.info(f"  {month_str}: {count} papers after category filter")
    return papers


def load_done(output_file: Path) -> set[str]:
    done = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["arxiv_id"])
                except Exception:
                    pass
    return done


async def fetch_one(
    session: aiohttp.ClientSession,
    arxiv_id: str,
    semaphore: asyncio.Semaphore,
    delay: float,
    api_key: str | None,
) -> dict | None:
    url = S2_URL.format(arxiv_id=arxiv_id)
    params = {"fields": S2_FIELDS, "limit": 100}
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    await semaphore.acquire()
    try:
        await asyncio.sleep(delay + random.uniform(0, 0.1))
        async with session.get(url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status == 429:
                log.warning(f"Rate limited on {arxiv_id}, sleeping 10s")
                await asyncio.sleep(10)
                return None
            if resp.status != 200:
                log.debug(f"HTTP {resp.status} for {arxiv_id}")
                return None
            data = await resp.json()
    except Exception as e:
        log.debug(f"Error fetching {arxiv_id}: {e}")
        return None
    finally:
        semaphore.release()

    refs = []
    for item in data.get("data", []):
        cp = item.get("citedPaper", {})
        ext = cp.get("externalIds") or {}
        ref_arxiv = ext.get("ArXiv")
        if not ref_arxiv:
            continue
        refs.append({
            "arxiv_id": ref_arxiv,
            "title": cp.get("title", ""),
            "abstract": cp.get("abstract") or "",
            "year": cp.get("year"),
        })
    return {"arxiv_id": arxiv_id, "refs": refs}


async def run_fetch(
    papers: list[dict],
    output_file: Path,
    done: set[str],
    workers: int,
    rps: float,
    api_key: str | None,
    limit: int | None,
) -> None:
    pending = [p for p in papers if p["arxiv_id"] not in done]
    if limit:
        pending = pending[:limit]
    log.info(f"Fetching refs for {len(pending)} papers ({len(done)} already done)")

    semaphore = asyncio.Semaphore(workers)
    delay_per_worker = workers / rps  # each worker sleeps this long between requests

    output_file.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    async with aiohttp.ClientSession() as session:
        with open(output_file, "a") as out_f:
            tasks = [
                asyncio.create_task(
                    fetch_one(session, p["arxiv_id"], semaphore, delay_per_worker, api_key)
                )
                for p in pending
            ]
            start = time.time()
            for i, coro in enumerate(asyncio.as_completed(tasks)):
                result = await coro
                if result and len(result["refs"]) > 0:
                    out_f.write(json.dumps(result) + "\n")
                    out_f.flush()
                    written += 1
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start
                    rate = (i + 1) / elapsed
                    eta = (len(pending) - i - 1) / rate if rate > 0 else 0
                    log.info(
                        f"Progress: {i+1}/{len(pending)} | written={written} "
                        f"| {rate:.1f} req/s | ETA {eta/60:.0f} min"
                    )

    log.info(f"Done. Wrote {written} records to {output_file}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--output", help="Override output JSONL path")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Max papers to fetch (for initial subset)")
    parser.add_argument("--api-key", default=os.environ.get("S2_API_KEY"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    arxiv_root = Path(cfg["arxiv_root"])
    months = cfg[f"{args.split}_months"]

    output_file = Path(args.output or cfg["runs_dir"]) / "dataset" / f"refs_{args.split}.jsonl"
    if args.output:
        output_file = Path(args.output)

    fetch_cfg = cfg.get("fetch", {})
    workers = args.workers or fetch_cfg.get("workers", 3)
    rps = fetch_cfg.get("rps", 0.9)
    if args.api_key:
        rps = min(rps * 10, 9.0)
        log.info(f"API key present → increased rate limit to {rps} rps")

    papers = iter_arxiv_papers(arxiv_root, months)
    log.info(f"Found {len(papers)} papers in split={args.split}")

    done = load_done(output_file)
    log.info(f"Already fetched: {len(done)}")

    asyncio.run(run_fetch(papers, output_file, done, workers, rps, args.api_key, args.limit))


if __name__ == "__main__":
    main()
