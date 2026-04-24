#!/usr/bin/env python3
"""
Build training/val/test dataset from fetched reference lists.

For each paper with fetched refs:
  1. Load the paper's metadata (abstract, title, categories)
  2. Load resolved references (those with abstracts in our local arxiv store)
  3. Filter: must have >= min_resolved_refs
  4. Write a training record:
     {
       "arxiv_id": str,
       "title": str,
       "abstract": str,         # the paper's own abstract (used as leakage check target)
       "categories": list[str],
       "created": str,
       "refs": [                # resolved references, up to max_refs_per_paper
           {"arxiv_id": str, "title": str, "abstract": str, "year": int},
           ...
       ],
       "prompt": str,           # formatted model input (reference list)
     }

The "target_proposal" (leakage-free, proposal-format abstract rewrite)
is added later by synthesize_cot.py via Claude API.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a research scientist with deep expertise in machine learning and AI. \
You will be given a list of papers (title and abstract each) that a researcher has been reading. \
Your task is to generate a structured research proposal for a novel research direction \
that these papers collectively suggest — as if you were the researcher proposing new work \
inspired by this body of literature."""

USER_TEMPLATE = """\
Below are {n_refs} papers from a researcher's reading list. \
Based on these references, propose a novel research direction.

{ref_block}

Generate a structured research proposal using this exact format:
<thinking>
[Analyze what themes, methods, and open problems span these references. \
Identify the most compelling gap or opportunity. \
Think step-by-step before writing the proposal.]
</thinking>
<proposal>
<problem>What core research problem should be addressed?</problem>
<gap>What gap in the existing literature motivates this work?</gap>
<key_insight>What key insight or hypothesis drives the proposed approach?</key_insight>
<approach>How would the proposed method work at a high level?</approach>
<expected_contributions>What are the expected scientific contributions?</expected_contributions>
</proposal>"""

REF_ENTRY_TEMPLATE = "[{idx}] {title} ({year})\nAbstract: {abstract}"


def build_ref_block(refs: list[dict], max_refs: int) -> str:
    entries = []
    for i, r in enumerate(refs[:max_refs]):
        abstract = (r.get("abstract") or "").strip()
        if len(abstract) > 400:
            abstract = abstract[:400] + "..."
        entries.append(REF_ENTRY_TEMPLATE.format(
            idx=i + 1,
            title=r.get("title", "Unknown"),
            year=r.get("year") or "n.d.",
            abstract=abstract or "(no abstract available)",
        ))
    return "\n\n".join(entries)


def load_local_abstracts(arxiv_root: Path) -> dict[str, dict]:
    """Build arxiv_id -> metadata map using the known directory structure.

    Iterates year/month subdirectories instead of rglob to avoid a full
    300K-file scan on network filesystems.
    """
    log.info("Building local arxiv index...")
    index = {}
    for year_dir in sorted(arxiv_root.iterdir()):
        if not year_dir.is_dir():
            continue
        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            for paper_dir in month_dir.iterdir():
                meta_path = paper_dir / "metadata.json"
                if not meta_path.exists():
                    continue
                try:
                    d = json.loads(meta_path.read_text())
                    aid = d.get("arxiv_id")
                    if aid:
                        index[aid] = d
                except Exception:
                    pass
            log.info(f"  {year_dir.name}/{month_dir.name}: {len(index)} total so far")
    log.info(f"Local index ready: {len(index)} papers")
    return index


def resolve_refs(raw_refs: list[dict], local_index: dict) -> list[dict]:
    """Keep only refs that exist in our local arxiv store and have abstracts."""
    resolved = []
    for r in raw_refs:
        aid = r.get("arxiv_id")
        if not aid:
            continue
        # Try to enrich from local store
        local = local_index.get(aid)
        if local and local.get("abstract"):
            resolved.append({
                "arxiv_id": aid,
                "title": local.get("title") or r.get("title", ""),
                "abstract": local["abstract"],
                "year": local.get("created", "")[:4] if local.get("created") else r.get("year"),
            })
        elif r.get("abstract"):
            # Use S2-returned abstract if not in local store
            resolved.append(r)
    return resolved


def process_split(
    refs_file: Path,
    local_index: dict,
    output_file: Path,
    min_resolved: int,
    max_refs: int,
    shuffle_seed: int = 42,
) -> int:
    if not refs_file.exists():
        log.error(f"Refs file not found: {refs_file}")
        return 0

    output_file.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_no_refs = 0
    skipped_not_local = 0

    rng = random.Random(shuffle_seed)
    records = []

    with open(refs_file) as f:
        for line in f:
            try:
                item = json.loads(line)
            except Exception:
                continue
            arxiv_id = item["arxiv_id"]
            raw_refs = item.get("refs", [])

            if len(raw_refs) == 0:
                skipped_no_refs += 1
                continue

            resolved = resolve_refs(raw_refs, local_index)
            if len(resolved) < min_resolved:
                skipped_not_local += 1
                continue

            # Get the paper's own metadata from local index
            paper_meta = local_index.get(arxiv_id)
            if not paper_meta or not paper_meta.get("abstract"):
                skipped_not_local += 1
                continue

            # Shuffle refs for variety
            rng.shuffle(resolved)

            ref_block = build_ref_block(resolved, max_refs)
            prompt = USER_TEMPLATE.format(n_refs=min(len(resolved), max_refs), ref_block=ref_block)

            records.append({
                "arxiv_id": arxiv_id,
                "title": paper_meta.get("title", ""),
                "abstract": paper_meta["abstract"],
                "categories": paper_meta.get("categories", []),
                "created": paper_meta.get("created", ""),
                "refs": resolved[:max_refs],
                "system": SYSTEM_PROMPT,
                "prompt": prompt,
            })

    rng.shuffle(records)

    with open(output_file, "w") as out_f:
        for rec in records:
            out_f.write(json.dumps(rec) + "\n")
            written += 1

    log.info(
        f"Wrote {written} examples | skipped_no_refs={skipped_no_refs} "
        f"skipped_not_local={skipped_not_local}"
    )
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    arxiv_root = Path(cfg["arxiv_root"])
    runs_dir = Path(cfg["runs_dir"])

    local_index = load_local_abstracts(arxiv_root)

    min_resolved = cfg.get("min_resolved_refs", 8)
    max_refs = cfg.get("max_refs_per_paper", 40)

    for split in args.splits:
        refs_file = runs_dir / "dataset" / f"refs_{split}.jsonl"
        output_file = runs_dir / "dataset" / f"{split}.jsonl"
        log.info(f"\n=== Processing split={split} ===")
        n = process_split(refs_file, local_index, output_file, min_resolved, max_refs)
        log.info(f"Split {split}: {n} examples → {output_file}")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
