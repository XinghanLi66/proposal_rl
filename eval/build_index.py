#!/usr/bin/env python3
"""
Build the FAS embedding index for a dataset split.

For each paper in the split, embed its abstract using a sentence encoder
and save to a .npz file:
  {
    "embeddings": float32 array [N, D],   # L2-normalized
    "arxiv_ids": str array [N],
    "abstracts": str array [N],
  }

This index is used by:
  - eval/evaluate.py  (final FAS computation)
  - train/reward.py   (live GRPO reward at training time)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_split_records(dataset_file: Path) -> tuple[list[str], list[str]]:
    arxiv_ids, abstracts = [], []
    with open(dataset_file) as f:
        for line in f:
            try:
                d = json.loads(line)
                if d.get("abstract"):
                    arxiv_ids.append(d["arxiv_id"])
                    abstracts.append(d["abstract"])
            except Exception:
                pass
    return arxiv_ids, abstracts


def load_local_records(
    arxiv_root: Path,
    months: list[str],
    target_categories: list[str],
) -> tuple[list[str], list[str]]:
    """Scan arxiv metadata directories directly for given months.

    Much faster than loading via dataset files, and produces a much larger
    corpus since ALL papers in those months are included (not just those
    with resolved S2 refs).  Used to build FAS retrieval indexes.
    """
    cat_set = set(target_categories) if target_categories else None
    arxiv_ids, abstracts = [], []
    for month_str in months:
        year_str, mon_str = month_str.split("-")
        month_dir = arxiv_root / year_str / mon_str
        if not month_dir.exists():
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
            if not d.get("abstract") or not d.get("arxiv_id"):
                continue
            if cat_set and not (set(d.get("categories", [])) & cat_set):
                continue
            arxiv_ids.append(d["arxiv_id"])
            abstracts.append(d["abstract"])
            count += 1
        log.info(f"  {month_str}: {count} papers (running total: {len(abstracts)})")
    return arxiv_ids, abstracts


def build_index_from_records(
    arxiv_ids: list[str],
    abstracts: list[str],
    output_file: Path,
    model_name: str,
    batch_size: int = 256,
    device: str = "cuda",
) -> None:
    log.info(f"Loading encoder: {model_name}")
    encoder = SentenceTransformer(model_name, device=device)
    log.info(f"Encoding {len(abstracts)} abstracts...")

    embeddings = encoder.encode(
        abstracts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_file,
        embeddings=embeddings.astype(np.float32),
        arxiv_ids=np.array(arxiv_ids),
        abstracts=np.array(abstracts),
    )
    log.info(f"Saved index: {output_file} | shape={embeddings.shape}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--splits", nargs="+", choices=["val", "test"], default=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--from-local",
        action="store_true",
        help=(
            "Build index by scanning arxiv metadata directories directly "
            "(ALL papers in the split months, not just those with S2 refs). "
            "Produces a much larger and more representative retrieval corpus."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    runs_dir = Path(cfg["runs_dir"])
    embed_model = cfg.get("fas", {}).get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")

    split_months = {
        "val": cfg.get("val_months", []),
        "test": cfg.get("test_months", []),
    }
    target_categories = cfg.get("target_categories", [])

    for split in args.splits:
        output_file = runs_dir / "eval" / f"{split}_index.npz"

        if args.from_local:
            arxiv_root = Path(cfg["arxiv_root"])
            months = split_months[split]
            log.info(f"Building {split} index from local arxiv metadata | months={months}")
            arxiv_ids, abstracts = load_local_records(arxiv_root, months, target_categories)
        else:
            dataset_file = runs_dir / "dataset" / f"{split}.jsonl"
            if not dataset_file.exists():
                log.warning(f"Dataset file not found: {dataset_file} — skipping")
                continue
            log.info(f"Building {split} index from dataset file")
            arxiv_ids, abstracts = load_split_records(dataset_file)

        log.info(f"Total for {split}: {len(abstracts)} abstracts")
        build_index_from_records(arxiv_ids, abstracts, output_file, embed_model, args.batch_size, args.device)


if __name__ == "__main__":
    main()
