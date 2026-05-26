#!/usr/bin/env python3
"""
Convert train.jsonl / train_cot.jsonl to Parquet files that verl expects.

verl RLHFDataset requires columns:
  prompt          — formatted chat-template string (or raw chat list in return_raw_chat mode)
  data_source     — "prs" | "fas" (selects reward fn)
  reward_model    — dict with "ground_truth" (the abstract)

verl SFT dataset (MultiTurnSFTDataset) requires columns:
  messages        — list of {role, content} dicts

Usage:
  # RL Parquet (train.jsonl → parquet)
  python train/make_parquet.py rl \
      --input  runs/dataset/train.jsonl \
      --output runs/dataset/train_prs.parquet \
      --reward-type prs \
      --config exp_config.yaml

  # SFT Parquet (train_cot.jsonl → parquet)
  python train/make_parquet.py sft \
      --input  runs/exp01/train_cot.jsonl \
      --output runs/exp01/train_sft.parquet \
      --config exp_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
import yaml

from train.prompt_builder import get_builder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _find_model(cfg: dict) -> str:
    candidates = [
        cfg.get("model_name_or_path", ""),
        "/newcpfs/user/yuanqianhao/hf_models/Qwen/Qwen2.5-7B-Instruct",
        "/newcpfs/user/gaochaochen/huimu/CodePrMP/models/Qwen2.5-7B-Instruct",
    ]
    for p in candidates:
        if p and Path(p).exists() and (Path(p) / "config.json").exists():
            return p
    raise FileNotFoundError("Base model not found. Set model_name_or_path in config.")


def build_rl_parquet(
    input_file: Path,
    output_file: Path,
    reward_type: str,
    cfg: dict,
    limit: int | None = None,
) -> None:
    from transformers import AutoTokenizer

    model_path = _find_model(cfg)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    builder = get_builder(cfg)
    log.info(f"Prompt builder: {type(builder).__name__}")

    rows = []
    with open(input_file) as f:
        for line in f:
            try:
                d = json.loads(line)
                if not d.get("arxiv_id"):
                    continue
            except Exception:
                continue
            rows.append(d)
            if limit and len(rows) >= limit:
                break

    records = []
    for r in rows:
        messages = [
            {"role": "system", "content": builder.system()},
            {"role": "user",   "content": builder.build(r)},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        records.append({
            "prompt":      prompt,
            "data_source": reward_type,
            "reward_model": {"ground_truth": r.get("abstract", "")},
            "arxiv_id":    r["arxiv_id"],
        })

    df = pd.DataFrame(records)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)
    log.info(f"Wrote {len(df)} RL rows → {output_file}")


def build_sft_parquet(
    input_file: Path,
    output_file: Path,
    cfg: dict,
    limit: int | None = None,
) -> None:
    strategy = cfg.get("prompt_builder", {}).get("strategy", "full_refs")
    builder = get_builder(cfg)
    log.info(f"SFT strategy: {strategy}  builder: {type(builder).__name__}")

    records = []
    with open(input_file) as f:
        for line in f:
            try:
                d = json.loads(line)
                if not d.get("cot_proposal") or d.get("leakage_flagged"):
                    continue
            except Exception:
                continue

            if strategy == "full_refs" and d.get("prompt"):
                system = d["system"]
                prompt = d["prompt"]
            else:
                system = builder.system()
                prompt = builder.build(d)

            messages = [
                {"role": "system",    "content": system},
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": d["cot_proposal"]},
            ]
            records.append({"messages": messages})
            if limit and len(records) >= limit:
                break

    df = pd.DataFrame(records)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)
    log.info(f"Wrote {len(df)} SFT rows → {output_file}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    rl_p = sub.add_parser("rl")
    rl_p.add_argument("--input",  required=True)
    rl_p.add_argument("--output", required=True)
    rl_p.add_argument("--reward-type", default="prs", choices=["prs", "fas", "ppl"])
    rl_p.add_argument("--config", default="configs/base.yaml")
    rl_p.add_argument("--limit",  type=int, default=None)

    sft_p = sub.add_parser("sft")
    sft_p.add_argument("--input",  required=True)
    sft_p.add_argument("--output", required=True)
    sft_p.add_argument("--config", default="configs/base.yaml")
    sft_p.add_argument("--limit",  type=int, default=None)

    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.mode == "rl":
        build_rl_parquet(
            Path(args.input), Path(args.output),
            args.reward_type, cfg, args.limit,
        )
    else:
        build_sft_parquet(Path(args.input), Path(args.output), cfg, args.limit)


if __name__ == "__main__":
    main()
