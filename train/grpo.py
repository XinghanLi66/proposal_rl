#!/usr/bin/env python3
"""
GRPO fine-tuning on top of the SFT checkpoint.

Uses TRL GRPOTrainer with three reward functions:
  - FAS reward (embedding similarity to val corpus)
  - Format reward (XML structure compliance)
  - Anti-leakage reward (penalizes proposals too similar to source abstract)

The val corpus FAISS index must be built before running this script
(via eval/build_index.py).

Launch with torchrun for multi-GPU:
  torchrun --nproc_per_node=8 train/grpo.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure the repo root is first in sys.path so that `train.reward` resolves
# to our package, not any other `train.py` on the system (e.g. LLaMAFactory).
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from train.reward import init_reward, reward_fas, reward_format, reward_antileak

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_dataset_records(dataset_file: Path, limit: int | None = None) -> list[dict]:
    records = []
    with open(dataset_file) as f:
        for line in f:
            try:
                d = json.loads(line)
                if not d.get("prompt"):
                    continue
                records.append({
                    "prompt": _build_prompt_messages(d),
                    "abstract": d.get("abstract", ""),    # passed to antileak reward via kwargs
                    "arxiv_id": d["arxiv_id"],
                })
            except Exception:
                pass
            if limit and len(records) >= limit:
                break
    return records


def _build_prompt_messages(d: dict) -> str:
    """Return the prompt as a list of messages (for chat template)."""
    return json.dumps([
        {"role": "system", "content": d["system"]},
        {"role": "user", "content": d["prompt"]},
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--sft-checkpoint", help="Path to SFT checkpoint (LoRA). Defaults to runs/sft/final")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit training examples (debug)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    grpo_cfg = cfg.get("grpo", {})
    runs_dir = Path(cfg["runs_dir"])

    sft_checkpoint = Path(args.sft_checkpoint) if args.sft_checkpoint else runs_dir / "sft" / "final"
    output_dir = Path(args.output_dir) if args.output_dir else runs_dir / "grpo"
    fas_index_file = grpo_cfg.get("fas_index_file", str(runs_dir / "eval" / "val_index.npz"))

    # Resolve base model path (inline, avoid re-import to dodge sys.path conflicts)
    base_model_path = cfg.get("model_name_or_path", "Qwen/Qwen2.5-7B-Instruct")
    candidates = [
        base_model_path,
        "/newcpfs/user/sujianghao/model/Qwen/Qwen2.5-7B-Instruct",
        str(Path(cfg.get("model_cache_dir", "")) / "Qwen2.5-7B-Instruct"),
    ]
    for p in candidates:
        if p and Path(p).exists() and (Path(p) / "config.json").exists():
            base_model_path = p
            break

    # Init reward
    embed_model = cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")
    import datetime as _dt
    rollout_log = str(runs_dir / "logs" / f"rollouts_grpo_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    init_reward(
        index_file=fas_index_file,
        embed_model=embed_model,
        topk=cfg.get("eval", {}).get("topk", 50),
        fas_weight=grpo_cfg.get("reward_fas_weight", 0.6),
        format_weight=grpo_cfg.get("reward_format_weight", 0.2),
        antileak_weight=grpo_cfg.get("reward_antileak_weight", 0.2),
        antileak_threshold=grpo_cfg.get("reward_antileak_threshold", 0.80),
        rollout_log_file=rollout_log,
    )
    log.info(f"Reward functions initialized; rollout log → {rollout_log}")

    # Load dataset
    dataset_file = runs_dir / "dataset" / "train.jsonl"
    records = load_dataset_records(dataset_file, args.limit)
    log.info(f"Loaded {len(records)} training examples")

    # The dataset needs "prompt" as a string that can be formatted by the tokenizer.
    # We store messages as JSON string and unpack in a custom format function.
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_prompt(example):
        messages = json.loads(example["prompt"])
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    hf_records = {
        "prompt": [format_prompt(r) for r in records],
        "abstract": [r["abstract"] for r in records],
        "arxiv_id": [r["arxiv_id"] for r in records],
    }
    dataset = Dataset.from_dict(hf_records)

    # LoRA config (continue training from SFT LoRA)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=grpo_cfg.get("lora_r", 64),
        lora_alpha=grpo_cfg.get("lora_alpha", 128),
        target_modules=grpo_cfg.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=grpo_cfg.get("lora_dropout", 0.05),
        bias="none",
    )

    ds_config = grpo_cfg.get("deepspeed_config")
    grpo_training_args = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=grpo_cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=grpo_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=grpo_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=grpo_cfg.get("learning_rate", 5e-6),
        lr_scheduler_type=grpo_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=grpo_cfg.get("warmup_ratio", 0.05),
        max_completion_length=grpo_cfg.get("max_completion_length", 1024),
        num_generations=grpo_cfg.get("num_generations", 8),
        beta=grpo_cfg.get("kl_coeff", 0.05),       # TRL 1.0.0: kl_coeff → beta
        reward_weights=[                            # per-function weights
            grpo_cfg.get("reward_fas_weight", 0.6),
            grpo_cfg.get("reward_format_weight", 0.2),
            grpo_cfg.get("reward_antileak_weight", 0.2),
        ],
        scale_rewards=True,                         # normalize rewards within each group
        logging_steps=grpo_cfg.get("logging_steps", 5),
        save_steps=grpo_cfg.get("save_steps", 100),
        save_total_limit=3,
        bf16=grpo_cfg.get("bf16", True),
        report_to="none",
        deepspeed=ds_config,
        logging_dir=str(output_dir / "logs"),
    )

    # Load model: base + SFT LoRA adapter
    log.info(f"Loading base model: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,
    )

    if sft_checkpoint.exists():
        log.info(f"Loading SFT LoRA from {sft_checkpoint}")
        model = PeftModel.from_pretrained(model, str(sft_checkpoint), is_trainable=True)
    else:
        log.warning(f"SFT checkpoint not found at {sft_checkpoint}, training from base")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fas, reward_format, reward_antileak],
        args=grpo_training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config if not sft_checkpoint.exists() else None,
    )

    log.info("Starting GRPO training...")
    trainer.train(resume_from_checkpoint=args.resume)

    log.info(f"Saving model to {output_dir / 'final'}")
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    log.info("GRPO training complete.")


if __name__ == "__main__":
    main()
