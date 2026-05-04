#!/usr/bin/env python3
"""
SFT cold-start: train on (reference list → CoT proposal) pairs.

Supports full fine-tuning and LoRA (set sft.finetune_mode in config).
Supports all prompt-builder strategies: prompts are rebuilt on-the-fly from
the `refs` field in train_cot.jsonl so that the SFT input distribution matches
the RL prompt distribution for each ablation.

Usage:
    torchrun --nproc_per_node=8 train/sft.py --config configs/base.yaml
    torchrun --nproc_per_node=8 train/sft.py --config my_config.yaml --resume

The output directory is controlled by sft.output_dir in the config
(default: {runs_dir}/sft/{strategy}).
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

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from train.prompt_builder import get_builder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def find_model(cfg: dict) -> str:
    candidates = [
        cfg.get("model_name_or_path", ""),
        "/newcpfs/user/yuanqianhao/hf_models/Qwen/Qwen2.5-7B-Instruct",
        "/newcpfs/user/gaochaochen/huimu/CodePrMP/models/Qwen2.5-7B-Instruct",
    ]
    for path in candidates:
        if path and Path(path).exists() and (Path(path) / "config.json").exists():
            return path
    raise FileNotFoundError("Base model not found. Set model_name_or_path in config.")


def load_records(dataset_file: Path) -> list[dict]:
    records = []
    with open(dataset_file) as f:
        for line in f:
            try:
                d = json.loads(line)
                if not d.get("cot_proposal") or d.get("leakage_flagged"):
                    continue
                records.append(d)
            except Exception:
                pass
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    sft = cfg.get("sft", {})
    runs_dir = Path(cfg["runs_dir"])

    strategy = cfg.get("prompt_builder", {}).get("strategy", "full_refs")
    finetune_mode = sft.get("finetune_mode", "lora")   # lora | full

    # Output dir: explicit override > runs/sft/{strategy}
    output_dir = Path(sft["output_dir"]) if sft.get("output_dir") else runs_dir / "sft" / strategy

    model_path   = find_model(cfg)
    dataset_file = runs_dir / "dataset" / "train_cot.jsonl"

    log.info(f"strategy={strategy}  finetune_mode={finetune_mode}  model={model_path}")
    log.info(f"output_dir={output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = load_records(dataset_file)
    log.info(f"Loaded {len(records)} training examples from {dataset_file}")

    # Build the prompt builder for this strategy.
    # Prompts are rebuilt from each record's `refs` field so the SFT input
    # distribution matches the RL training distribution for this ablation.
    # The cot_proposal (assistant target) is the same for all strategies.
    pb_cfg = cfg.get("prompt_builder", {}).copy()
    pb_cfg["runs_dir"] = str(runs_dir)
    builder = get_builder({**cfg, "prompt_builder": pb_cfg})
    log.info(f"Prompt builder: {type(builder).__name__}")

    rebuilt = 0
    texts = []
    for r in records:
        if strategy == "full_refs" and r.get("prompt"):
            # full_refs is the original strategy — reuse stored prompt directly
            system = r["system"]
            prompt = r["prompt"]
        else:
            # Rebuild prompt for this strategy from the refs field
            system = builder.system()
            prompt = builder.build(r)
            rebuilt += 1
        messages = [
            {"role": "system",    "content": system},
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": r["cot_proposal"]},
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))

    if rebuilt:
        log.info(f"Rebuilt {rebuilt}/{len(records)} prompts using strategy={strategy}")
    dataset = Dataset.from_dict({"text": texts})

    peft_config = None
    if finetune_mode == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=sft.get("lora_r", 64),
            lora_alpha=sft.get("lora_alpha", 128),
            target_modules=sft.get("lora_target_modules",
                                   ["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"]),
            lora_dropout=sft.get("lora_dropout", 0.05),
            bias="none",
        )

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=sft.get("num_train_epochs", 2),
        per_device_train_batch_size=sft.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=sft.get("gradient_accumulation_steps", 8),
        learning_rate=sft.get("learning_rate", 2e-4),
        lr_scheduler_type=sft.get("lr_scheduler_type", "cosine"),
        warmup_ratio=sft.get("warmup_ratio", 0.05),
        max_length=sft.get("max_seq_length", 8192),
        logging_steps=sft.get("logging_steps", 10),
        save_steps=sft.get("save_steps", 200),
        save_total_limit=3,
        bf16=sft.get("bf16", True),
        dataloader_num_workers=sft.get("dataloader_num_workers", 4),
        report_to="none",
        deepspeed=sft.get("deepspeed_config"),
        dataset_text_field="text",
        packing=True,
        logging_dir=str(output_dir / "logs"),
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, use_cache=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,     # None → full fine-tune
        processing_class=tokenizer,
    )

    log.info("Starting SFT training...")
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    log.info(f"SFT complete → {output_dir}/final")


if __name__ == "__main__":
    main()
