#!/usr/bin/env python3
"""
RL fine-tuning (GRPO or RLOO) on top of the SFT checkpoint.

Reward mode is controlled by rl.reward_type in the config:
  prs  — Paper Recovery Score (default): cosine sim(proposal, abstract)
  fas  — Future Alignment Score (legacy): similarity to val corpus index

Usage:
    torchrun --nproc_per_node=8 train/rl.py --config configs/base.yaml
    torchrun --nproc_per_node=8 train/rl.py --config configs/base.yaml --resume
"""

from __future__ import annotations

import argparse
import datetime
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
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

from train.prompt_builder import get_builder
from train.reward import (
    init_encoder, init_reward,
    reward_prs, reward_fas, reward_format, reward_antileak,
)

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


def load_records(dataset_file: Path, builder, tokenizer, limit: int | None) -> dict:
    records = []
    with open(dataset_file) as f:
        for line in f:
            try:
                d = json.loads(line)
                if not d.get("prompt") or not d.get("arxiv_id"):
                    continue
                records.append(d)
            except Exception:
                pass
            if limit and len(records) >= limit:
                break

    prompts, abstracts, arxiv_ids = [], [], []
    for r in records:
        messages = [
            {"role": "system", "content": builder.system()},
            {"role": "user",   "content": builder.build(r)},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))
        abstracts.append(r.get("abstract", ""))
        arxiv_ids.append(r["arxiv_id"])

    return {"prompt": prompts, "abstract": abstracts, "arxiv_id": arxiv_ids}


def build_trainer(algo: str, model, reward_funcs, args, dataset, tokenizer, peft_config):
    if algo == "grpo":
        from trl import GRPOTrainer
        return GRPOTrainer(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
    elif algo == "rloo":
        from trl import RLOOTrainer
        return RLOOTrainer(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
    else:
        raise ValueError(f"Unknown rl.algo: {algo!r}. Choose 'grpo' or 'rloo'.")


def build_training_args(algo: str, rl: dict, reward_weights: list[float], output_dir: Path):
    common = dict(
        output_dir=str(output_dir),
        num_train_epochs=rl.get("num_train_epochs", 1),
        per_device_train_batch_size=rl.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=rl.get("gradient_accumulation_steps", 4),
        learning_rate=float(rl.get("learning_rate", 5e-6)),
        lr_scheduler_type=rl.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(rl.get("warmup_ratio", 0.05)),
        max_completion_length=int(rl.get("max_completion_length", 2048)),
        beta=float(rl.get("kl_coeff", 0.05)),
        reward_weights=reward_weights,
        scale_rewards=True,
        logging_steps=rl.get("logging_steps", 5),
        save_steps=rl.get("save_steps", 100),
        save_total_limit=3,
        bf16=rl.get("bf16", True),
        report_to="none",
        deepspeed=rl.get("deepspeed_config"),
        logging_dir=str(output_dir / "logs"),
    )
    if algo == "grpo":
        from trl import GRPOConfig
        return GRPOConfig(num_generations=rl.get("num_generations", 8), **common)
    else:
        from trl import RLOOConfig
        return RLOOConfig(rloo_k=rl.get("rloo_k", 4), **common)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rl       = cfg.get("rl", {})
    fas_cfg  = cfg.get("fas", {})
    runs_dir = Path(cfg["runs_dir"])

    algo          = rl.get("algo", "grpo")
    finetune_mode = rl.get("finetune_mode", "lora")
    reward_type   = rl.get("reward_type", "prs")
    sft_ckpt      = Path(rl.get("sft_checkpoint") or str(runs_dir / "sft" / "final"))
    output_dir    = Path(rl["output_dir"]) if rl.get("output_dir") else runs_dir / "rl"
    embed_model   = fas_cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")

    log.info(f"algo={algo}  finetune_mode={finetune_mode}  reward={reward_type}")

    base_path = find_model(cfg)
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    builder = get_builder(cfg)
    log.info(f"Prompt builder: {type(builder).__name__}")

    # Rollout logs go next to the output dir so per-experiment dashboards find them
    rollout_log_dir = output_dir.parent / "logs"
    rollout_log_dir.mkdir(parents=True, exist_ok=True)
    rollout_log = str(rollout_log_dir /
                      f"rollouts_rl_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")

    # Reward setup — diverges by reward_type
    if reward_type == "prs":
        init_encoder(embed_model, rollout_log_file=rollout_log)
        reward_funcs = [reward_prs, reward_format]
        reward_weights = [
            rl.get("reward_prs_weight", 0.8),
            rl.get("reward_format_weight", 0.2),
        ]
    elif reward_type == "fas":
        fas_index = str(runs_dir / "eval" / "val_index.npz")
        init_reward(
            index_file=fas_index,
            embed_model=embed_model,
            topk=fas_cfg.get("topk", 50),
            fas_weight=rl.get("reward_fas_weight", 0.6),
            format_weight=rl.get("reward_format_weight", 0.2),
            antileak_weight=rl.get("reward_antileak_weight", 0.2),
            antileak_threshold=rl.get("reward_antileak_threshold", 0.80),
            rollout_log_file=rollout_log,
        )
        reward_funcs = [reward_fas, reward_format, reward_antileak]
        reward_weights = [
            rl.get("reward_fas_weight", 0.6),
            rl.get("reward_format_weight", 0.2),
            rl.get("reward_antileak_weight", 0.2),
        ]
    else:
        raise ValueError(f"Unknown rl.reward_type: {reward_type!r}. Choose 'prs' or 'fas'.")

    log.info(f"Rollout log → {rollout_log}")

    # Dataset
    dataset_file = runs_dir / "dataset" / "train.jsonl"
    hf_data = load_records(dataset_file, builder, tokenizer, rl.get("limit"))
    dataset = Dataset.from_dict(hf_data)
    log.info(f"Loaded {len(dataset)} training examples")

    # LoRA config
    peft_config = None
    if finetune_mode == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rl.get("lora_r", 64),
            lora_alpha=rl.get("lora_alpha", 128),
            target_modules=rl.get("lora_target_modules",
                                  ["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"]),
            lora_dropout=rl.get("lora_dropout", 0.05),
            bias="none",
        )

    training_args = build_training_args(algo, rl, reward_weights, output_dir)

    _sft_is_lora = sft_ckpt.exists() and (sft_ckpt / "adapter_config.json").exists()
    _sft_is_full = sft_ckpt.exists() and not _sft_is_lora

    if _sft_is_lora:
        log.info(f"Loading base model: {base_path}")
        model = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=torch.bfloat16, trust_remote_code=True, use_cache=False,
        )
        log.info(f"Loading SFT LoRA adapter: {sft_ckpt}")
        model = PeftModel.from_pretrained(model, str(sft_ckpt), is_trainable=True)
    elif _sft_is_full:
        log.info(f"Loading full SFT model as base: {sft_ckpt}")
        model = AutoModelForCausalLM.from_pretrained(
            str(sft_ckpt), torch_dtype=torch.bfloat16, trust_remote_code=True, use_cache=False,
        )
    else:
        log.warning(f"SFT checkpoint not found at {sft_ckpt}, starting from base model")
        model = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=torch.bfloat16, trust_remote_code=True, use_cache=False,
        )

    trainer = build_trainer(
        algo, model, reward_funcs, training_args, dataset, tokenizer,
        peft_config if not _sft_is_lora else None,
    )

    log.info(f"Starting {algo.upper()} training...")
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    log.info("RL training complete.")


if __name__ == "__main__":
    main()
