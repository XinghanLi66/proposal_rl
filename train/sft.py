#!/usr/bin/env python3
"""
SFT training via verl's FSDP SFT trainer.

verl SFTTrainer uses torchrun + FSDP (no DeepSpeed).
The dataset must be a Parquet file with a "messages" column containing
JSON-serialized list[{role, content}] dicts.

Usage:
  torchrun --nproc_per_node=8 train/sft.py --config exp_config.yaml

  # Smoke test (16 examples, 1 epoch, no config file needed):
  torchrun --nproc_per_node=8 train/sft.py --smoke [--output /tmp/smoke_sft]

The config must have a top-level "verl_sft" section (merged with
configs/verl_sft.yaml defaults) and a "sft.output_dir" entry.
Alternatively the script can be driven purely by the shell wrappers in
scripts/ablations/_combined_lib.sh, which invoke it with overrides.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.distributed
import yaml
from omegaconf import OmegaConf


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


def _smoke_cfg(output_dir: str) -> dict:
    """Minimal config for a 1-epoch smoke test — no config file required."""
    base_runs = str(Path(_REPO_ROOT) / "runs")
    return {
        "runs_dir": base_runs,
        "model_name_or_path": _find_model({}),
        "prompt_builder": {"strategy": "full_refs"},
        "sft": {
            "finetune_mode": "full",
            "output_dir": output_dir,
            "dataset_file": str(Path(base_runs) / "dataset" / "train_cot.jsonl"),
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 2e-5,
            "warmup_ratio": 0.05,
            "max_seq_length": 8192,
            "save_steps": 9999,
            "limit": 16,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--smoke", action="store_true",
                        help="Run a 1-epoch smoke test without a config file")
    parser.add_argument("--output", default="/newcpfs/lxh/agentic-training/proposal_rl/runs/smoke_sft",
                        help="Output dir for --smoke")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        cfg = _smoke_cfg(args.output)
        args.config = "configs/base.yaml"  # only used for make_parquet subprocess
    else:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    sft = cfg.get("sft", {})
    runs_dir = Path(cfg["runs_dir"])
    strategy = cfg.get("prompt_builder", {}).get("strategy", "full_refs")
    finetune_mode = sft.get("finetune_mode", "full")

    output_dir = Path(sft["output_dir"]) if sft.get("output_dir") else runs_dir / "sft" / strategy
    dataset_file = Path(sft["dataset_file"]) if sft.get("dataset_file") else runs_dir / "dataset" / "train_cot.jsonl"
    model_path = _find_model(cfg)

    # Convert JSONL → Parquet if needed
    parquet_file = dataset_file.with_suffix(".parquet")
    if not parquet_file.exists() or parquet_file.stat().st_mtime < dataset_file.stat().st_mtime:
        import subprocess
        rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            subprocess.run(
                [
                    sys.executable, "train/make_parquet.py", "sft",
                    "--input", str(dataset_file),
                    "--output", str(parquet_file),
                    "--config", args.config,
                ] + (["--limit", str(sft["limit"])] if sft.get("limit") else []),
                cwd=_REPO_ROOT, check=True,
            )
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    # Build verl SFT config by composing the referenced sub-configs manually.
    # sft_trainer_engine.yaml uses Hydra defaults (model, engine, optim, profiler)
    # which plain OmegaConf.load() doesn't resolve. We compose them explicitly here.
    import importlib.resources as _res
    import verl.trainer.config as _vcfg_pkg

    def _load_sub(sub_path: str) -> "OmegaConf":
        with _res.files(_vcfg_pkg).joinpath(sub_path).open() as _f:
            return OmegaConf.create(_f.read())

    vcfg = OmegaConf.merge(
        _load_sub("model/hf_model.yaml"),              # → model
        OmegaConf.create({}),
    )
    vcfg = OmegaConf.create({
        "model":    _load_sub("model/hf_model.yaml"),
        "engine":   _load_sub("engine/fsdp.yaml"),
        "optim":    _load_sub("optim/fsdp.yaml"),
        "profiler": _load_sub("profiler/profiler.yaml"),
    })
    # Merge the top-level sft_trainer_engine fields (data, checkpoint, trainer) on top.
    # We only take the non-defaults section.
    top_cfg = _load_sub("sft_trainer_engine.yaml")
    # Drop the Hydra `defaults` key (it's not an OmegaConf dict key)
    vcfg = OmegaConf.merge(vcfg, OmegaConf.masked_copy(top_cfg, [k for k in top_cfg if k != "defaults"]))
    # Finally overlay our project-specific settings
    project_overrides_path = Path(_REPO_ROOT) / "configs" / "verl_sft.yaml"
    if project_overrides_path.exists():
        vcfg = OmegaConf.merge(vcfg, OmegaConf.load(str(project_overrides_path)))

    n_gpu = int(os.environ.get("WORLD_SIZE", "1"))
    global_bsz = sft.get("per_device_train_batch_size", 2) * n_gpu * sft.get("gradient_accumulation_steps", 1)

    # 4×H800 80GB OOM profile (FSDP, no flash-attn):
    #   full-FT at max_token_len=8192 → ~79 GB per GPU (OOM)
    #   LoRA r=64 all-linear at max_token_len=8192 → also OOM
    # Fixes: gradient checkpointing + reduced token budget when n_gpu < 8.
    # 6144 covers full_refs sequences (max ~5.7k tokens) with safe headroom.
    is_lora = finetune_mode == "lora"
    tight_memory = n_gpu < 8
    use_grad_ckpt = is_lora or tight_memory
    effective_max_tokens = 6144 if tight_memory else sft.get("max_seq_length", 8192)

    overrides = OmegaConf.create({
        "data": {
            "train_files": str(parquet_file),
            "train_batch_size": global_bsz,
            "micro_batch_size_per_gpu": sft.get("per_device_train_batch_size", 2),
            "max_token_len_per_gpu": effective_max_tokens,
            "max_length": effective_max_tokens,
        },
        "model": {
            "path": model_path,
            "lora_rank": sft.get("lora_r", 0) if is_lora else 0,
            "lora_alpha": sft.get("lora_alpha", 128) if is_lora else None,
            "enable_gradient_checkpointing": use_grad_ckpt,
            "override_config": {"attn_implementation": "sdpa"},
        },
        "optim": {
            "lr": float(sft.get("learning_rate", 2e-4)),
            "lr_warmup_steps_ratio": float(sft.get("warmup_ratio", 0.05)),
        },
        "trainer": {
            "default_local_dir": str(output_dir),
            "total_epochs": sft.get("num_train_epochs", 2),
            "save_freq": sft.get("save_steps", 200),
            "resume_mode": "auto" if args.resume else "disable",
            "n_gpus_per_node": n_gpu,
            "logger": ["console"],
        },
    })
    vcfg = OmegaConf.merge(vcfg, overrides)

    # run_sft initializes the distributed process group (required by verl SFTTrainer)
    from verl.trainer.sft_trainer import run_sft
    run_sft(vcfg)

    # Merge sharded FSDP checkpoint → HF model at output_dir/final
    # Only rank 0 runs the merge (single-process; verl.model_merger is not distributed)
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        final_dir = output_dir / "final"
        ckpt_dirs = sorted(output_dir.glob("global_step_*"))
        if ckpt_dirs and not (final_dir / "config.json").exists():
            import subprocess
            subprocess.run(
                [
                    sys.executable, "-m", "verl.model_merger", "merge",
                    "--backend", "fsdp",
                    "--local_dir", str(ckpt_dirs[-1]),
                    "--target_dir", str(final_dir),
                ],
                cwd=_REPO_ROOT, check=False,
            )


if __name__ == "__main__":
    main()
