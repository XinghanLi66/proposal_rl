#!/usr/bin/env python3
"""
RL fine-tuning (GRPO) via verl's Ray PPO/GRPO trainer.

verl uses Ray + FSDP for the actor/rollout and does not require DeepSpeed.
The dataset must be a Parquet file built by train/make_parquet.py.
Reward is computed by train/verl_reward.py via verl's custom_reward_function hook.

Usage:
  # Full experiment run:
  python train/rl.py --config exp_config.yaml [--resume]

  # Smoke test (1 step, 8 examples, no config file needed):
  python train/rl.py --smoke [--output /tmp/smoke_rl]

(Single-process launcher — Ray handles the distributed workers internally.)

Reward modes (rl.reward_type in config):
  prs  — Paper Recovery Score: cosine sim(proposal, abstract)
  fas  — Future Alignment Score: similarity to held-out future corpus index
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ── Environment setup (must run before Ray/vLLM/torch are imported) ──────────
# Route all cache dirs to /dev/shm to avoid /tmp (100G, often full) and NFS.
def _setup_env():
    defaults = {
        "TORCHINDUCTOR_CACHE_DIR": "/dev/shm/torch_inductor",
        "TRITON_CACHE_DIR":        "/dev/shm/triton_cache",
        "TMPDIR":                  "/dev/shm/tmp",
        "OMP_NUM_THREADS":         "1",
        # torchrl registers 'fp32_overrides' vLLM plugin that imports the
        # removed vllm.worker module (vLLM >=0.7). Disable external plugins.
        "VLLM_PLUGINS":            "",
        # Ray workers all import sympy which calls dlopen('gmpy2.so') concurrently.
        # On glibc 2.28, concurrent dlopen of TLS-bearing libraries triggers:
        #   _dl_allocate_tls_init: Assertion `listp != NULL` failed!
        # Setting SYMPY_GROUND_TYPES=python tells sympy to use pure-Python integer
        # arithmetic and skip the gmpy2 import entirely, eliminating the race.
        "SYMPY_GROUND_TYPES": "python",
        # vLLM v1 multiprocessing executor triggers intermittent "none_dealloc: deallocating
        # None" Python GC refcount corruption during collective_rpc in update_weights.
        # Uniproc mode runs the GPU worker in-process, avoiding cross-process IPC teardown.
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    }
    # LD_PRELOAD: force libcuda into the process address space before any threads
    # are created, preventing glibc 2.28 TLS assertion failures when many Ray
    # workers concurrently dlopen CUDA (_dl_allocate_tls_init: listp != NULL).
    _libcuda = "/usr/lib/x86_64-linux-gnu/libcuda.so.1"
    if os.path.exists(_libcuda):
        _existing = os.environ.get("LD_PRELOAD", "")
        os.environ["LD_PRELOAD"] = (f"{_existing}:{_libcuda}" if _existing else _libcuda)
    for k, v in defaults.items():
        os.environ.setdefault(k, v)
    for d in [os.environ["TORCHINDUCTOR_CACHE_DIR"],
              os.environ["TRITON_CACHE_DIR"],
              os.environ["TMPDIR"]]:
        Path(d).mkdir(parents=True, exist_ok=True)

_setup_env()

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
    """Minimal config for a 1-step smoke test — no config file required.
    Output goes to /dev/shm (tmpfs, ~1.3T free) not /tmp (often 100% full)."""
    base_runs = str(Path(_REPO_ROOT) / "runs")
    model_path = _find_model({
        "model_name_or_path": str(
            Path(_REPO_ROOT) / "runs/model_cache"
            / "models--Qwen--Qwen2.5-7B-Instruct"
            / "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
        ),
    })
    return {
        "runs_dir": base_runs,
        "model_name_or_path": model_path,
        "fas": {"embed_model": "sentence-transformers/all-MiniLM-L6-v2"},
        "rl": {
            "algo": "grpo",
            "finetune_mode": "full",
            "sft_checkpoint": "",
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 5e-6,
            "warmup_ratio": 0.05,
            "max_completion_length": 2048,
            "num_generations": 8,
            "kl_coeff": 0.05,
            "save_steps": 9999,
            "reward_type": "prs",
            "reward_antileak_threshold": 0.80,
            "limit": 8,
            "output_dir": output_dir,
            "n_gpus": 8,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml",
                        help="Experiment config YAML (ignored when --smoke is set)")
    parser.add_argument("--smoke", action="store_true",
                        help="Run a 1-step smoke test without a config file")
    parser.add_argument("--output", default="/newcpfs/lxh/agentic-training/proposal_rl/runs/smoke_rl",
                        help="Output dir for --smoke")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        cfg = _smoke_cfg(args.output)
    else:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    rl = cfg.get("rl", {})
    fas_cfg = cfg.get("fas", {})
    runs_dir = Path(cfg["runs_dir"])

    algo          = rl.get("algo", "grpo")
    finetune_mode = rl.get("finetune_mode", "full")
    reward_type   = rl.get("reward_type", "prs")
    sft_ckpt      = Path(rl.get("sft_checkpoint") or str(runs_dir / "sft" / "final"))
    output_dir    = Path(rl["output_dir"]) if rl.get("output_dir") else runs_dir / "rl"
    n_gpu         = int(os.environ.get("NGPU", rl.get("n_gpus", 8)))

    dataset_file  = runs_dir / "dataset" / "train.jsonl"
    # Use a limit-specific parquet name when --limit is set (e.g. smoke/mini runs),
    # so that a limited rebuild never overwrites the canonical full-dataset parquet.
    _limit = rl.get("limit")
    if _limit:
        parquet_file = runs_dir / "dataset" / f"train_{reward_type}_limit{_limit}.parquet"
    else:
        parquet_file = runs_dir / "dataset" / f"train_{reward_type}.parquet"

    # Build Parquet if stale
    if not parquet_file.exists() or parquet_file.stat().st_mtime < dataset_file.stat().st_mtime:
        import subprocess
        subprocess.run(
            [
                sys.executable, "train/make_parquet.py", "rl",
                "--input", str(dataset_file),
                "--output", str(parquet_file),
                "--reward-type", reward_type,
                "--config", args.config if not args.smoke else "configs/base.yaml",
            ] + (["--limit", str(_limit)] if _limit else []),
            cwd=_REPO_ROOT, check=True,
        )

    # Set env vars consumed by train/verl_reward.py workers
    embed_model = fas_cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")
    os.environ["EMBED_MODEL"] = embed_model
    if reward_type == "fas":
        fas_index = str(runs_dir / "eval" / "val_index.npz")
        os.environ["FAS_INDEX_FILE"] = fas_index

    base_model = _find_model(cfg)

    # PPL reward uses the same base model as the actor for P_policy(abstract | proposal).
    # Point reward workers at the SFT checkpoint (or base model if no SFT was run).
    if reward_type == "ppl":
        _ref_path = str(sft_ckpt) if (sft_ckpt / "config.json").exists() else base_model
        os.environ["REF_MODEL_PATH"] = _ref_path

    # verl PPO/GRPO config: start from the generated full config (has all defaults),
    # then overlay our project-specific settings from verl_ppo.yaml.
    import importlib.resources as _res
    import verl.trainer.config as _vcfg_pkg
    with _res.files(_vcfg_pkg).joinpath("_generated_ppo_trainer.yaml").open() as _f:
        vcfg = OmegaConf.create(_f.read())
    verl_cfg_path = Path(_REPO_ROOT) / "configs" / "verl_ppo.yaml"
    vcfg = OmegaConf.merge(vcfg, OmegaConf.load(str(verl_cfg_path)))

    use_grpo = (algo == "grpo")
    actor_lr = float(rl.get("learning_rate", 5e-6))

    # 4×H800 80GB OOM profile for RL (FSDP actor + vLLM rollout, no flash-attn):
    # full-FT actor update at max_token_len=16384 OOMs. Fixes: gradient checkpointing
    # + reduced token budget + lower vLLM memory utilization when n_gpu < 8.
    tight_memory = n_gpu < 8
    rl_grad_ckpt = rl.get("gradient_checkpointing", False) or tight_memory
    max_prompt_len = 3072 if tight_memory else 4096
    max_resp_len = int(rl.get("max_completion_length", 2048))
    ppo_max_token = 8192 if tight_memory else 16384
    # LoRA weight update spikes GPU0 by ~16GB on top of steady-state allocations:
    # vLLM 35.5 GB + FSDP actor 8.5 GB + FSDP ref 5.7 GB + LoRA update spike ~16 GB
    # = ~65.7 GB steady-state + spike overhead → OOM at step 1041 on 79.11 GB H800.
    # Use 0.35 for LoRA (reserves ~7.9 GB more vs 0.45) to survive weight-update spikes.
    is_lora_rl = (finetune_mode == "lora")
    if tight_memory:
        vllm_gpu_util = 0.35 if is_lora_rl else 0.45
    else:
        vllm_gpu_util = 0.6
    # CUDA graph capture adds ~28GB overhead on top of KV cache on 80GB H800.
    # With vllm_gpu_util=0.45 (~35GB KV) + graphs (~28GB) + FSDP init, GPU 0 OOMs
    # before the ref model can be cast (torch.nn.Module.convert() crash).
    # Disable graph capture for tight-memory runs — eager mode uses ~35GB total.
    vllm_enforce_eager = tight_memory

    overrides = OmegaConf.create({
        "data": {
            "train_files": str(parquet_file),
            "val_files":   str(parquet_file),
            "train_batch_size":   n_gpu * rl.get("per_device_train_batch_size", 1)
                                  * rl.get("gradient_accumulation_steps", 1),
            "max_prompt_length":  max_prompt_len,
            "max_response_length": max_resp_len,
            "return_raw_chat":    True,
        },
        "actor_rollout_ref": {
            "model": {
                "path": str(sft_ckpt) if (sft_ckpt / "config.json").exists() else base_model,
                "lora_rank": rl.get("lora_r", 0) if finetune_mode == "lora" else 0,
                "lora_alpha": rl.get("lora_alpha", 128),
                "enable_gradient_checkpointing": rl_grad_ckpt,
                "override_config": {"attn_implementation": "sdpa"},
            },
            "actor": {
                "optim": {
                    "lr": actor_lr,
                    "lr_warmup_steps_ratio": float(rl.get("warmup_ratio", 0.05)),
                },
                "ppo_mini_batch_size": n_gpu * rl.get("per_device_train_batch_size", 1),
                "ppo_micro_batch_size_per_gpu": rl.get("per_device_train_batch_size", 1),
                "ppo_max_token_len_per_gpu": ppo_max_token,
                "use_kl_loss": True,
                "kl_loss_coef": float(rl.get("kl_coeff", 0.05)),
                "clip_ratio": 0.2,
                "rollout_n": int(rl.get("num_generations", 8)),
            },
            "rollout": {
                "n": int(rl.get("num_generations", 8)),
                "tensor_model_parallel_size": 1,
                "gpu_memory_utilization": vllm_gpu_util,
                "enforce_eager": vllm_enforce_eager,
                "max_num_batched_tokens": ppo_max_token,
                # PPL reward: each AgentLoopWorker loads a full 7B model into CPU RAM.
                # 8 workers × ~120 GB = ~960 GB → OOM. Limit to 1 agent worker.
                **({"agent": {"num_workers": 1}} if reward_type == "ppl" else {}),
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": rl.get("per_device_train_batch_size", 1),
                "fsdp_config": {"param_offload": False},
            },
        },
        "algorithm": {
            "adv_estimator": "grpo" if use_grpo else "reinforce_plus_plus",
            "use_kl_in_reward": False,
            "kl_ctrl": {"kl_coef": float(rl.get("kl_coeff", 0.05))},
        },
        "reward": {
            "custom_reward_function": {
                "path": str(Path(_REPO_ROOT) / "train" / "verl_reward.py"),
                "name": "compute_score",
                "reward_kwargs": {
                    "antileak_threshold": float(rl.get("reward_antileak_threshold", 0.80)),
                },
            },
            # PPL reward loads a 7B model per worker — keep to 1 to avoid
            # 4 independent model copies flooding CPU RAM (~230 GB each).
            **({"num_workers": 1} if reward_type == "ppl" else {}),
        },
        "trainer": {
            "default_local_dir": str(output_dir),
            "total_epochs": int(rl.get("num_train_epochs", 1)),
            "save_freq":    int(rl.get("save_steps", 100)),
            "n_gpus_per_node": n_gpu,
            "logger": ["console"],
            "resume_mode": "auto" if args.resume else "disable",
        },
        "ray_kwargs": {
            "ray_init": {
                "num_cpus": max(n_gpu * 4, 32),
                # Ray sockets/IPC must be on a local filesystem — NFS (e.g. /newcpfs)
                # does not support Unix domain sockets (EOPNOTSUPP).
                "_temp_dir": "/dev/shm/ray_tmp",
                # PPL: cap object store so the default 200 GB reservation doesn't leave
                # insufficient headroom for the 7B reward model loaded on CPU.
                # Rollout batches are tiny (8 samples × 2048 tokens), 20 GB is ample.
                **({"object_store_memory": 20 * 1024 ** 3} if reward_type == "ppl" else {}),
            },
        },
    })

    vcfg = OmegaConf.merge(vcfg, overrides)

    # Save merged config for reproducibility
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "verl_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(vcfg))

    # Pre-download sentence-transformers before Ray workers start to avoid
    # concurrent HF cache writes causing "Unrecognized model" errors.
    from sentence_transformers import SentenceTransformer as _ST
    _ST(embed_model)
    del _ST

    # Pre-initialize CUDA before Ray forks workers to avoid glibc 2.28 TLS
    # assertion failures (_dl_allocate_tls_init: listp != NULL).
    import torch
    if torch.cuda.is_available():
        torch.ones(1, device="cuda:0")

    # Clear inductor cache to avoid UnpicklingError from truncated files left
    # by a previous run killed mid-write (e.g. disk-full crash).
    _inductor_cache = os.environ["TORCHINDUCTOR_CACHE_DIR"]
    shutil.rmtree(_inductor_cache, ignore_errors=True)
    Path(_inductor_cache).mkdir(parents=True, exist_ok=True)

    # Launch verl PPO/GRPO trainer
    from verl.trainer.main_ppo import run_ppo
    run_ppo(vcfg)

    # Merge sharded FSDP checkpoint → HF model at output_dir/final
    final_dir = output_dir / "final"
    # Check for the merged weights, not just the directory (a failed previous
    # attempt may have created an empty final/ directory).
    if not (final_dir / "config.json").exists():
        ckpt_dirs = sorted(output_dir.glob("global_step_*"))
        if ckpt_dirs:
            last_ckpt = ckpt_dirs[-1]
            # verl saves FSDP shards under global_step_N/actor/, not directly
            # under global_step_N/.  The merger's --local_dir must point at the
            # actor subdirectory where the rank shards live.
            actor_dir = last_ckpt / "actor"
            hf_subdir = actor_dir / "huggingface"
            if not (hf_subdir / "config.json").exists():
                hf_subdir.mkdir(parents=True, exist_ok=True)
                _config_src = (
                    sft_ckpt if (sft_ckpt / "config.json").exists() else Path(base_model)
                )
                for _f in _config_src.glob("*.json"):
                    shutil.copy2(_f, hf_subdir / _f.name)
                for _f in _config_src.glob("*.model"):
                    shutil.copy2(_f, hf_subdir / _f.name)
                for _f in _config_src.glob("tokenizer*"):
                    shutil.copy2(_f, hf_subdir / _f.name)
            import subprocess
            final_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    sys.executable, "-m", "verl.model_merger", "merge",
                    "--backend", "fsdp",
                    "--local_dir", str(actor_dir),
                    "--target_dir", str(final_dir),
                ],
                cwd=_REPO_ROOT, check=False,
            )


if __name__ == "__main__":
    main()
