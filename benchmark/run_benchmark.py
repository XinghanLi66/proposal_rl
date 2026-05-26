#!/usr/bin/env python3
"""
Single-run benchmark launcher for MLS-Bench tasks.

Generates proposals from a checkpoint (or Claude API) and runs them through the
BenchmarkPipeline on a single MLS-Bench task.

Usage:
  # Local checkpoint, single task/subtask
  python benchmark/run_benchmark.py \\
    --checkpoint runs/exps/exp09_.../rl/final \\
    --task dl_lr_schedule --subtask resnet20-cifar10 --n-samples 20

  # Claude API baseline
  python benchmark/run_benchmark.py \\
    --api-model claude-opus-4-6 \\
    --task dl_lr_schedule --n-samples 20

  # Full sweep (all checkpoints × all strategies)
  python benchmark/sweep.py

  # View results
  python benchmark/report.py --task dl_lr_schedule --sort pass_at_5
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import yaml

from benchmark.pipeline import BenchmarkConfig, BenchmarkPipeline, compute_summary
from benchmark.tasks import get_task, REGISTRY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="Path to local model checkpoint directory")
    src.add_argument("--api-model", help="Claude model ID (e.g. claude-opus-4-6)")

    parser.add_argument("--task", required=True, choices=list(REGISTRY),
                        help="Benchmark task name")
    parser.add_argument("--subtask", default="",
                        help="Subtask spec (e.g. resnet20-cifar10); default = task default")
    parser.add_argument("--strategy", default="full_refs",
                        help="Prompt-builder strategy")
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--gpu", default="0", help="CUDA device index for eval")
    parser.add_argument("--machine", default="", help="SSH host for remote execution")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output", default=None,
                        help="Run directory (default: runs/benchmark/<label>_<task>/)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--dry-run", action="store_true",
                        help="Build prompts and print; do not run workers")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    runs_dir = Path(cfg["runs_dir"])

    task = get_task(args.task, args.subtask or None)
    print(f"Task: {task.name}  subtask: {args.subtask or '(default)'}  "
          f"baseline={task.baseline_metric():.4f}  threshold={task.pass_threshold:.4f}")

    if args.checkpoint:
        label = Path(args.checkpoint).parent.parent.name
    else:
        label = args.api_model

    run_dir = Path(args.output) if args.output else (
        runs_dir / "benchmark" / f"{label}_{args.task}"
    )

    config = BenchmarkConfig(
        task_name=args.task,
        checkpoint=args.checkpoint,
        strategy=args.strategy,
        n_samples=args.n_samples,
        run_id=str(uuid.uuid4())[:8],
        gpu_device=args.gpu,
        is_api=args.api_model is not None,
        api_model=args.api_model or "",
        run_dir=str(run_dir),
        subtask=args.subtask,
        machine=args.machine,
        temperature=args.temperature,
    )

    if args.dry_run:
        print(f"[dry-run] would write run to: {run_dir}")
        print(f"  config: {json.dumps(vars(config) if hasattr(config, '__dict__') else {}, indent=2)}")
        return

    pipeline = BenchmarkPipeline(config, task, run_dir)
    pipeline.start()
    pipeline.wait()

    summary = compute_summary(pipeline.samples, task)
    print(f"\n{'='*60}")
    print(f"Results: {summary['n_passed']}/{summary['n_done']} passed  "
          f"(errors: {summary['n_errors']})")
    print(f"  pass@1={summary['pass_at_1']:.3f}  "
          f"pass@3={summary.get('pass_at_3', float('nan')):.3f}  "
          f"pass@5={summary.get('pass_at_5', float('nan')):.3f}")
    print(f"  mean_improvement={summary['mean_improvement']}")
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
