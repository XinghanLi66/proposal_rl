#!/usr/bin/env python3
"""
Full checkpoint × strategy sweep launcher.

Evaluates every checkpoint against every prompt strategy, assigning each run
to a dedicated machine:GPU slot.  Up to 32 pipelines run concurrently.

Slot layout: M0–M3 × GPU 0–7 = 32 slots, round-robin across runs.

Usage:
    # Latest exp09-exp17 checkpoints × 5 strategies = 45 runs (20 samples each)
    python benchmark/sweep.py

    # Subset by exp prefix
    python benchmark/sweep.py --checkpoints exp09,exp15

    # Single strategy
    python benchmark/sweep.py --strategies full_refs

    # Dry run — print assignment table, don't launch
    python benchmark/sweep.py --dry-run

    # Fewer samples for a quick smoke test
    python benchmark/sweep.py --n-samples 3 --dry-run
"""
from __future__ import annotations

import argparse
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from benchmark.checkpoints import CHECKPOINT_REGISTRY
from benchmark.pipeline import BenchmarkConfig, BenchmarkPipeline, PipelineStep
from benchmark.prompt_cache import ALL_STRATEGIES
from benchmark.slot_pool import Slot, SlotPool, default_pool
from benchmark.tasks import get_task, SUBTASKS

# ── Default slot layout ───────────────────────────────────────────────────────

_MACHINES = ["lxh_agent_0", "lxh_agent_1", "lxh_agent_2", "lxh_agent_3"]
_ALL_SLOTS: list[tuple[str, str]] = [
    (machine, str(gpu))
    for machine in _MACHINES
    for gpu in range(8)
]  # 32 slots — kept for dry-run display


# ── Run spec ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RunSpec:
    label: str       # display name from CHECKPOINT_REGISTRY
    checkpoint: str  # absolute path to rl/final
    strategy: str    # prompt strategy (one of ALL_STRATEGIES)
    subtask: str     # e.g. "resnet20-cifar10"
    # machine/gpu are no longer fixed per run — dynamically assigned by SlotPool
    # kept here only for the dry-run display table
    machine: str = ""
    gpu: str = ""


def _build_specs(
    selected_ckpts: list[tuple[str, str, str]],
    selected_strategies: list[str],
    subtask: str,
    exps_root: Path,
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for label, rel_path, _native_strategy in selected_ckpts:
        ckpt_path = exps_root / rel_path
        if not ckpt_path.exists():
            print(f"  [skip] {label} — not found: {ckpt_path}", flush=True)
            continue
        if not _checkpoint_usable(ckpt_path):
            print(f"  [skip] {label} — no model artifacts: {ckpt_path}", flush=True)
            continue
        for strategy in selected_strategies:
            specs.append(RunSpec(
                label=label,
                checkpoint=str(ckpt_path),
                strategy=strategy,
                subtask=subtask,
            ))
    return specs


def _checkpoint_usable(ckpt_path: Path) -> bool:
    """Return True when rl/final exists and has enough files to load as HF/LoRA."""
    if not ckpt_path.exists() or not ckpt_path.is_dir():
        return False
    if (ckpt_path / "config.json").exists() or (ckpt_path / "adapter_config.json").exists():
        return True
    if any(ckpt_path.glob("*.safetensors")):
        return True
    if any(ckpt_path.glob("pytorch_model*.bin")):
        return True
    return False


def _exp_id(entry: tuple[str, str, str]) -> str:
    m = re.search(r"(exp\d{2})", entry[1])
    return m.group(1) if m else entry[0].split()[0]


def _timestamp_key(entry: tuple[str, str, str]) -> str:
    exp_dir = Path(entry[1]).parts[0]
    m = re.search(r"_(\d{8}_\d{6})$", exp_dir)
    return m.group(1) if m else ""


def _latest_usable_per_exp(
    entries: list[tuple[str, str, str]],
    exps_root: Path,
) -> list[tuple[str, str, str]]:
    """Group registry entries by expNN and keep the newest usable checkpoint."""
    latest: dict[str, tuple[str, str, str]] = {}
    for entry in entries:
        _, rel_path, _ = entry
        ckpt_path = exps_root / rel_path
        if not _checkpoint_usable(ckpt_path):
            print(f"  [skip] {_exp_id(entry)} candidate has no usable final: {ckpt_path}", flush=True)
            continue
        exp = _exp_id(entry)
        prev = latest.get(exp)
        if prev is None or (_timestamp_key(entry), rel_path) > (_timestamp_key(prev), prev[1]):
            latest[exp] = entry
    return [latest[k] for k in sorted(latest, key=lambda x: int(x[3:]))]


def _parse_gpus(gpu_spec: str) -> list[int]:
    """Parse a GPU range/list like '0-7' or '0,1,3'."""
    gpu_spec = gpu_spec.strip()
    if "-" in gpu_spec and "," not in gpu_spec:
        lo, hi = gpu_spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(g.strip()) for g in gpu_spec.split(",") if g.strip()]


def _parse_explicit_slots(slots_spec: str) -> list[Slot]:
    """Parse semicolon-separated machine:gpu-spec entries."""
    slots: list[Slot] = []
    for chunk in slots_spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                f"Bad --slots entry {chunk!r}; expected machine:gpu-spec"
            )
        machine, gpu_spec = chunk.split(":", 1)
        machine = machine.strip()
        if not machine:
            raise ValueError(f"Bad --slots entry {chunk!r}; empty machine")
        for gpu in _parse_gpus(gpu_spec):
            slots.append(
                Slot(machine=machine, gpu=str(gpu), label=f"{machine}:GPU{gpu}")
            )
    if not slots:
        raise ValueError("--slots did not define any GPU slots")
    return slots


# ── Sweep runner ──────────────────────────────────────────────────────────────

def _run_one(spec: RunSpec, runs_dir: Path, n_samples: int,
             task_name: str, results: dict, lock: threading.Lock,
             run_sem: threading.Semaphore,
             slot_pool: SlotPool,
             temperature: float,
             gen_gpu: str = "0",
             gen_machine: str = "") -> None:
    """Launch one (checkpoint × strategy) pipeline run.

    run_sem limits concurrent pipelines (generation phase); workers acquire
    individual GPU slots from slot_pool dynamically.

    gen_gpu / gen_machine: the GPU and machine used for proposal generation
    (model inference).  Separate from the worker slots to avoid OOM when
    multiple sweeps run concurrently.
    """
    with run_sem:
        run_id = uuid.uuid4().hex[:8]
        run_dir = runs_dir / "benchmark" / f"{task_name}_{spec.subtask}_{run_id}"
        task = get_task(task_name, subtask=spec.subtask)
        cfg = BenchmarkConfig(
            task_name=task_name,
            checkpoint=spec.checkpoint,
            strategy=spec.strategy,
            n_samples=n_samples,
            run_id=run_id,
            gpu_device=gen_gpu,    # used for generation; slot_pool overrides for workers
            max_workers=1,         # ignored when slot_pool is active
            is_api=False,
            run_dir=str(run_dir),
            subtask=spec.subtask,
            machine=gen_machine,   # used for generation; slot_pool overrides for workers
            temperature=temperature,
        )
        pipeline = BenchmarkPipeline(cfg, task, run_dir, slot_pool=slot_pool)
        with lock:
            results[spec] = {"pipeline": pipeline, "run_dir": run_dir, "status": "running"}
        pipeline.start()
        pipeline.wait()
        with lock:
            results[spec]["status"] = "done"


def _progress_loop(results: dict, lock: threading.Lock,
                   total: int, stop: threading.Event) -> None:
    while not stop.is_set():
        with lock:
            n_running = sum(1 for v in results.values() if v["status"] == "running")
            n_done = sum(1 for v in results.values() if v["status"] == "done")
            n_pass = 0
            n_fail = 0
            n_err = 0
            for v in results.values():
                if v["status"] != "done":
                    continue
                for s in v["pipeline"].samples:
                    if s.step == PipelineStep.DONE:
                        if s.passed:
                            n_pass += 1
                        else:
                            n_fail += 1
                    elif s.step == PipelineStep.ERROR:
                        n_err += 1
        print(
            f"\r[{time.strftime('%H:%M:%S')}]  "
            f"runs: {n_done}/{total} done, {n_running} active  |  "
            f"samples: {n_pass} pass  {n_fail} fail  {n_err} error   ",
            end="", flush=True,
        )
        stop.wait(timeout=15)
    print()


def run_sweep(specs: list[RunSpec], runs_dir: Path,
              n_samples: int, max_parallel: int,
              task_name: str = "dl_lr_schedule",
              slot_pool: SlotPool | None = None,
              temperature: float = 0.7,
              gen_gpu: str = "0",
              gen_machine: str = "") -> dict:
    """
    Run all specs concurrently, sharing a single slot_pool for worker GPU assignment.

    max_parallel limits concurrent pipeline *generation* phases (CPU/model bound).
    Individual workers acquire GPU slots from slot_pool dynamically.

    gen_gpu / gen_machine: dedicated GPU/machine for proposal generation inference.
    """
    if slot_pool is None:
        slot_pool = default_pool()

    run_sem = threading.Semaphore(max_parallel)
    results: dict[RunSpec, dict] = {}
    lock = threading.Lock()
    stop_progress = threading.Event()

    threads = []
    for spec in specs:
        t = threading.Thread(
            target=_run_one,
            args=(spec, runs_dir, n_samples, task_name, results, lock,
                  run_sem, slot_pool, temperature, gen_gpu, gen_machine),
            daemon=True,
        )
        t.start()
        threads.append(t)

    progress_t = threading.Thread(
        target=_progress_loop,
        args=(results, lock, len(specs), stop_progress),
        daemon=True,
    )
    progress_t.start()

    for t in threads:
        t.join()

    stop_progress.set()
    progress_t.join(timeout=2)

    return results


# ── Summary table ─────────────────────────────────────────────────────────────

def _print_summary(results: dict[RunSpec, dict], strategies: list[str]) -> None:
    # Group by checkpoint label
    by_label: dict[str, dict[str, dict]] = {}
    for spec, info in results.items():
        by_label.setdefault(spec.label, {})[spec.strategy] = info

    col_w = 22
    header = f"{'Checkpoint':<42}" + "".join(f"{s[:col_w]:<{col_w+2}}" for s in strategies)
    print("\n" + header)
    print("-" * len(header))

    for label in sorted(by_label):
        row = f"{label:<42}"
        for strategy in strategies:
            info = by_label[label].get(strategy)
            if info is None:
                row += f"{'—':<{col_w+2}}"
                continue
            pipeline = info["pipeline"]
            samples = pipeline.samples
            done = [s for s in samples if s.step == PipelineStep.DONE]
            errs = [s for s in samples if s.step == PipelineStep.ERROR]
            passed = [s for s in done if s.passed]
            if done:
                mean_d = sum(s.improvement for s in done if s.improvement is not None) / len(done)
                sign = "+" if mean_d >= 0 else ""
                cell = f"{len(passed)}/{len(done)} ({sign}{mean_d:.2f}%)"
            else:
                cell = f"0/0 ({len(errs)} err)"
            row += f"{cell:<{col_w+2}}"
        print(row)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    all_subtasks = SUBTASKS.get("dl_lr_schedule", [])
    default_subtask = all_subtasks[0] if all_subtasks else "resnet20-cifar10"

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoints", default="all",
        help="Comma-separated exp prefixes (e.g. 'exp09,exp15') or 'all'",
    )
    parser.add_argument(
        "--strategies", default="all",
        help="Comma-separated strategy names or 'all'",
    )
    parser.add_argument(
        "--subtask", default=default_subtask,
        help=f"Single subtask to use for all runs (default: {default_subtask})",
    )
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--latest-only", action="store_true",
                        help="Keep only the latest usable rl/final per expNN.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Checkpoint sampling temperature (default: 0.7)")
    parser.add_argument("--task", default="dl_lr_schedule")
    parser.add_argument(
        "--max-parallel", type=int, default=len(_MACHINES),
        help=f"Max concurrent generation pipelines (default: {len(_MACHINES)}; "
             "workers are bounded by the slot pool size)",
    )
    parser.add_argument(
        "--machines", default=",".join(_MACHINES),
        help="Comma-separated SSH aliases for worker machines "
             f"(default: {','.join(_MACHINES)})",
    )
    parser.add_argument(
        "--gpus", default="0-7",
        help="GPU range or list for workers, e.g. '0-7', '0,1,2,3' (default: 0-7)",
    )
    parser.add_argument(
        "--slots", default="",
        help=(
            "Explicit semicolon-separated worker slots, overriding --machines/--gpus; "
            "example: 'lxh_agent_0:0-7;lxh_agent_1:1-7;lxh_agent_2:0-7'"
        ),
    )
    parser.add_argument("--runs-dir", default=None,
                        help="Root runs directory (default: <repo>/runs)")
    parser.add_argument("--gen-gpu", default="0",
                        help="Fallback generation GPU when no slot pool is used (default: 0).")
    parser.add_argument("--gen-machine", default="",
                        help="Fallback generation machine when no slot pool is used (default: local).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print run table and exit without launching")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir) if args.runs_dir else _REPO / "runs"
    exps_root = runs_dir / "exps"

    # Resolve checkpoints
    if args.checkpoints == "all":
        selected_ckpts = CHECKPOINT_REGISTRY
    else:
        prefixes = [p.strip() for p in args.checkpoints.split(",")]
        selected_ckpts = [
            entry for entry in CHECKPOINT_REGISTRY
            if any(entry[0].strip().startswith(p) for p in prefixes)
        ]
        if not selected_ckpts:
            print(f"No checkpoints matched prefixes: {prefixes}")
            print("Available labels:")
            for label, _, _ in CHECKPOINT_REGISTRY:
                print(f"  {label}")
            sys.exit(1)

    if args.latest_only:
        selected_ckpts = _latest_usable_per_exp(selected_ckpts, exps_root)
        if not selected_ckpts:
            print("No usable latest checkpoints found.")
            sys.exit(1)

    # Resolve strategies
    if args.strategies == "all":
        selected_strategies = ALL_STRATEGIES
    else:
        selected_strategies = [s.strip() for s in args.strategies.split(",")]
        unknown = [s for s in selected_strategies if s not in ALL_STRATEGIES]
        if unknown:
            print(f"Unknown strategies: {unknown}  Available: {ALL_STRATEGIES}")
            sys.exit(1)

    # Validate subtask
    if args.subtask not in all_subtasks:
        print(f"Unknown subtask {args.subtask!r}  Available: {all_subtasks}")
        sys.exit(1)

    specs = _build_specs(selected_ckpts, selected_strategies, args.subtask, exps_root)
    if not specs:
        print("No valid runs to launch.")
        sys.exit(1)

    # Build slot pool from explicit --slots, or from --machines / --gpus.
    slot_log = runs_dir / "slot_pool.log"
    slot_desc: str
    if args.slots.strip():
        try:
            slots = _parse_explicit_slots(args.slots)
        except ValueError as exc:
            print(str(exc))
            sys.exit(1)
        slot_pool = SlotPool(slots, log_file=slot_log)
        machines = sorted({slot.machine for slot in slots})
        slot_desc = f"{slot_pool.total} explicit slots"
    else:
        machines = [m.strip() for m in args.machines.split(",") if m.strip()]
        gpus = _parse_gpus(args.gpus)
        slot_pool = SlotPool.from_machines(machines, gpus, log_file=slot_log)
        slot_desc = f"{slot_pool.total} slots ({len(machines)} machines × {len(gpus)} GPUs)"

    n_total_samples = len(specs) * args.n_samples
    print(f"Sweep: {len(selected_ckpts)} checkpoints × {len(selected_strategies)} strategies "
          f"= {len(specs)} runs × {args.n_samples} samples = {n_total_samples} total  "
          f"[subtask: {args.subtask}]")
    print(f"Slot pool: {slot_desc}  log → {slot_log}")
    print("Generation: shared slot pool (generation releases a slot before worker eval)")
    print(f"Temperature: {args.temperature}")
    print(f"Max parallel pipelines: {args.max_parallel}  "
          f"(workers dynamically assigned from pool)")
    print()
    print(f"  {'Checkpoint':<42} {'Strategy':<26}")
    print(f"  {'-'*42} {'-'*26}")
    for spec in specs:
        print(f"  {spec.label:<42} {spec.strategy:<26}")
    print()

    if args.dry_run:
        print("Dry run — not launching.")
        return

    print("Launching…")
    results = run_sweep(specs, runs_dir, args.n_samples, args.max_parallel,
                        args.task, slot_pool=slot_pool,
                        temperature=args.temperature,
                        gen_gpu=args.gen_gpu, gen_machine=args.gen_machine)
    _print_summary(results, selected_strategies)
    print("\nDone. Run dashboard_tui.py to view results.")


if __name__ == "__main__":
    main()
