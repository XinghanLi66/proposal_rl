#!/usr/bin/env python3
"""
Relaunch failed or stuck samples from an existing benchmark run.

ERROR samples: re-uses existing proposal.txt (skips generation) and runs
  a fresh worker phase.

GENERATING-stuck samples (--include-stuck): the pipeline died mid-generation.
  Resets these to pending and runs the full generation+worker pipeline.

Usage:
    # Relaunch only error samples (re-use existing proposals)
    python benchmark/relaunch_errors.py \\
        --run-dir runs/benchmark/dl_lr_schedule_resnet20-cifar10_b1702e64 \\
        --machines lxh_agent_3 --gpus 4-7

    # Also handle stuck-generating samples (full regeneration)
    python benchmark/relaunch_errors.py \\
        --run-dir runs/benchmark/dl_lr_schedule_resnet20-cifar10_b1702e64 \\
        --machines lxh_agent_3 --gpus 4-7 --include-stuck

    # Relaunch all run dirs under a parent directory
    python benchmark/relaunch_errors.py \\
        --sweep-dir runs/benchmark \\
        --filter dl_lr_schedule_resnet20 \\
        --machines lxh_agent_0,lxh_agent_1,lxh_agent_2,lxh_agent_3 --gpus 0-7 \\
        --include-stuck
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from benchmark.pipeline import BenchmarkConfig, BenchmarkPipeline, PipelineStep, SampleState
from benchmark.slot_pool import SlotPool
from benchmark.tasks import get_task


def _relaunch(run_dir: Path, slot_pool: SlotPool,
              include_stuck: bool = False) -> None:
    config = BenchmarkConfig.load(run_dir)
    task = get_task(config.task_name, subtask=config.subtask)

    # Find errored and/or stuck-generating samples
    error_indices = []
    stuck_indices = []
    for i in range(config.n_samples):
        state_f = run_dir / f"sample_{i:02d}" / "state.json"
        if not state_f.exists():
            continue
        d = json.loads(state_f.read_text())
        if d["step"] == "error":
            error_indices.append(i)
        elif d["step"] == "generating" and include_stuck:
            stuck_indices.append(i)

    relaunch_indices = error_indices + stuck_indices
    if not relaunch_indices:
        print("  No samples to relaunch.")
        return

    if error_indices:
        print(f"  Relaunching {len(error_indices)} error samples: {error_indices}")
    if stuck_indices:
        print(f"  Relaunching {len(stuck_indices)} stuck-generating samples: {stuck_indices}")

    # Build a minimal pipeline shell
    pipeline = BenchmarkPipeline(config, task, run_dir, slot_pool=slot_pool)
    # Load existing sample states
    for i in range(config.n_samples):
        s = SampleState.load(run_dir / f"sample_{i:02d}")
        if s is not None:
            pipeline._samples[i] = s

    threads = []

    # --- ERROR samples: re-use existing proposal, just re-run worker ---
    for i in error_indices:
        proposal_f = run_dir / f"sample_{i:02d}" / "proposal.txt"
        if not proposal_f.exists():
            print(f"  sample_{i:02d}: no proposal.txt — will regenerate instead")
            stuck_indices.append(i)
            continue

        proposal = proposal_f.read_text()
        record_f = run_dir / f"sample_{i:02d}" / "prompt_record.json"
        record = json.loads(record_f.read_text()) if record_f.exists() else {}

        # Reset workspace so worker starts fresh
        workspace = run_dir / f"sample_{i:02d}" / "workspace"
        if workspace.exists():
            shutil.rmtree(workspace)

        # Reset state to pending
        pipeline._samples[i].step = PipelineStep.PENDING
        pipeline._samples[i].error = None

        def _worker(idx=i, rec=record, prop=proposal):
            caller = f"sample_{idx:02d}"
            with slot_pool.slot(caller=caller) as slot:
                pipeline._run_worker(idx, rec, prop,
                                     gpu_device=slot.gpu, machine=slot.machine)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        threads.append(t)

    # --- STUCK-GENERATING samples: full generation + worker via pipeline ---
    for i in stuck_indices:
        # Clear any partial state so _generate_one can run cleanly
        for fname in ("proposal.txt", "prompt_record.json", "worker.log",
                      "worker_prompt.txt"):
            f = run_dir / f"sample_{i:02d}" / fname
            if f.exists():
                f.unlink(missing_ok=True)
        workspace = run_dir / f"sample_{i:02d}" / "workspace"
        if workspace.exists():
            shutil.rmtree(workspace)

        # Reset to pending so pipeline._run_all_from picks it up
        pipeline._samples[i].step = PipelineStep.PENDING
        pipeline._samples[i].error = None
        pipeline._samples[i].save(run_dir / f"sample_{i:02d}")

        def _full(idx=i):
            caller = f"sample_{idx:02d}"
            record, proposal = pipeline._generate_one(idx)
            if record is None or proposal is None:
                return
            with slot_pool.slot(caller=caller) as slot:
                pipeline._run_worker(idx, record, proposal,
                                     gpu_device=slot.gpu, machine=slot.machine)

        t = threading.Thread(target=_full, daemon=True)
        t.start()
        threads.append(t)

    t0 = time.time()
    done_prev = 0
    while any(t.is_alive() for t in threads):
        with pipeline._lock:
            done = sum(1 for i in error_indices
                       if pipeline._samples[i].step in (PipelineStep.DONE, PipelineStep.ERROR))
        if done != done_prev:
            print(f"  [{time.strftime('%H:%M:%S')}] {done}/{len(error_indices)} done  "
                  f"elapsed={time.time()-t0:.0f}s")
            done_prev = done
        time.sleep(15)

    for t in threads:
        t.join()

    # Summary
    print("\nRelaunch results:")
    passed, failed, err = [], [], []
    for i in error_indices:
        s = pipeline._samples[i]
        if s.step == PipelineStep.DONE:
            info = f"improvement={s.improvement} passed={s.passed}"
            if s.passed:
                passed.append((i, info))
            else:
                failed.append((i, info))
        else:
            err.append((i, s.error))

    print(f"  PASS ({len(passed)}): {[x[0] for x in passed]}")
    for i, r in passed:
        print(f"    sample_{i:02d}: {r}")
    print(f"  FAIL ({len(failed)}): {[x[0] for x in failed]}")
    for i, r in failed:
        print(f"    sample_{i:02d}: {r}")
    print(f"  ERROR ({len(err)}): {[x[0] for x in err]}")
    for i, e in err:
        print(f"    sample_{i:02d}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir",
                       help="Path to a single existing benchmark run directory")
    group.add_argument("--sweep-dir",
                       help="Parent directory; relaunch all matching subdirs")
    parser.add_argument("--filter", default="",
                        help="Only process subdirs whose name contains this string "
                             "(used with --sweep-dir)")
    parser.add_argument("--machines", required=True,
                        help="Comma-separated SSH aliases for worker machines")
    parser.add_argument("--gpus", default="0-7",
                        help="GPU range or list, e.g. '4-7' or '0,1,2,3'")
    parser.add_argument("--log-file", default=None,
                        help="Optional path for slot pool log")
    parser.add_argument("--include-stuck", action="store_true",
                        help="Also relaunch samples stuck in 'generating' state "
                             "(full generation+worker)")
    args = parser.parse_args()

    machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    gpu_spec = args.gpus.strip()
    if "-" in gpu_spec and "," not in gpu_spec:
        lo, hi = gpu_spec.split("-")
        gpus = list(range(int(lo), int(hi) + 1))
    else:
        gpus = [int(g) for g in gpu_spec.split(",")]

    if args.run_dir:
        run_dirs = [Path(args.run_dir).resolve()]
    else:
        sweep_dir = Path(args.sweep_dir).resolve()
        run_dirs = sorted([
            d for d in sweep_dir.iterdir()
            if d.is_dir() and (not args.filter or args.filter in d.name)
            and (d / "config.json").exists()
        ])
        print(f"Found {len(run_dirs)} run dirs under {sweep_dir} (filter={args.filter!r})")

    log_file = args.log_file
    if not log_file:
        base = run_dirs[0].parent if run_dirs else Path(".")
        log_file = str(base / "slot_pool_relaunch.log")

    slot_pool = SlotPool.from_machines(machines, gpus, log_file=log_file)
    print(f"Slot pool: {slot_pool.total} slots ({machines} × GPU{gpus})  log → {log_file}")

    if len(run_dirs) == 1:
        # Single run: _relaunch is fine (its internal threads share the slot pool)
        run_dir = run_dirs[0]
        if not run_dir.exists():
            print(f"Run dir not found: {run_dir}")
            sys.exit(1)
        print(f"\n=== {run_dir.name} ===")
        _relaunch(run_dir, slot_pool, include_stuck=args.include_stuck)
    else:
        # Multiple runs: launch _relaunch for each in a thread so all share the slot pool
        # concurrently (slot pool gates actual GPU usage).
        run_threads: list[threading.Thread] = []
        for run_dir in run_dirs:
            if not run_dir.exists():
                print(f"Run dir not found: {run_dir}")
                continue
            t = threading.Thread(
                target=_relaunch,
                args=(run_dir, slot_pool),
                kwargs={"include_stuck": args.include_stuck},
                daemon=True,
            )
            t.start()
            run_threads.append(t)

        # Progress monitor
        t0 = time.time()
        while any(t.is_alive() for t in run_threads):
            n_done = sum(1 for rd in run_dirs
                         if (rd / "config.json").exists() and
                         all(
                             json.loads((rd / f"sample_{i:02d}" / "state.json").read_text()).get("step")
                             in ("done", "error")
                             for i in range(
                                 json.loads((rd / "config.json").read_text()).get("n_samples", 20))
                             if (rd / f"sample_{i:02d}" / "state.json").exists()
                         ))
            n_pass = n_fail = n_err = 0
            for rd in run_dirs:
                for sf in rd.glob("sample_*/state.json"):
                    try:
                        d = json.loads(sf.read_text())
                        if d["step"] == "done":
                            if d.get("passed"):
                                n_pass += 1
                            else:
                                n_fail += 1
                        elif d["step"] == "error":
                            n_err += 1
                    except Exception:
                        pass
            print(
                f"\r[{time.strftime('%H:%M:%S')}]  "
                f"elapsed={time.time()-t0:.0f}s  "
                f"pass={n_pass} fail={n_fail} err={n_err}   ",
                end="", flush=True,
            )
            time.sleep(30)
        print()
        for t in run_threads:
            t.join()


if __name__ == "__main__":
    main()
