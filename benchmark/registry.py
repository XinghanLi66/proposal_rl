"""
Central run registry — every pipeline appends one JSON line here on start.

Format: runs/benchmark/REGISTRY.jsonl
Each line is a RunRecord (see below). The dashboard reads this file to
discover all runs without having to scan the filesystem.

Thread/process-safe: uses fcntl.flock for exclusive append on Linux/NFS.
"""
from __future__ import annotations

import fcntl
import json
import re
import socket
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REGISTRY_FILENAME = "REGISTRY.jsonl"


@dataclass
class RunRecord:
    run_id: str
    run_dir: str           # absolute path, accessible from any NFS-mounted machine
    task: str              # e.g. "dl_lr_schedule"
    subtask: str           # e.g. "resnet20-cifar10"
    model_label: str       # e.g. "exp09_top_k_refs_sft_rl" (timestamp stripped)
    checkpoint: Optional[str]
    is_api: bool
    api_model: Optional[str]
    n_samples: int
    started_at: str        # ISO 8601 UTC
    machine: str           # hostname
    strategy: str = ""     # prompt strategy, e.g. "top_k_refs"

    @property
    def short_label(self) -> str:
        return self.model_label

    @classmethod
    def from_dict(cls, d: dict) -> "RunRecord":
        return cls(
            run_id=d.get("run_id", ""),
            run_dir=d.get("run_dir", ""),
            task=d.get("task", ""),
            subtask=d.get("subtask", ""),
            model_label=d.get("model_label", ""),
            checkpoint=d.get("checkpoint"),
            is_api=d.get("is_api", False),
            api_model=d.get("api_model"),
            n_samples=d.get("n_samples", 0),
            started_at=d.get("started_at", ""),
            machine=d.get("machine", ""),
            strategy=d.get("strategy", ""),
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ── Writing ───────────────────────────────────────────────────────────────────

def register_run(runs_dir: Path, config) -> None:
    """Append a RunRecord to REGISTRY.jsonl. Safe for concurrent multi-machine use."""
    entry = RunRecord(
        run_id=config.run_id,
        run_dir=str(config.run_dir),
        task=config.task_name,
        subtask=getattr(config, "subtask", ""),
        model_label=_extract_model_label(config),
        checkpoint=config.checkpoint,
        is_api=config.is_api,
        api_model=config.api_model if config.is_api else None,
        n_samples=config.n_samples,
        started_at=datetime.now(timezone.utc).isoformat(),
        machine=socket.gethostname(),
        strategy=getattr(config, "strategy", ""),
    )
    path = _registry_path(runs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry.to_dict()) + "\n"
    try:
        with open(path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(line)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except Exception:
        pass  # Non-critical — dashboard falls back to filesystem scan


def _extract_model_label(config) -> str:
    if config.is_api:
        return config.api_model or "api"
    if config.checkpoint:
        p = Path(config.checkpoint)
        for part in p.parts:
            if part == "rl":
                break
        # Walk up to find rl/ then take its parent
        for i, part in enumerate(p.parts):
            if part == "rl" and i > 0:
                return p.parts[i - 1]
    return config.run_id


def _strip_timestamp(name: str) -> str:
    """Strip _YYYYMMDD_HHMMSS suffix produced by training scripts."""
    return re.sub(r"_\d{8}_\d{6}$", "", name)


# ── Reading ───────────────────────────────────────────────────────────────────

def load_registry(runs_dir: Path) -> list[RunRecord]:
    """Read all run records. Skips malformed lines silently."""
    path = _registry_path(runs_dir)
    if not path.exists():
        return []
    records: list[RunRecord] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(RunRecord.from_dict(json.loads(line)))
                    except Exception:
                        pass
    except Exception:
        pass
    return records


def backfill_registry(runs_dir: Path) -> int:
    """
    Scan runs/benchmark/*/config.json for runs not yet in the registry
    and append them. Returns number of entries added.
    Useful for runs started before the registry existed.
    """
    existing_ids = {r.run_id for r in load_registry(runs_dir)}
    bench_dir = runs_dir / "benchmark"
    if not bench_dir.exists():
        return 0
    added = 0
    for cfg_path in sorted(bench_dir.glob("*/config.json")):
        try:
            d = json.loads(cfg_path.read_text())
            run_id = d.get("run_id", "")
            if not run_id or run_id in existing_ids:
                continue
            run_dir = cfg_path.parent

            # Build a minimal RunRecord from config.json
            checkpoint = d.get("checkpoint")
            is_api = d.get("is_api", False)
            api_model = d.get("api_model", "")
            if is_api:
                label = api_model or "api"
            elif checkpoint:
                parts = Path(checkpoint).parts
                label = ""
                for i, p in enumerate(parts):
                    if p == "rl" and i > 0:
                        label = parts[i - 1]
                        break
            else:
                label = run_id

            entry = RunRecord(
                run_id=run_id,
                run_dir=str(run_dir),
                task=d.get("task_name", ""),
                subtask=d.get("subtask", ""),
                model_label=label,
                checkpoint=checkpoint,
                is_api=is_api,
                api_model=api_model if is_api else None,
                n_samples=d.get("n_samples", 0),
                started_at=cfg_path.stat().st_mtime.__str__(),
                machine="(backfilled)",
            )
            path = _registry_path(runs_dir)
            with open(path, "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(entry.to_dict()) + "\n")
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
            existing_ids.add(run_id)
            added += 1
        except Exception:
            pass
    return added


def _registry_path(runs_dir: Path) -> Path:
    return runs_dir / "benchmark" / REGISTRY_FILENAME


# ── Sorting helpers (used by dashboard) ──────────────────────────────────────

def model_sort_key(label: str) -> tuple:
    """Sort exp09 < exp10 < ... < claude-* (API models last)."""
    m = re.match(r"exp(\d+)", label)
    if m:
        return (0, int(m.group(1)), label)
    return (1, 0, label)
