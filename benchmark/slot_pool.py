"""
SlotPool — thread-safe pool of (machine, gpu) compute slots.

Each slot represents one GPU on one machine.  Worker threads call
``acquire()`` (blocks until a slot is free) and ``release(slot)`` when done.

Designed to be shared across multiple BenchmarkPipeline instances so that
the total concurrency is bounded globally, not per-pipeline.

Interface is intentionally backend-agnostic: a future DLC/Slurm backend
can subclass SlotPool and override acquire/release without changing pipeline
code.

A log_file path may be provided to record every acquire/release event with
timestamps, caller info, and current pool utilisation.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


# ── Slot descriptor ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Slot:
    machine: str   # SSH alias (empty string = local)
    gpu: str       # CUDA_VISIBLE_DEVICES value, e.g. "0"
    label: str = ""  # human-readable name, e.g. "M0-GPU0"

    def __str__(self) -> str:
        return self.label or f"{self.machine or 'local'}:GPU{self.gpu}"


# ── Base pool ─────────────────────────────────────────────────────────────────

class SlotPool:
    """
    Thread-safe blocking pool of compute slots.

    Usage::

        pool = SlotPool.from_machines(["lxh_agent_0", "lxh_agent_1"], gpus=range(8))
        slot = pool.acquire()          # blocks until a slot is free
        try:
            run_job(slot.machine, slot.gpu)
        finally:
            pool.release(slot)

    Or as a context manager per slot::

        with pool.slot() as slot:
            run_job(slot.machine, slot.gpu)
    """

    def __init__(self, slots: list[Slot], log_file: Path | str | None = None) -> None:
        self._all: list[Slot] = list(slots)
        self._free: list[Slot] = list(slots)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._in_use: set[Slot] = set()
        self._log_file: Path | None = Path(log_file) if log_file else None
        self._log_lock = threading.Lock()
        if self._log_file:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log(f"INIT total={len(slots)} slots: {[str(s) for s in slots]}")

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if not self._log_file:
            return
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._log_lock:
            with open(self._log_file, "a") as f:
                f.write(f"[{ts}] {msg}\n")

    # ── Construction helpers ──────────────────────────────────────────────────

    @classmethod
    def from_machines(
        cls,
        machines: list[str],
        gpus: range | list[int | str],
        log_file: Path | str | None = None,
    ) -> "SlotPool":
        """Build a pool for every (machine, gpu) combination."""
        slots = [
            Slot(machine=m, gpu=str(g),
                 label=f"{m}:GPU{g}" if m else f"local:GPU{g}")
            for m in machines
            for g in gpus
        ]
        return cls(slots, log_file=log_file)

    @classmethod
    def local(cls, gpus: range | list[int | str],
              log_file: Path | str | None = None) -> "SlotPool":
        """Pool of local GPUs only."""
        return cls.from_machines([""], gpus, log_file=log_file)

    # ── Acquire / release ─────────────────────────────────────────────────────

    def acquire(self, caller: str = "") -> Slot:
        """Block until a slot is available, then return it."""
        waited = False
        with self._cond:
            if not self._free:
                waited = True
                self._log(
                    f"WAIT  caller={caller!r} free={len(self._free)} "
                    f"busy={len(self._in_use)}/{len(self._all)}"
                )
            while not self._free:
                self._cond.wait()
            slot = self._free.pop(0)
            self._in_use.add(slot)
        self._log(
            f"ACQUIRE slot={slot} caller={caller!r} waited={waited} "
            f"free={len(self._free)} busy={len(self._in_use)}/{len(self._all)}"
        )
        return slot

    def release(self, slot: Slot, caller: str = "") -> None:
        """Return a slot to the pool."""
        with self._cond:
            self._in_use.discard(slot)
            self._free.append(slot)
            self._cond.notify()
        self._log(
            f"RELEASE slot={slot} caller={caller!r} "
            f"free={len(self._free)} busy={len(self._in_use)}/{len(self._all)}"
        )

    def slot(self, caller: str = "") -> "_SlotContext":
        """Context manager: acquire on enter, release on exit."""
        return _SlotContext(self, caller=caller)

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def total(self) -> int:
        return len(self._all)

    @property
    def free_count(self) -> int:
        with self._lock:
            return len(self._free)

    @property
    def busy_count(self) -> int:
        with self._lock:
            return len(self._in_use)

    def status(self) -> dict:
        with self._lock:
            return {
                "total": len(self._all),
                "free": len(self._free),
                "busy": len(self._in_use),
                "busy_slots": [str(s) for s in self._in_use],
            }

    def __repr__(self) -> str:
        st = self.status()
        return f"SlotPool(total={st['total']}, free={st['free']}, busy={st['busy']})"


class _SlotContext:
    def __init__(self, pool: SlotPool, caller: str = "") -> None:
        self._pool = pool
        self._slot: Slot | None = None
        self._caller = caller

    def __enter__(self) -> Slot:
        self._slot = self._pool.acquire(caller=self._caller)
        return self._slot

    def __exit__(self, *_) -> None:
        if self._slot is not None:
            self._pool.release(self._slot, caller=self._caller)
            self._slot = None


# ── Pre-built slot layouts ────────────────────────────────────────────────────

#: All 32 slots across the 4 DSW machines (M0–M3 × GPU 0–7)
ALL_DSW_MACHINES = ["lxh_agent_0", "lxh_agent_1", "lxh_agent_2", "lxh_agent_3"]
ALL_DSW_GPUS = list(range(8))


def default_pool() -> SlotPool:
    """32-slot pool covering all 4 DSW machines × 8 GPUs each."""
    return SlotPool.from_machines(ALL_DSW_MACHINES, ALL_DSW_GPUS)
