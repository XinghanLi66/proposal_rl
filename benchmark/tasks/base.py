from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path


class AbstractBenchmarkTask(ABC):
    """
    A benchmark task defines a fixed problem, a baseline, and a scorer.

    The loyal worker receives `task_context()` alongside the proposal and
    must write `{"val_metric": <float>}` to `result.json` in the workspace.
    The worker PASSES if `improvement(result) >= pass_threshold`.
    """

    name: str = ""

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Human-readable name of the metric (e.g. 'val_bpc', 'val_acc')."""

    @property
    @abstractmethod
    def lower_is_better(self) -> bool:
        """True if a lower metric value is better (e.g. loss, bpc)."""

    @property
    @abstractmethod
    def pass_threshold(self) -> float:
        """Minimum improvement over baseline to count as 'pass'."""

    @abstractmethod
    def baseline_metric(self) -> float:
        """Return the pre-computed baseline metric value."""

    def task_context(self, train_iters: int | None = None) -> str:
        """
        Full task description + baseline code, formatted as a string to be
        injected into the loyal worker's prompt.
        Subclasses may accept train_iters to speed up smoke tests.
        """
        raise NotImplementedError

    def setup_workspace(self, workspace: Path, python_bin: str = "python",
                        train_iters: int | None = None) -> None:
        """Copy required files (baseline code, dataset) into the worker workspace."""
        raise NotImplementedError

    def passed(self, metric: float) -> bool:
        baseline = self.baseline_metric()
        if self.lower_is_better:
            return (baseline - metric) >= self.pass_threshold
        else:
            return (metric - baseline) >= self.pass_threshold

    def improvement(self, metric: float) -> float:
        """Signed improvement over baseline (positive = better)."""
        if self.lower_is_better:
            return self.baseline_metric() - metric
        else:
            return metric - self.baseline_metric()
