"""MLS-Bench task adapters for the proposal RL benchmark pipeline.

Each task wraps one MLS-Bench research question. The worker receives the
editable region (a single function) and must implement the proposal's approach
there. Evaluation is run directly via the conda env (no Apptainer/SLURM).
"""
from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path
from typing import ClassVar

from benchmark.tasks.base import AbstractBenchmarkTask

_MLS_BENCH_ROOT = Path("/newcpfs/lxh/MLS-Bench")
_THIS_DIR = Path(__file__).parent
_BENCHMARK_DIR = Path(__file__).parent.parent


# ── Subtask registry ──────────────────────────────────────────────────────────

# Full set of subtasks per task (label → eval config).
# set active_subtasks in the concrete class to select which ones run.
_DL_LR_SCHEDULE_SUBTASKS: dict[str, dict] = {
    "resnet20-cifar10":   {"arch": "resnet20",    "dataset": "cifar10",  "epochs": 200},
    "resnet56-cifar100":  {"arch": "resnet56",    "dataset": "cifar100", "epochs": 200},
    "mobilenetv2-fmnist": {"arch": "mobilenetv2", "dataset": "fmnist",   "epochs": 200},
}

_DL_ACTIVATION_FUNCTION_SUBTASKS: dict[str, dict] = {
    "resnet20-cifar10":   {"arch": "resnet20",    "dataset": "cifar10",  "epochs": 200},
    "vgg16bn-cifar100":   {"arch": "vgg16bn",     "dataset": "cifar100", "epochs": 200},
    "mobilenetv2-fmnist": {"arch": "mobilenetv2", "dataset": "fmnist",   "epochs": 200},
}

_CV_DATA_AUGMENTATION_SUBTASKS: dict[str, dict] = {
    "resnet20-cifar10":   {"arch": "resnet20",    "dataset": "cifar10",  "epochs": 200},
    "resnet56-cifar100":  {"arch": "resnet56",    "dataset": "cifar100", "epochs": 200},
    "mobilenetv2-fmnist": {"arch": "mobilenetv2", "dataset": "fmnist",   "epochs": 200},
}


# ── Base class ────────────────────────────────────────────────────────────────

class MLSBenchTask(AbstractBenchmarkTask):
    """Base for MLS-Bench tasks. Subclasses set task-specific constants."""

    mls_task_name: ClassVar[str] = ""
    pkg_name: ClassVar[str] = ""
    reference_baseline: ClassVar[str] = ""
    active_subtasks: ClassVar[list[str]] = []
    all_subtask_specs: ClassVar[dict[str, dict]] = {}

    # Per-subtask baseline values keyed by label (from leaderboard.csv, seed=42)
    _baseline_values: ClassVar[dict[str, float]] = {}

    # Editable region in the template (1-indexed, inclusive)
    edit_file: ClassVar[str] = ""       # relative to workspace, e.g. "pytorch-vision/custom_schedule.py"
    edit_start: ClassVar[int] = 0
    edit_end: ClassVar[int] = 0
    edit_template: ClassVar[Path | None] = None  # source template file

    python_bin: ClassVar[str] = "/newcpfs/lxh/miniconda3/envs/loongflow_ml/bin/python"

    @property
    def _task_dir(self) -> Path:
        return _MLS_BENCH_ROOT / "tasks" / self.mls_task_name

    @property
    def _pkg_dir(self) -> Path:
        return _MLS_BENCH_ROOT / "vendor" / "external_packages" / self.pkg_name

    @property
    def _data_root(self) -> Path:
        return _MLS_BENCH_ROOT / "vendor" / "data"

    def baseline_metric(self) -> float:
        return self._baseline_values[self.active_subtasks[0]]

    def eval_papers(self) -> list[str]:
        fp = _THIS_DIR / "mls_tasks" / self.name / "frontline_papers.txt"
        if fp.exists():
            return [l.strip() for l in fp.read_text().splitlines()
                    if l.strip() and not l.startswith("#")]
        return []

    def _read_editable_region(self) -> str:
        lines = self.edit_template.read_text().splitlines()
        return "\n".join(lines[self.edit_start - 1 : self.edit_end]) + "\n"

    def setup_workspace(self, workspace: Path, python_bin: str = "python",
                        train_iters: int | None = None) -> None:
        workspace.mkdir(parents=True, exist_ok=True)

        # 1. Create edited file directory and place the template
        edit_path = workspace / self.edit_file
        edit_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.edit_template, edit_path)

        # 2. Extract baseline editable region as worker starting point
        (workspace / "editable_region.py").write_text(self._read_editable_region())

        # 3. Copy run_mls_eval.py into workspace
        shutil.copy2(_BENCHMARK_DIR / "run_mls_eval.py", workspace / "run_mls_eval.py")

        # 4. Write run.sh
        subtasks_json = json.dumps(self.active_subtasks)
        py = self.python_bin
        data_root = str(self._data_root)
        template = str(self.edit_template)
        (workspace / "run.sh").write_text(
            f"#!/bin/bash\nset -e\nRESULT=\"${{1:-result.json}}\"\n"
            f"{py} run_mls_eval.py \\\n"
            f"  --editable editable_region.py \\\n"
            f"  --template \"{template}\" \\\n"
            f"  --edit-start {self.edit_start} \\\n"
            f"  --edit-end {self.edit_end} \\\n"
            f"  --subtasks '{subtasks_json}' \\\n"
            f"  --data-root \"{data_root}\" \\\n"
            f"  --python \"{py}\" \\\n"
            f"  --gpu \"${{CUDA_VISIBLE_DEVICES:-0}}\" \\\n"
            f"  --out-json \"$RESULT\"\n"
        )
        os.chmod(workspace / "run.sh", 0o755)

    def task_context(self, train_iters: int | None = None,
                     python_bin: str = "python") -> str:
        task_desc = (self._task_dir / "task_description.md").read_text()
        editable_content = self._read_editable_region()
        label = self.active_subtasks[0]
        baseline_acc = self._baseline_values[label]
        return (
            f"{task_desc}\n\n"
            f"**Reference baseline ({self.reference_baseline}, {label}):** "
            f"test_acc = {baseline_acc:.2f}%  (pass if ≥ {baseline_acc + self.pass_threshold:.2f}%)\n\n"
            f"## Editable Region\n\n"
            f"Modify **only** the function in `editable_region.py`.\n"
            f"Preserve the exact function signature.\n\n"
            f"Current content:\n```python\n{editable_content}```\n\n"
            f"Run evaluation with:\n```bash\nbash run.sh result.json\n```\n"
            f"This writes `{{\"val_metric\": <test_acc>, \"_sig\": \"...\"}}` to result.json."
        )


# ── Concrete task: dl-lr-schedule ─────────────────────────────────────────────

class DlLrScheduleTask(MLSBenchTask):
    """
    MLS-Bench dl-lr-schedule: design a learning rate schedule.

    Editable region: get_lr(epoch, total_epochs, base_lr, config) → float
    Primary evaluation: ResNet-20 on CIFAR-10, 200 epochs.
    """

    name = "dl_lr_schedule"
    mls_task_name = "dl-lr-schedule"
    pkg_name = "pytorch-vision"
    reference_baseline = "warmup_cosine"

    # Start with single fast subtask; extend active_subtasks for full evaluation
    active_subtasks: ClassVar[list[str]] = ["resnet20-cifar10"]
    all_subtask_specs: ClassVar[dict[str, dict]] = _DL_LR_SCHEDULE_SUBTASKS

    _baseline_values: ClassVar[dict[str, float]] = {
        "resnet20-cifar10":   92.71,
        "resnet56-cifar100":  72.43,
        "mobilenetv2-fmnist": 94.83,
    }

    edit_file = "pytorch-vision/custom_schedule.py"
    edit_start = 246
    edit_end = 269
    edit_template = _MLS_BENCH_ROOT / "tasks" / "dl-lr-schedule" / "edits" / "custom_template.py"

    @property
    def metric_name(self) -> str:
        return "test_acc"

    @property
    def lower_is_better(self) -> bool:
        return False

    @property
    def pass_threshold(self) -> float:
        return 0.30


# ── Concrete task: dl-activation-function ────────────────────────────────────

class DlActivationFunctionTask(MLSBenchTask):
    """
    MLS-Bench dl-activation-function: design a custom activation function.

    Editable region: CustomActivation(nn.Module) with forward(x) method.
    Reference baseline: GELU (locally replicated: 92.97% on resnet20-cifar10, seed=42).
    Pass threshold: +0.30pp over GELU → ≥ 93.27%.
    Primary evaluation: ResNet-20 on CIFAR-10, 200 epochs.
    """

    name = "dl_activation_function"
    mls_task_name = "dl-activation-function"
    pkg_name = "pytorch-vision"
    reference_baseline = "gelu"

    active_subtasks: ClassVar[list[str]] = ["resnet20-cifar10"]
    all_subtask_specs: ClassVar[dict[str, dict]] = _DL_ACTIVATION_FUNCTION_SUBTASKS

    # resnet20-cifar10: locally replicated on H800 (seed=42) = 92.97%
    # Leaderboard shows 93.11% on different hardware; 0.14pp gap is CUDA non-determinism.
    # Using locally replicated value so pass threshold is consistent with our eval environment.
    _baseline_values: ClassVar[dict[str, float]] = {
        "resnet20-cifar10":   92.97,
        "vgg16bn-cifar100":   71.38,
        "mobilenetv2-fmnist": 94.75,
    }

    edit_file = "pytorch-vision/custom_activation.py"
    edit_start = 32
    edit_end = 49
    edit_template = _MLS_BENCH_ROOT / "tasks" / "dl-activation-function" / "edits" / "custom_template.py"

    @property
    def metric_name(self) -> str:
        return "test_acc"

    @property
    def lower_is_better(self) -> bool:
        return False

    @property
    def pass_threshold(self) -> float:
        # Published activations locally: GELU 92.84–92.97% (2 runs), SiLU 92.50%, Mish 92.65%.
        # +0.30pp sits 0.30pp above our local GELU max and 0.16pp above MLS-Bench leaderboard GELU
        # (93.11%) — no published activation function should pass.
        return 0.30


# ── Concrete task: cv-data-augmentation ──────────────────────────────────────

class CvDataAugmentationTask(MLSBenchTask):
    """
    MLS-Bench cv-data-augmentation: design a training data augmentation pipeline.

    Editable region: build_train_transform(config) → transforms.Compose
    Reference baseline: Cutout (locally replicated: 93.67% on resnet20-cifar10, seed=42).
    Pass threshold: +0.20pp over Cutout → ≥ 93.87%.
    Primary evaluation: ResNet-20 on CIFAR-10, 200 epochs.

    Local replication of all published baselines (H800, seed=42):
      standard (RandomCrop+HFlip): 92.54%
      cutout:        93.67%  ← reference
      trivialaugment: 93.57%
      randaugment:   93.20%
    """

    name = "cv_data_augmentation"
    mls_task_name = "cv-data-augmentation"
    pkg_name = "pytorch-vision"
    reference_baseline = "cutout"

    active_subtasks: ClassVar[list[str]] = ["resnet20-cifar10"]
    all_subtask_specs: ClassVar[dict[str, dict]] = _CV_DATA_AUGMENTATION_SUBTASKS

    # resnet20-cifar10: locally replicated Cutout (seed=42) = 93.67% — best published method.
    # Others are leaderboard values (not the primary reference subtask).
    _baseline_values: ClassVar[dict[str, float]] = {
        "resnet20-cifar10":   93.67,
        "resnet56-cifar100":  74.54,
        "mobilenetv2-fmnist": 94.72,
    }

    edit_file = "pytorch-vision/custom_augment.py"
    edit_start = 246
    edit_end = 275
    edit_template = _MLS_BENCH_ROOT / "tasks" / "cv-data-augmentation" / "edits" / "custom_template.py"

    @property
    def metric_name(self) -> str:
        return "test_acc"

    @property
    def lower_is_better(self) -> bool:
        return False

    @property
    def pass_threshold(self) -> float:
        # Cutout (best published, local) = 93.67%; +0.20pp ensures no published method passes:
        #   cutout 93.67%, trivialaugment 93.57%, randaugment 93.20% — all below 93.87%.
        return 0.20
