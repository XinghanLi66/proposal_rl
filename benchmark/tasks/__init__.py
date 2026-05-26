from benchmark.tasks.base import AbstractBenchmarkTask
from benchmark.tasks.mls_bench import (
    CvDataAugmentationTask,
    DlActivationFunctionTask,
    DlLrScheduleTask,
)

REGISTRY: dict[str, type[AbstractBenchmarkTask]] = {
    "dl_lr_schedule": DlLrScheduleTask,
    "dl_activation_function": DlActivationFunctionTask,
    "cv_data_augmentation": CvDataAugmentationTask,
}

# Subtasks available per task (for sweep scripts and the dashboard).
SUBTASKS: dict[str, list[str]] = {
    "dl_lr_schedule": ["resnet20-cifar10", "resnet56-cifar100", "mobilenetv2-fmnist"],
    "dl_activation_function": ["resnet20-cifar10", "vgg16bn-cifar100", "mobilenetv2-fmnist"],
    "cv_data_augmentation": ["resnet20-cifar10", "resnet56-cifar100", "mobilenetv2-fmnist"],
}


def get_task(name: str, subtask: str | None = None) -> AbstractBenchmarkTask:
    """
    Instantiate a task by name.  If subtask is given, override active_subtasks
    so the task runs that specific evaluation setting.
    """
    if name not in REGISTRY:
        raise ValueError(f"Unknown benchmark task: {name!r}. Available: {list(REGISTRY)}")
    task = REGISTRY[name]()
    if subtask:
        available = getattr(task, "all_subtask_specs", {})
        if available and subtask not in available:
            raise ValueError(
                f"Unknown subtask {subtask!r} for task {name!r}. "
                f"Available: {list(available)}"
            )
        task.active_subtasks = [subtask]  # instance-level override shadows class default
    return task
