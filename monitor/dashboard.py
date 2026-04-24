#!/usr/bin/env python3
"""
Live terminal dashboard for the proposal_rl pipeline.
Run in a separate tmux pane:  python monitor/dashboard.py --runs-dir runs/

Shows:
  - Current pipeline stage
  - Data progress (fetch / build / CoT synthesis)
  - Training metrics (loss, FAS, LR) from training log files
  - GPU utilization (via nvidia-smi)
  - Recent log lines

Refreshes every 3 seconds.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box


# ---- Helpers ---------------------------------------------------------------

def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        result = subprocess.run(["wc", "-l", str(path)], capture_output=True, text=True)
        return int(result.stdout.split()[0])
    except Exception:
        return 0


def tail_lines(path: Path, n: int = 8) -> list[str]:
    if not path.exists():
        return []
    try:
        result = subprocess.run(["tail", f"-{n}", str(path)], capture_output=True, text=True)
        return result.stdout.strip().splitlines()
    except Exception:
        return []


def read_metrics(log_dir: Path) -> list[dict]:
    """Read training metrics from HF Trainer log JSON files."""
    metrics = []
    trainer_log = log_dir / "trainer_state.json"
    if trainer_log.exists():
        try:
            state = json.loads(trainer_log.read_text())
            history = state.get("log_history", [])
            metrics = [h for h in history if "loss" in h]
        except Exception:
            pass
    return metrics


def read_eval_summary(runs_dir: Path) -> dict | None:
    """Read the latest eval summary if available."""
    for pattern in ["grpo/final/eval_results/summary.json", "sft/final/eval_results/summary.json"]:
        p = runs_dir / pattern
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
    return None


def get_gpu_stats() -> list[dict]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 6:
                gpus.append({
                    "idx": parts[0], "name": parts[1][:12],
                    "util": parts[2], "mem_used": parts[3], "mem_total": parts[4], "temp": parts[5]
                })
        return gpus
    except Exception:
        return []


def detect_stage(runs_dir: Path) -> str:
    """Infer current pipeline stage from which files exist."""
    if (runs_dir / "grpo" / "final" / "config.json").exists():
        return "grpo_done"
    if (runs_dir / "grpo").exists() and any((runs_dir / "grpo").iterdir()):
        return "grpo"
    if (runs_dir / "sft" / "final" / "config.json").exists():
        return "sft_done"
    if (runs_dir / "sft").exists() and any((runs_dir / "sft").iterdir()):
        return "sft"
    if (runs_dir / "dataset" / "train_cot.jsonl").exists():
        return "cot_done"
    if (runs_dir / "dataset" / "train.jsonl").exists():
        return "building_cot"
    if (runs_dir / "dataset" / "refs_train.jsonl").exists():
        return "building_dataset"
    return "fetching_refs"


STAGE_LABELS = {
    "fetching_refs":     "[yellow]● Fetching references (S2 API)[/]",
    "building_dataset":  "[yellow]● Building dataset[/]",
    "building_cot":      "[yellow]● Synthesizing CoT (Claude API)[/]",
    "cot_done":          "[green]✓ CoT done[/] — waiting for training",
    "sft":               "[cyan]● SFT training[/]",
    "sft_done":          "[green]✓ SFT done[/] — GRPO pending",
    "grpo":              "[bright_cyan]● GRPO training[/]",
    "grpo_done":         "[bright_green]✓ GRPO done — evaluation pending[/]",
}


# ---- Panel builders --------------------------------------------------------

def make_data_panel(runs_dir: Path) -> Panel:
    dataset_dir = runs_dir / "dataset"
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("File", style="dim", width=24)
    table.add_column("Lines", justify="right")
    table.add_column("Size")

    files = [
        ("refs_train.jsonl", "Train refs"),
        ("refs_val.jsonl", "Val refs"),
        ("train.jsonl", "Train dataset"),
        ("val.jsonl", "Val dataset"),
        ("test.jsonl", "Test dataset"),
        ("train_cot.jsonl", "CoT train"),
    ]
    for fname, label in files:
        p = dataset_dir / fname
        if p.exists():
            sz = p.stat().st_size
            sz_str = f"{sz/1e6:.1f}M" if sz > 1e6 else f"{sz/1e3:.1f}K"
            table.add_row(label, str(count_lines(p)), sz_str)
        else:
            table.add_row(f"[dim]{label}[/]", "[dim]—[/]", "[dim]—[/]")

    return Panel(table, title="[bold]Data[/bold]", border_style="blue")


def make_training_panel(runs_dir: Path, stage: str) -> Panel:
    if stage in ("fetching_refs", "building_dataset", "building_cot", "cot_done"):
        return Panel(Text("Waiting for training to start...", style="dim"), title="[bold]Training[/bold]", border_style="magenta")

    active_dir = runs_dir / ("grpo" if "grpo" in stage else "sft")
    metrics = read_metrics(active_dir / "logs")

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Step", justify="right")
    table.add_column("Loss", justify="right")
    table.add_column("LR", justify="right")
    table.add_column("Epoch", justify="right")

    for m in metrics[-10:]:
        table.add_row(
            str(m.get("step", "")),
            f"{m.get('loss', m.get('train_loss', '')):.4f}" if "loss" in m or "train_loss" in m else "—",
            f"{m.get('learning_rate', 0):.2e}",
            f"{m.get('epoch', 0):.2f}",
        )

    return Panel(table, title=f"[bold]Training — {stage.upper()}[/bold]", border_style="magenta")


def make_eval_panel(runs_dir: Path) -> Panel:
    summary = read_eval_summary(runs_dir)
    if summary is None:
        return Panel(Text("No eval results yet.", style="dim"), title="[bold]Evaluation (FAS)[/bold]", border_style="green")

    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Metric", style="dim")
    table.add_column("Value", style="bold green")

    for k, v in summary.items():
        if k not in ("checkpoint", "split", "n_examples", "gen_time_s"):
            table.add_row(k, str(v))

    return Panel(table, title="[bold]Evaluation (FAS)[/bold]", border_style="green")


def make_gpu_panel() -> Panel:
    gpus = get_gpu_stats()
    if not gpus:
        return Panel(Text("nvidia-smi unavailable", style="dim"), title="[bold]GPUs[/bold]", border_style="yellow")

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("GPU", width=4)
    table.add_column("Name", width=12)
    table.add_column("Util%", justify="right")
    table.add_column("Mem (GB)", justify="right")
    table.add_column("°C", justify="right")

    for g in gpus:
        util = int(g["util"]) if g["util"].isdigit() else 0
        util_str = f"[green]{util}%[/]" if util > 50 else f"[dim]{util}%[/]"
        mem_used = float(g["mem_used"]) / 1024
        mem_total = float(g["mem_total"]) / 1024
        table.add_row(g["idx"], g["name"], util_str, f"{mem_used:.0f}/{mem_total:.0f}", g["temp"])

    return Panel(table, title="[bold]GPUs[/bold]", border_style="yellow")


def make_log_panel(runs_dir: Path, stage: str) -> Panel:
    log_paths = []
    if "grpo" in stage:
        log_paths = list((runs_dir / "grpo" / "logs").glob("*.log")) if (runs_dir / "grpo" / "logs").exists() else []
    elif "sft" in stage:
        log_paths = list((runs_dir / "sft" / "logs").glob("*.log")) if (runs_dir / "sft" / "logs").exists() else []

    # Also check data pipeline logs
    log_paths += list((runs_dir / "logs").glob("*.log")) if (runs_dir / "logs").exists() else []

    if not log_paths:
        return Panel(Text("No logs yet.", style="dim"), title="[bold]Recent Logs[/bold]", border_style="dim")

    latest = max(log_paths, key=lambda p: p.stat().st_mtime)
    lines = tail_lines(latest, 10)
    text = Text("\n".join(lines), overflow="fold")
    return Panel(text, title=f"[bold]Logs — {latest.name}[/bold]", border_style="dim")


# ---- Main loop -------------------------------------------------------------

def build_layout(runs_dir: Path) -> Layout:
    stage = detect_stage(runs_dir)
    stage_text = STAGE_LABELS.get(stage, stage)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    header = Panel(
        Text.from_markup(f"[bold]proposal_rl dashboard[/bold]   Stage: {stage_text}   [{timestamp}]"),
        border_style="white",
    )

    layout = Layout()
    layout.split_column(
        Layout(header, name="header", size=3),
        Layout(name="body"),
        Layout(make_log_panel(runs_dir, stage), name="footer", size=14),
    )
    layout["body"].split_row(
        Layout(name="left"),
        Layout(name="right"),
    )
    layout["left"].split_column(
        Layout(make_data_panel(runs_dir), name="data"),
        Layout(make_eval_panel(runs_dir), name="eval"),
    )
    layout["right"].split_column(
        Layout(make_training_panel(runs_dir, stage), name="training"),
        Layout(make_gpu_panel(), name="gpus"),
    )
    return layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs", help="Path to runs/ directory")
    parser.add_argument("--refresh", type=float, default=3.0, help="Refresh interval in seconds")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    console = Console()

    with Live(build_layout(runs_dir), refresh_per_second=1 / args.refresh, screen=True, console=console) as live:
        while True:
            time.sleep(args.refresh)
            live.update(build_layout(runs_dir))


if __name__ == "__main__":
    main()
