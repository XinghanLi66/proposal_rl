#!/usr/bin/env python3
"""
Aggregate benchmark results across checkpoints and print a comparison table
with a bar-chart visualization of pass@k.

Usage:
  python benchmark/report.py
  python benchmark/report.py --task dl_lr_schedule --sort pass_at_5
  python benchmark/report.py --no-chart
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import yaml
from rich.console import Console
from rich.table import Table
from rich import box

_console = Console()


_TS_RE = re.compile(r"_(\d{8}_\d{6})")


def _exp_base(label: str) -> str:
    """Strip trailing _YYYYMMDD_HHMMSS timestamp from a label to get the experiment base name."""
    return _TS_RE.sub("", label)


def _ts_key(label: str) -> str:
    """Return the timestamp string for sorting, or '' if absent."""
    m = _TS_RE.search(label)
    return m.group(1) if m else ""


def load_summaries(runs_dir: Path, task: str | None, all_runs: bool = False) -> list[dict]:
    summaries = []
    for summary_file in sorted((runs_dir / "benchmark").glob("*/summary.json")):
        try:
            s = json.loads(summary_file.read_text())
        except Exception:
            continue
        if task and s.get("task") != task:
            continue
        summaries.append(s)

    if all_runs:
        return summaries

    # Keep only the latest run per (exp_base, strategy) pair.
    best: dict[tuple[str, str], dict] = {}
    for s in summaries:
        label    = s.get("label", s.get("checkpoint", ""))
        strategy = s.get("strategy", "")
        key      = (_exp_base(label), strategy)
        prev     = best.get(key)
        if prev is None:
            best[key] = s
            continue
        ts_new  = _ts_key(label)
        ts_prev = _ts_key(prev.get("label", prev.get("checkpoint", "")))
        if ts_new > ts_prev:
            best[key] = s
        elif ts_new == ts_prev and s.get("n_proposals", 0) > prev.get("n_proposals", 0):
            best[key] = s

    return list(best.values())


def fmt(v: float | None, decimals: int = 3) -> str:
    if v is None:
        return "—"
    return f"{v:.{decimals}f}"


def fmt_pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v*100:.1f}%"


def _bar(value: float, width: int = 20, color: str = "green") -> str:
    """Render a fraction [0,1] as a Rich-colored block bar."""
    if math.isnan(value):
        return "░" * width + " NaN"
    filled = max(0, min(width, round(value * width)))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{color}]{bar}[/{color}] {value*100:4.1f}%"


def print_table(summaries: list[dict], sort_by: str = "pass_at_5",
                show_chart: bool = True) -> None:
    delta_key = next(
        (k for k in (summaries[0] if summaries else {}) if k.startswith("mean_") and k.endswith("_delta")),
        "mean_delta"
    )
    baseline_key = next(
        (k for k in (summaries[0] if summaries else {}) if k.startswith("baseline_")),
        "baseline_metric"
    )

    summaries = sorted(
        summaries,
        key=lambda s: (lambda v: -1.0 if (v is None or (isinstance(v, float) and math.isnan(v))) else v)(s.get(sort_by)),
        reverse=True,
    )

    # ── Rich table ──────────────────────────────────────────────────────────────
    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold cyan")
    table.add_column("Label", style="bold", no_wrap=True)
    table.add_column("Task", style="dim")
    table.add_column("n", justify="right")
    table.add_column("pass", justify="right", style="green")
    table.add_column("err", justify="right", style="red")
    table.add_column("p@1", justify="right")
    table.add_column("p@3", justify="right")
    table.add_column("p@5", justify="right", style="bold")
    table.add_column("p@10", justify="right")
    table.add_column("Δmetric", justify="right")
    table.add_column("err%", justify="right", style="dim")

    for s in summaries:
        label    = s.get("label", s.get("checkpoint", "?"))
        task     = s.get("task", "?")
        n        = str(s.get("n_proposals", 0))
        passed   = str(s.get("n_passed", 0))
        errors   = str(s.get("n_errors", 0))
        p1       = fmt(s.get("pass_at_1"))
        p3       = fmt(s.get("pass_at_3"))
        p5       = fmt(s.get("pass_at_5"))
        p10      = fmt(s.get("pass_at_10"))
        delta    = s.get(delta_key)
        if delta is None:
            delta_s = "—"
        elif delta >= 0:
            delta_s = f"[green]+{fmt(delta, 4)}[/green]"
        else:
            delta_s = f"[red]{fmt(delta, 4)}[/red]"
        err_rate = fmt_pct(s.get("worker_error_rate"))
        table.add_row(label, task, n, passed, errors, p1, p3, p5, p10, delta_s, err_rate)

    _console.print(table)
    _console.print(f"[dim]Sorted by: {sort_by}  |  {len(summaries)} run(s)[/dim]")

    # ── Bar chart ────────────────────────────────────────────────────────────────
    if show_chart and summaries:
        _console.print()
        _console.rule("[bold]pass@k bar chart[/bold]")
        # Choose colors cycling through a palette
        colors = ["green", "blue", "magenta", "cyan", "yellow"]
        for i, s in enumerate(summaries):
            label = s.get("label", "?")[:35]
            color = colors[i % len(colors)]
            _console.print(f"[bold {color}]{label:<35}[/bold {color}]")
            for k, key in [(1, "pass_at_1"), (3, "pass_at_3"), (5, "pass_at_5"), (10, "pass_at_10")]:
                v = s.get(key)
                if v is not None:
                    _console.print(f"  p@{k:<3} {_bar(v, 30, color)}")
        _console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default=None, help="Filter by task name")
    parser.add_argument("--sort", default="pass_at_5",
                        help="Column to sort by (default: pass_at_5)")
    parser.add_argument("--no-chart", action="store_true", help="Skip bar chart visualization")
    parser.add_argument("--all-runs", action="store_true",
                        help="Show all runs including older attempts (default: latest per exp only)")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    runs_dir = Path(cfg["runs_dir"])

    summaries = load_summaries(runs_dir, args.task, all_runs=args.all_runs)
    if not summaries:
        _console.print("[yellow]No benchmark results found in runs/benchmark/*/summary.json[/yellow]")
        sys.exit(0)

    print_table(summaries, args.sort, show_chart=not args.no_chart)


if __name__ == "__main__":
    main()
