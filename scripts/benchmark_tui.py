#!/usr/bin/env python3
"""
Benchmark Pipeline TUI — observe the full proposal → worker → eval pipeline.

Each of N samples gets its own row in the sample list. Selecting a sample
shows six panels: Prompt, Proposal, Worker Input, Worker Log, Eval Log, Result.

Usage:
  python scripts/benchmark_tui.py                        # interactive config
  python scripts/benchmark_tui.py --run-dir runs/benchmark/my_run   # load existing
  python scripts/benchmark_tui.py --config configs/base.yaml

Keyboard:
  ↑/↓        Navigate sample list
  1-6        Switch content sub-panel
  r          Re-run selected sample (if done/error)
  s          Save summary to JSON
  q/Ctrl+C   Quit
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import ClassVar

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import (
    Button, Footer, Header, Input, Label, ListItem, ListView,
    Select, Static, TabbedContent, TabPane,
)

from benchmark.pipeline import (
    BenchmarkConfig, BenchmarkPipeline, PipelineStep, SampleState, compute_summary,
)
from benchmark.prompt_cache import ALL_STRATEGIES
from benchmark.tasks.mls_bench import DlLrScheduleTask

# ── Constants ──────────────────────────────────────────────────────────────────

ALL_TASKS = [
    ("dl_lr_schedule", "dl-lr-schedule (ResNet/CIFAR, ~10 min)"),
]

DEFAULT_STRATEGY = "full_refs"

STEP_ICONS = {
    PipelineStep.PENDING:         ("○", "dim"),
    PipelineStep.BUILDING_PROMPT: ("⏳", "yellow"),
    PipelineStep.GENERATING:      ("⏳", "yellow"),
    PipelineStep.RUNNING_WORKER:  ("⏳", "cyan"),
    PipelineStep.RUNNING_EVAL:    ("⏳", "cyan"),
    PipelineStep.DONE:            ("✅", "green"),
    PipelineStep.ERROR:           ("✗", "red"),
}

PIPELINE_STEPS: list[tuple[PipelineStep, str]] = [
    (PipelineStep.BUILDING_PROMPT, "Build prompt"),
    (PipelineStep.GENERATING,      "Generate proposal"),
    (PipelineStep.RUNNING_WORKER,  "Worker (implement)"),
    (PipelineStep.RUNNING_EVAL,    "MLS eval"),
    (PipelineStep.DONE,            "Done"),
]

SUB_PANELS = [
    ("prompt",       "Prompt"),
    ("proposal",     "Proposal"),
    ("worker_input", "Worker Input"),
    ("worker_log",   "Worker Log"),
    ("eval_log",     "Eval Log"),
    ("result",       "Result"),
]

_TASK_REGISTRY = {"dl_lr_schedule": DlLrScheduleTask}


def _scan_experiments(runs_dir: Path) -> list[dict]:
    import yaml as _yaml
    exps: list[dict] = [
        {"label": "claude-opus-4-6",   "checkpoint": None, "is_api": True,
         "model": "claude-opus-4-6",   "strategy": DEFAULT_STRATEGY},
        {"label": "claude-sonnet-4-6", "checkpoint": None, "is_api": True,
         "model": "claude-sonnet-4-6", "strategy": DEFAULT_STRATEGY},
    ]
    exps_root = runs_dir / "exps"
    registered_paths: set[str] = set()

    # 1. Explicit registry — fixed order, both old and new versions visible
    for label, rel_path, strategy in _CHECKPOINT_REGISTRY:
        ckpt = exps_root / rel_path
        if ckpt.exists():
            exps.append({"label": label, "checkpoint": str(ckpt),
                         "is_api": False, "model": None, "strategy": strategy})
            registered_paths.add(str(ckpt.resolve()))

    # 2. Auto-scan fallback for any checkpoints not in the explicit registry
    try:
        finals = sorted(
            exps_root.glob("*/rl/final"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
    except Exception:
        finals = []
    seen: set[str] = set()
    for ckpt in finals:
        if not (ckpt / "config.json").exists():
            continue
        if str(ckpt.resolve()) in registered_paths:
            continue
        exp_name = ckpt.parent.parent.name
        parts = exp_name.rsplit("_", 2)
        base = "_".join(parts[:-2]) if len(parts) > 2 else exp_name
        if base not in seen:
            seen.add(base)
            exp_dir = ckpt.parent.parent
            strategy = DEFAULT_STRATEGY
            cfg_path = exp_dir / "config_base.yaml"
            if cfg_path.exists():
                try:
                    with open(cfg_path) as f:
                        cfg = _yaml.safe_load(f)
                    strategy = cfg.get("prompt_builder", {}).get("strategy", DEFAULT_STRATEGY)
                except Exception:
                    pass
            exps.append({"label": base, "checkpoint": str(ckpt),
                         "is_api": False, "model": None, "strategy": strategy})
    return exps


def _sample_label(s: SampleState) -> str:
    icon, _ = STEP_ICONS[s.step]
    idx = f"{s.index + 1:02d}"
    if s.step == PipelineStep.DONE and s.improvement is not None:
        sign = "+" if s.improvement >= 0 else ""
        return f"{idx} {icon} {sign}{s.improvement:.2f}%"
    if s.step == PipelineStep.ERROR:
        return f"{idx} {icon} error"
    if s.step == PipelineStep.PENDING:
        return f"{idx} {icon} pending"
    return f"{idx} {icon} {s.step.label()}"


def _step_markup(steps: list[tuple[PipelineStep, str]], current: PipelineStep) -> str:
    step_order = [p for p, _ in PIPELINE_STEPS]
    try:
        current_idx = step_order.index(current)
    except ValueError:
        current_idx = -1 if current == PipelineStep.PENDING else len(step_order)

    lines = []
    for j, (step, label) in enumerate(PIPELINE_STEPS):
        if current == PipelineStep.ERROR and j <= current_idx:
            icon, color = "✗", "red"
        elif j < current_idx or current == PipelineStep.DONE:
            icon, color = "✅", "green"
        elif j == current_idx:
            icon, color = "⏳", "yellow"
        else:
            icon, color = "○", "dim"
        lines.append(f"[{color}]{icon}[/{color}] {j+1}. {label}")
    return "\n".join(lines)


def _markup_escape(text: str) -> str:
    return text.replace("[", r"\[")


def _read_file_tail(path: Path, max_chars: int = 20000) -> str:
    if not path.exists():
        return "(not yet written)"
    try:
        text = path.read_text(errors="replace")
        if len(text) > max_chars:
            return f"[...truncated, showing last {max_chars} chars...]\n\n" + _markup_escape(text[-max_chars:])
        return _markup_escape(text) or "(empty)"
    except Exception as exc:
        return f"(read error: {exc})"


def _format_result(sample_dir: Path, state: SampleState, baseline: float,
                   pass_threshold: float) -> str:
    if state.step == PipelineStep.PENDING:
        return "Not yet run."
    if state.step in (PipelineStep.BUILDING_PROMPT, PipelineStep.GENERATING,
                      PipelineStep.RUNNING_WORKER, PipelineStep.RUNNING_EVAL):
        return f"Running: {state.step.label()}…"
    if state.step == PipelineStep.ERROR:
        return f"[red]ERROR[/red]\n\n{state.error or '(unknown)'}"

    lines = []
    result_path = sample_dir / "result.json"
    if result_path.exists():
        try:
            r = json.loads(result_path.read_text())
            val = r.get("val_metric")
            if val is not None:
                delta = val - baseline
                sign = "+" if delta >= 0 else ""
                status = "[green]PASS ✅[/green]" if state.passed else "[red]FAIL ✗[/red]"
                lines += [
                    f"[bold]Status:[/bold] {status}",
                    f"[bold]test_acc:[/bold] {val:.2f}%",
                    f"[bold]Baseline:[/bold] {baseline:.2f}%  (pass if Δ ≥ +{pass_threshold:.1f}%)",
                    f"[bold]Improvement:[/bold] {sign}{delta:.2f}%",
                    f"[bold]Elapsed:[/bold] {state.elapsed_s:.0f}s",
                ]
                for k, v in r.items():
                    if k.startswith("test_acc_"):
                        lines.append(f"  {k}: {v:.2f}%")
            else:
                lines += [
                    "[red]val_metric is null[/red]",
                    f"Error: {r.get('error', '(none)')}",
                ]
        except Exception as exc:
            lines.append(f"(parse error: {exc})")
    else:
        lines.append("result.json not found")

    return "\n".join(lines)


# ── Messages ──────────────────────────────────────────────────────────────────

class SampleUpdate(Message):
    def __init__(self, index: int, state: SampleState) -> None:
        super().__init__()
        self.index = index
        self.state = state


# ── CSS ───────────────────────────────────────────────────────────────────────

from benchmark.checkpoints import CHECKPOINT_REGISTRY as _CHECKPOINT_REGISTRY

ALL_MACHINES = [
    ("Local (this machine)", ""),
    ("M0  (lxh_agent_0)",    "lxh_agent_0"),
    ("M1  (lxh_agent_1)",    "lxh_agent_1"),
    ("M2  (lxh_agent_2)",    "lxh_agent_2"),
    ("M3  (lxh_agent_3)",    "lxh_agent_3"),
]

APP_CSS = """
Screen { layout: vertical; }

/* Config bar — two rows */
#config-bar {
    height: 9;
    border: solid $primary-darken-2;
    padding: 0 1;
    layout: vertical;
}
#config-row1, #config-row2 {
    layout: horizontal;
    align: left middle;
    height: 4;
}
#config-bar Label { width: auto; margin-right: 1; color: $text-muted; }
#task-select      { width: 34; margin-right: 2; }
#ckpt-select      { width: 26; margin-right: 2; }
#strategy-select  { width: 22; margin-right: 2; }
#n-input          { width: 5;  margin-right: 2; }
#machine-select   { width: 22; margin-right: 2; }
#run-btn          { min-width: 9; margin-right: 1; }
#stop-btn         { min-width: 9; }

/* Progress bar */
#progress-bar {
    height: 1;
    padding: 0 1;
    background: $surface;
    color: $text-muted;
}

/* Main area */
#main-area { height: 1fr; layout: horizontal; }

/* Sample list */
#sample-panel {
    width: 22;
    border: solid $primary;
}
#sample-panel-title {
    background: $primary; color: $text;
    text-align: center; height: 1; padding: 0 1;
}
#sample-list { height: 1fr; }

/* Right side */
#right-panel { width: 1fr; layout: vertical; }

/* Pipeline steps */
#steps-panel {
    height: 9;
    border: solid $primary-darken-2;
    padding: 0 1;
}
#steps-content { height: 1fr; }

/* Content panel */
#content-panel { height: 1fr; }

/* Sub-panel buttons */
#sub-panel-bar {
    height: 3;
    layout: horizontal;
    align: left middle;
    padding: 0 1;
    background: $surface-darken-1;
}
.sub-btn       { min-width: 14; margin-right: 1; }
.sub-btn.active { background: $primary; }

/* Content display */
#content-display { height: 1fr; }
#content-text { padding: 1 2; }

/* Select/Input theming (same as demo_tui) */
#task-select > SelectCurrent,
#ckpt-select > SelectCurrent,
#strategy-select > SelectCurrent,
#machine-select > SelectCurrent {
    background: #1c1c2e !important; color: #cdd6f4 !important;
    border: tall #1e9afe !important;
}
#task-select > SelectCurrent Static#label,
#ckpt-select > SelectCurrent Static#label,
#strategy-select > SelectCurrent Static#label,
#machine-select > SelectCurrent Static#label,
#task-select > SelectCurrent.-has-value Static#label,
#ckpt-select > SelectCurrent.-has-value Static#label,
#strategy-select > SelectCurrent.-has-value Static#label,
#machine-select > SelectCurrent.-has-value Static#label {
    background: transparent !important; color: #cdd6f4 !important;
}
#task-select > SelectCurrent .arrow,
#ckpt-select > SelectCurrent .arrow,
#strategy-select > SelectCurrent .arrow,
#machine-select > SelectCurrent .arrow {
    background: transparent !important; color: #cdd6f4 !important;
}
#n-input {
    background: #1c1c2e !important; color: #cdd6f4 !important;
    border: tall #1e9afe !important;
}
"""


# ── App ───────────────────────────────────────────────────────────────────────

class BenchmarkTUI(App):
    """Full-pipeline benchmark TUI."""

    CSS = APP_CSS
    TITLE = "Benchmark Pipeline"
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("q", "quit", "Quit"),
        Binding("1", "sub_panel('prompt')",       "Prompt",       show=False),
        Binding("2", "sub_panel('proposal')",     "Proposal",     show=False),
        Binding("3", "sub_panel('worker_input')", "Worker Input", show=False),
        Binding("4", "sub_panel('worker_log')",   "Worker Log",   show=False),
        Binding("5", "sub_panel('eval_log')",     "Eval Log",     show=False),
        Binding("6", "sub_panel('result')",       "Result",       show=False),
        Binding("s", "save_summary", "Save summary"),
    ]

    _selected_sample: reactive[int] = reactive(0)
    _active_sub_panel: reactive[str] = reactive("prompt")

    def __init__(self, run_dir: Path | None = None, runs_dir: Path | None = None):
        super().__init__()
        self._runs_dir = runs_dir or (_REPO / "runs")
        self._run_dir: Path | None = run_dir
        self._pipeline: BenchmarkPipeline | None = None
        self._exps = _scan_experiments(self._runs_dir)
        self._task: DlLrScheduleTask | None = None
        self._refresh_timer: Timer | None = None

    # ── Composition ────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header()
        yield from self._compose_config_bar()
        yield Static("", id="progress-bar")
        with Horizontal(id="main-area"):
            with Vertical(id="sample-panel"):
                yield Static("Samples", id="sample-panel-title")
                yield ListView(id="sample-list")
            with Vertical(id="right-panel"):
                with Vertical(id="steps-panel"):
                    yield Static("", id="steps-content")
                with Vertical(id="content-panel"):
                    with Horizontal(id="sub-panel-bar"):
                        for key, label in SUB_PANELS:
                            cls = "sub-btn active" if key == "prompt" else "sub-btn"
                            yield Button(label, id=f"sub-btn-{key}", classes=cls)
                    with VerticalScroll(id="content-display"):
                        yield Static("", id="content-text", markup=True)
        yield Footer()

    def _compose_config_bar(self):
        with Vertical(id="config-bar"):
            with Horizontal(id="config-row1"):
                yield Label("Task:")
                yield Select(
                    [(label, name) for name, label in ALL_TASKS],
                    id="task-select", value="dl_lr_schedule",
                )
                yield Label("Checkpoint:")
                yield Select(
                    [(e["label"], e["label"]) for e in self._exps],
                    id="ckpt-select",
                    value=self._exps[0]["label"] if self._exps else Select.BLANK,
                )
                yield Label("Strategy:")
                yield Select(
                    [(s, s) for s in ALL_STRATEGIES],
                    id="strategy-select", value=DEFAULT_STRATEGY,
                )
                yield Label("N:")
                yield Input("20", id="n-input")
            with Horizontal(id="config-row2"):
                yield Label("Machine:")
                yield Select(
                    [(label, value) for label, value in ALL_MACHINES],
                    id="machine-select", value="",
                )
                yield Button("▶ Run", id="run-btn", variant="success")
                yield Button("■ Stop", id="stop-btn", variant="error")

    # ── Startup ────────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        self._refresh_timer = self.set_interval(1.0, self._poll_state)
        if self._run_dir is not None:
            self._load_run(self._run_dir)
        else:
            self._init_empty_list(20)

    def _init_empty_list(self, n: int) -> None:
        lv = self.query_one("#sample-list", ListView)
        lv.clear()
        for i in range(n):
            lv.append(ListItem(Label(f"{i+1:02d} ○  pending"), id=f"sample-{i}"))

    def _load_run(self, run_dir: Path) -> None:
        try:
            task_cls = DlLrScheduleTask  # extend with registry lookup later
            self._task = task_cls()
            self._pipeline = BenchmarkPipeline.load(
                run_dir, self._task, self._on_sample_update
            )
            self._rebuild_sample_list()
            self._refresh_content()
        except Exception as exc:
            self.notify(f"Failed to load run: {exc}", severity="error")

    def _rebuild_sample_list(self) -> None:
        if self._pipeline is None:
            return
        lv = self.query_one("#sample-list", ListView)
        lv.clear()
        for s in self._pipeline.samples:
            lv.append(ListItem(Label(_sample_label(s)), id=f"sample-{s.index}"))

    # ── Run button ─────────────────────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run-btn":
            self._start_run()
        elif event.button.id == "stop-btn":
            if self._pipeline:
                self._pipeline.stop()
                self.notify("Stopping pipeline…")
        else:
            # Sub-panel buttons
            for key, _ in SUB_PANELS:
                if event.button.id == f"sub-btn-{key}":
                    self.action_sub_panel(key)
                    break

    def _start_run(self) -> None:
        try:
            task_name = self.query_one("#task-select", Select).value
            ckpt_label = self.query_one("#ckpt-select", Select).value
            strategy = self.query_one("#strategy-select", Select).value
            n = int(self.query_one("#n-input", Input).value or "20")
            machine = str(self.query_one("#machine-select", Select).value or "")
            if machine == "None":
                machine = ""
        except Exception as exc:
            self.notify(f"Config error: {exc}", severity="error")
            return

        exp = next((e for e in self._exps if e["label"] == ckpt_label), None)
        if exp is None:
            self.notify("Unknown checkpoint", severity="error")
            return

        task_cls = _TASK_REGISTRY.get(str(task_name))
        if task_cls is None:
            self.notify(f"Unknown task: {task_name}", severity="error")
            return

        import uuid as _uuid
        run_id = _uuid.uuid4().hex[:8]
        run_dir = self._runs_dir / "benchmark" / f"{task_name}_{run_id}"

        cfg = BenchmarkConfig(
            task_name=str(task_name),
            checkpoint=exp.get("checkpoint"),
            strategy=str(strategy),
            n_samples=n,
            run_id=run_id,
            max_workers=2,
            is_api=exp.get("is_api", False),
            api_model=exp.get("model", "claude-opus-4-6"),
            run_dir=str(run_dir),
            machine=machine,
        )

        self._task = task_cls()
        self._pipeline = BenchmarkPipeline(cfg, self._task, run_dir, self._on_sample_update)
        self._run_dir = run_dir
        self._init_empty_list(n)
        self._pipeline.start()
        self.notify(f"Run started → {run_dir.name}")

    # ── State updates ──────────────────────────────────────────────────────────

    def _on_sample_update(self, index: int, state: SampleState) -> None:
        self.post_message(SampleUpdate(index, state))

    def on_sample_update(self, msg: SampleUpdate) -> None:
        if self._pipeline is None:
            return
        self._update_sample_label(msg.index, msg.state)
        if msg.index == self._selected_sample:
            self._refresh_content()
        self._refresh_progress()

    def _update_sample_label(self, i: int, s: SampleState) -> None:
        try:
            item = self.query_one(f"#sample-{i}", ListItem)
            item.query_one(Label).update(_sample_label(s))
        except NoMatches:
            pass

    def _poll_state(self) -> None:
        """Periodic refresh for streaming logs (worker.log, eval.log)."""
        if self._pipeline is None:
            return
        s = self._pipeline.samples[self._selected_sample]
        if s.step in (PipelineStep.RUNNING_WORKER, PipelineStep.RUNNING_EVAL):
            self._refresh_content()
        self._refresh_progress()

    def _refresh_progress(self) -> None:
        if self._pipeline is None:
            return
        samples = self._pipeline.samples
        total = len(samples)
        done = sum(1 for s in samples if s.step == PipelineStep.DONE)
        errors = sum(1 for s in samples if s.step == PipelineStep.ERROR)
        passed = sum(1 for s in samples if s.passed)
        running = sum(1 for s in samples if s.step in (
            PipelineStep.BUILDING_PROMPT, PipelineStep.GENERATING,
            PipelineStep.RUNNING_WORKER, PipelineStep.RUNNING_EVAL))

        if done > 0:
            imps = [s.improvement for s in samples if s.improvement is not None]
            mean_d = sum(imps) / len(imps)
            sign = "+" if mean_d >= 0 else ""
            delta_str = f"  Mean Δ: {sign}{mean_d:.2f}%"
        else:
            delta_str = ""

        pass_str = f"  Pass: {passed}/{done}" if done > 0 else ""
        msg = (f"Progress: {done+errors}/{total} done | {running} running"
               f"{pass_str}{delta_str}  |  Errors: {errors}")
        try:
            self.query_one("#progress-bar", Static).update(msg)
        except NoMatches:
            pass

    # ── Navigation ─────────────────────────────────────────────────────────────

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.list_view.id != "sample-list":
            return
        if event.item is None:
            return
        try:
            idx = int(event.item.id.split("-")[1])
            self._selected_sample = idx
            self._refresh_content()
        except (ValueError, IndexError, AttributeError):
            pass

    def action_sub_panel(self, key: str) -> None:
        self._active_sub_panel = key
        # Update button styling
        for k, _ in SUB_PANELS:
            try:
                btn = self.query_one(f"#sub-btn-{k}", Button)
                if k == key:
                    btn.add_class("active")
                else:
                    btn.remove_class("active")
            except NoMatches:
                pass
        self._refresh_content()

    # ── Content refresh ────────────────────────────────────────────────────────

    def _refresh_content(self) -> None:
        if self._pipeline is None:
            return
        i = self._selected_sample
        samples = self._pipeline.samples
        if i >= len(samples):
            return
        state = samples[i]
        sample_dir = self._pipeline.get_sample_dir(i)

        # Pipeline steps panel
        markup = _step_markup(PIPELINE_STEPS, state.step)
        if state.step == PipelineStep.ERROR and state.error:
            markup += f"\n\n[red]Error:[/red] {state.error[:120]}"
        try:
            self.query_one("#steps-content", Static).update(markup)
        except NoMatches:
            pass

        # Content panel
        panel = self._active_sub_panel
        content = self._get_panel_content(panel, sample_dir, state)
        try:
            self.query_one("#content-text", Static).update(content)
        except NoMatches:
            pass

    def _get_panel_content(self, panel: str, sample_dir: Path,
                           state: SampleState) -> str:
        if panel == "prompt":
            rec_path = sample_dir / "prompt_record.json"
            if not rec_path.exists():
                return "(prompt not yet built)"
            try:
                rec = json.loads(rec_path.read_text())
                system = _markup_escape(rec.get("system", ""))
                prompt = _markup_escape(rec.get("prompt", ""))
                strategy = _markup_escape(rec.get("strategy", ""))
                return (f"[bold blue]System:[/bold blue] {system}\n\n"
                        f"[bold blue]User[/bold blue] [dim]({strategy}):[/dim]\n{prompt}")
            except Exception as exc:
                return f"(parse error: {exc})"

        elif panel == "proposal":
            return _read_file_tail(sample_dir / "proposal.txt")

        elif panel == "worker_input":
            return _read_file_tail(sample_dir / "worker_prompt.txt")

        elif panel == "worker_log":
            return _read_file_tail(sample_dir / "worker.log")

        elif panel == "eval_log":
            return _read_file_tail(sample_dir / "eval.log")

        elif panel == "result":
            if self._task is None:
                return "(no task loaded)"
            return _format_result(
                sample_dir, state,
                baseline=self._task.baseline_metric(),
                pass_threshold=self._task.pass_threshold,
            )

        return "(unknown panel)"

    # ── Actions ────────────────────────────────────────────────────────────────

    def action_save_summary(self) -> None:
        if self._pipeline is None or self._task is None or self._run_dir is None:
            self.notify("No run loaded", severity="warning")
            return
        summary = compute_summary(self._pipeline.samples, self._task)
        out = self._run_dir / "summary.json"
        out.write_text(json.dumps(summary, indent=2))
        self.notify(f"Summary saved → {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", help="Load an existing run directory")
    parser.add_argument("--config", help="Path to training config YAML (unused here)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else None
    runs_dir = _REPO / "runs"
    app = BenchmarkTUI(run_dir=run_dir, runs_dir=runs_dir)
    app.run()


if __name__ == "__main__":
    main()
