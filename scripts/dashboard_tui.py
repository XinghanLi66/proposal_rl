#!/usr/bin/env python3
"""
Benchmark Sweep Dashboard — live matrix view of all registered runs.

Run discovery is automatic: every pipeline appends to
  runs/benchmark/REGISTRY.jsonl
on start, so the dashboard needs no configuration.

Usage:
  python scripts/dashboard_tui.py
  python scripts/dashboard_tui.py --runs-dir /path/to/runs
  python scripts/dashboard_tui.py --backfill   # import pre-registry runs

Keyboard (matrix):
  ↑ ↓ ← →     Navigate matrix cells
  Enter        Open full detail view for selected cell
  b            Backfill registry from existing run dirs
  r            Force refresh now
  q            Quit

Keyboard (detail screen):
  Esc          Back to dashboard
  ↑ / ↓        Previous / next sample
  1–9          Switch content sub-panel
               1:Prompt  2:Proposal  3:Worker Input  4:Worker Log
               5:Eval Log  6:Result
               7:Abstract  (problem+approach extracted from proposal)
               8:Worker Steps  (tool-call trace parsed from worker.log)
               9:Final Code  (editable_region.py written by worker)
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path
from typing import ClassVar

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

from textual.app import App, ComposeResult, Screen
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import (
    Button, DataTable, Footer, Header, Label, ListItem, ListView,
    Markdown, Static, TabbedContent, TabPane,
)

def _markup_escape(text: str) -> str:
    """Escape all [ chars so Textual's Content.from_markup() never sees unintended tags."""
    return text.replace("[", "\\[")

from benchmark.pipeline import PipelineStep, SampleState
from benchmark.prompt_cache import ALL_STRATEGIES, load_cached
from benchmark.registry import RunRecord, backfill_registry, load_registry, model_sort_key

# ── Constants ─────────────────────────────────────────────────────────────────

_RUNS_DIR = _REPO / "runs"

# Strategy columns shown in the matrix (ordered).
STRATEGY_COLS = ALL_STRATEGIES  # e.g. ["top_k_refs", "with_research_question", ...]

# Per-subtask baseline metrics for the result panel.
BASELINES: dict[tuple[str, str], tuple[float, float]] = {
    ("dl_lr_schedule", "resnet20-cifar10"):   (92.71, 0.30),
    ("dl_lr_schedule", "resnet56-cifar100"):  (72.43, 0.30),
    ("dl_lr_schedule", "mobilenetv2-fmnist"): (94.83, 0.30),
}

STEP_ICONS = {
    PipelineStep.PENDING:         ("○", "dim"),
    PipelineStep.BUILDING_PROMPT: ("⏳", "yellow"),
    PipelineStep.GENERATING:      ("⏳", "yellow"),
    PipelineStep.RUNNING_WORKER:  ("⏳", "cyan"),
    PipelineStep.RUNNING_EVAL:    ("⏳", "cyan"),
    PipelineStep.DONE:            ("✅", "green"),
    PipelineStep.ERROR:           ("✗", "red"),
}

PIPELINE_STEPS = [
    (PipelineStep.BUILDING_PROMPT, "Build prompt"),
    (PipelineStep.GENERATING,      "Generate proposal"),
    (PipelineStep.RUNNING_WORKER,  "Worker (implement)"),
    (PipelineStep.RUNNING_EVAL,    "MLS eval"),
    (PipelineStep.DONE,            "Done"),
]

SUB_PANELS = [
    ("prompt",        "1:Prompt"),
    ("proposal",      "2:Proposal"),
    ("worker_input",  "3:Worker Input"),
    ("worker_log",    "4:Worker Log"),
    ("eval_log",      "5:Eval Log"),
    ("result",        "6:Result"),
    ("abstract",      "7:Abstract"),
    ("worker_steps",  "8:Worker Steps"),
    ("final_code",    "9:Final Code"),
]


# ── Strategy inference (for old registry entries without strategy field) ──────

_LABEL_STRATEGY_MAP = {
    "top_k_refs":          "top_k_refs",
    "topk_rw":             "top_k_related_work",
    "related_work":        "related_work",
    "research_q":          "with_research_question",
    "full_refs":           "full_refs",
}


def _infer_strategy(model_label: str) -> str:
    """Guess prompt strategy from model checkpoint name (for backward compat)."""
    for fragment, strategy in _LABEL_STRATEGY_MAP.items():
        if fragment in model_label:
            return strategy
    return model_label  # fallback: use label as strategy key


# ── Shared helpers ────────────────────────────────────────────────────────────

def _load_samples(run_dir: Path) -> list[SampleState]:
    samples = []
    i = 0
    while True:
        sd = run_dir / f"sample_{i:02d}"
        if not sd.exists():
            break
        s = SampleState.load(sd)
        if s is not None:
            samples.append(s)
        i += 1
    return samples


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


def _read_file_tail(path: Path, max_chars: int = 20000) -> str:
    if not path.exists():
        return "(not yet written)"
    try:
        text = path.read_text(errors="replace")
        if len(text) > max_chars:
            raw = text[-max_chars:]
            return f"[dim]\\[...truncated, showing last {max_chars} chars...][/dim]\n\n" + _markup_escape(raw)
        return _markup_escape(text) if text else "(empty)"
    except Exception as exc:
        return f"(read error: {_markup_escape(str(exc))})"


def _step_markup(current: PipelineStep) -> str:
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


def _extract_abstract(proposal_path: Path) -> str:
    """Extract <problem> and <approach> sections from proposal.txt using regex."""
    import re
    if not proposal_path.exists():
        return "(proposal.txt not yet written)"
    text = proposal_path.read_text(errors="replace")
    out = []
    for tag in ("title", "problem", "approach", "novelty", "experiment"):
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        if m:
            content = _markup_escape(m.group(1).strip())
            out.append(f"[bold cyan]<{tag}>[/bold cyan]\n{content}\n")
    if not out:
        text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()
        return _markup_escape(text) or "(empty proposal)"
    return "\n".join(out)


def _extract_worker_steps(worker_log_path: Path) -> str:
    """Parse worker.log JSON stream and list tool calls in order."""
    if not worker_log_path.exists():
        return "(worker.log not yet written)"
    lines = worker_log_path.read_text(errors="replace").splitlines()
    steps = []
    step_num = 0
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") != "assistant":
            continue
        for block in d.get("message", {}).get("content", []):
            if block.get("type") != "tool_use":
                continue
            step_num += 1
            name = block.get("name", "")
            inp = block.get("input", {})
            if name == "Bash":
                cmd = _markup_escape(inp.get("command", "").strip().replace("\n", " ↵ "))
                desc = _markup_escape(inp.get("description", ""))
                line_out = f"[bold yellow]{step_num:>3}. Bash[/bold yellow]  {desc or cmd[:120]}"
                if desc and cmd:
                    line_out += f"\n      [dim]{cmd[:120]}[/dim]"
            elif name == "Edit":
                fp = _markup_escape(inp.get("file_path", "").split("/")[-1])
                old = _markup_escape((inp.get("old_string") or "").strip()[:40].replace("\n", "↵"))
                new = _markup_escape((inp.get("new_string") or "").strip()[:40].replace("\n", "↵"))
                line_out = f"[bold green]{step_num:>3}. Edit[/bold green]  {fp}\n      [dim]-{old}[/dim]\n      [dim]+{new}[/dim]"
            elif name == "Write":
                fp = _markup_escape(inp.get("file_path", "").split("/")[-1])
                line_out = f"[bold green]{step_num:>3}. Write[/bold green] {fp}"
            elif name == "Read":
                fp = _markup_escape(inp.get("file_path", "").split("/")[-1])
                line_out = f"[bold blue]{step_num:>3}. Read[/bold blue]  {fp}"
            else:
                line_out = f"[dim]{step_num:>3}. {_markup_escape(name)}[/dim]"
            steps.append(line_out)
    if not steps:
        return "(no tool calls found yet)"
    return "\n".join(steps)


def _extract_final_code(sample_dir: Path) -> str:
    """Read editable_region.py — check sample dir first (post-completion copy), then workspace."""
    for path in (sample_dir / "editable_region.py",
                 sample_dir / "workspace" / "editable_region.py"):
        if path.exists():
            return _markup_escape(path.read_text(errors="replace"))
    return "(editable_region.py not yet written)"


def _format_result(sample_dir: Path, state: SampleState,
                   baseline: float, pass_threshold: float) -> str:
    if state.step == PipelineStep.PENDING:
        return "Not yet run."
    if state.step in (PipelineStep.BUILDING_PROMPT, PipelineStep.GENERATING,
                      PipelineStep.RUNNING_WORKER, PipelineStep.RUNNING_EVAL):
        return f"Running: {state.step.label()}…"
    if state.step == PipelineStep.ERROR:
        return f"[red]ERROR[/red]\n\n{state.error or '(unknown)'}"
    result_path = sample_dir / "result.json"
    if not result_path.exists():
        return "result.json not found"
    try:
        r = json.loads(result_path.read_text())
        val = r.get("val_metric")
        if val is None:
            return f"[red]val_metric is null[/red]\nError: {r.get('error', '(none)')}"
        delta = val - baseline
        sign = "+" if delta >= 0 else ""
        status = "[green]PASS ✅[/green]" if state.passed else "[red]FAIL ✗[/red]"
        lines = [
            f"[bold]Status:[/bold]      {status}",
            f"[bold]test_acc:[/bold]    {val:.2f}%",
            f"[bold]Baseline:[/bold]    {baseline:.2f}%  (pass if Δ ≥ +{pass_threshold:.1f}%)",
            f"[bold]Improvement:[/bold] {sign}{delta:.2f}%",
            f"[bold]Elapsed:[/bold]     {state.elapsed_s:.0f}s",
        ]
        for k, v in r.items():
            if k.startswith("test_acc_"):
                lines.append(f"  {k}: {v:.2f}%")
        return "\n".join(lines)
    except Exception as exc:
        return f"(parse error: {exc})"


def _cell_summary(samples: list[SampleState]) -> str:
    if not samples:
        return "—"
    total = len(samples)
    done = [s for s in samples if s.step == PipelineStep.DONE]
    errors = [s for s in samples if s.step == PipelineStep.ERROR]
    running = [s for s in samples
               if s.step not in (PipelineStep.DONE, PipelineStep.ERROR, PipelineStep.PENDING)]
    passed = [s for s in done if s.passed]
    n_fin = len(done) + len(errors)

    if n_fin == 0:
        if running:
            return f"⏳ {len(running)}/{total}"
        return "○ pending"

    imps = [s.improvement for s in done if s.improvement is not None]
    mean = sum(imps) / len(imps) if imps else 0.0
    sign = "+" if mean >= 0 else ""
    verdict = f"✅{len(passed)}" if passed else "✗"
    return f"{n_fin}/{total} {sign}{mean:.2f}% {verdict}"


# ── Messages ─────────────────────────────────────────────────────────────────

class PromptBuilt(Message):
    """Posted when a strategy prompt has been built (or failed)."""
    def __init__(self, strategy: str, entry: dict | None) -> None:
        super().__init__()
        self.strategy = strategy
        self.entry = entry


# ── Detail Screen ─────────────────────────────────────────────────────────────

class DetailScreen(Screen):
    """Full-screen detail view for one (model, subtask) run."""

    CSS = """
    DetailScreen { layout: vertical; }
    #ctx-bar { height: 1; padding: 0 1; background: $surface-darken-1; color: $primary; }
    #detail-main { height: 1fr; layout: horizontal; }
    #detail-sample-panel { width: 22; border-right: solid $primary-darken-2; }
    #detail-sample-title {
        height: 1; background: $primary; color: $text;
        text-align: center; padding: 0 1;
    }
    #detail-sample-list { height: 1fr; }
    #detail-right { width: 1fr; layout: vertical; }
    #detail-steps { height: 9; border-bottom: solid $primary-darken-2; padding: 0 1; }
    #detail-sub-bar {
        height: 3; layout: horizontal; align: left middle;
        padding: 0 1; background: $surface-darken-1;
    }
    .sub-btn { min-width: 15; margin-right: 1; }
    .sub-btn.active { background: $primary; }
    #detail-scroll { height: 1fr; }
    #detail-content { padding: 1 2; }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("1", "sub('prompt')",        show=False),
        Binding("2", "sub('proposal')",      show=False),
        Binding("3", "sub('worker_input')",  show=False),
        Binding("4", "sub('worker_log')",    show=False),
        Binding("5", "sub('eval_log')",      show=False),
        Binding("6", "sub('result')",        show=False),
        Binding("7", "sub('abstract')",      show=False),
        Binding("8", "sub('worker_steps')",  show=False),
        Binding("9", "sub('final_code')",    show=False),
    ]

    def __init__(self, run_dir: Path, model_label: str, subtask: str,
                 task_name: str = "dl_lr_schedule") -> None:
        super().__init__()
        self._run_dir = run_dir
        self._model_label = model_label
        self._subtask = subtask
        self._task_name = task_name
        self._samples: list[SampleState] = []
        self._sel = 0
        self._panel = "worker_log"

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="ctx-bar")
        with Horizontal(id="detail-main"):
            with Vertical(id="detail-sample-panel"):
                yield Static("Samples", id="detail-sample-title")
                yield ListView(id="detail-sample-list")
            with Vertical(id="detail-right"):
                with Vertical(id="detail-steps"):
                    yield Static("", id="detail-steps-content")
                with Horizontal(id="detail-sub-bar"):
                    for key, label in SUB_PANELS:
                        cls = "sub-btn active" if key == self._panel else "sub-btn"
                        yield Button(label, id=f"dsub-{key}", classes=cls)
                with VerticalScroll(id="detail-scroll"):
                    yield Static("", id="detail-content", markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self._refresh()
        self._timer = self.set_interval(2.0, self._refresh)
        try:
            self.query_one("#ctx-bar", Static).update(
                f"{self._model_label}  ×  {self._subtask}"
            )
        except NoMatches:
            pass

    def _refresh(self) -> None:
        self._samples = _load_samples(self._run_dir)
        self._rebuild_list()
        self._refresh_detail()

    def _rebuild_list(self) -> None:
        lv = self.query_one("#detail-sample-list", ListView)
        # Update in-place to avoid lv.clear() destroying widgets that Textual may
        # still hold a reference to (e.g. for deferred scroll_visible), which would
        # raise ValueError: ListItem() is not in list.
        existing = list(lv.query(ListItem))
        n_new, n_old = len(self._samples), len(existing)
        for i in range(min(n_new, n_old)):
            try:
                existing[i].query_one(Label).update(_sample_label(self._samples[i]))
            except Exception:
                pass
        for i in range(n_old, n_new):
            lv.append(ListItem(Label(_sample_label(self._samples[i]))))
        for item in existing[n_new:]:
            item.remove()

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.list_view.id != "detail-sample-list":
            return
        if event.item is None:
            return
        try:
            lv = self.query_one("#detail-sample-list", ListView)
            pos = lv.index  # reactive int: index of highlighted item
            if pos is not None:
                self._sel = self._samples[pos].index
                self._refresh_detail()
        except (ValueError, IndexError, AttributeError):
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid.startswith("dsub-"):
            self.action_sub(bid[5:])

    def action_sub(self, key: str) -> None:
        self._panel = key
        for k, _ in SUB_PANELS:
            try:
                btn = self.query_one(f"#dsub-{k}", Button)
                if k == key:
                    btn.add_class("active")
                else:
                    btn.remove_class("active")
            except NoMatches:
                pass
        self._refresh_detail()

    def _refresh_detail(self) -> None:
        if not self._samples or self._sel >= len(self._samples):
            return
        s = self._samples[self._sel]
        sd = self._run_dir / f"sample_{self._sel:02d}"

        # Steps panel
        markup = _step_markup(s.step)
        if s.step == PipelineStep.ERROR and s.error:
            markup += f"\n\n[red]Error:[/red] {s.error[:120]}"
        try:
            self.query_one("#detail-steps-content", Static).update(markup)
        except NoMatches:
            pass

        # Content panel
        content = self._panel_content(sd, s)
        try:
            self.query_one("#detail-content", Static).update(content)
        except NoMatches:
            pass

    def _panel_content(self, sd: Path, s: SampleState) -> str:
        p = self._panel
        if p == "prompt":
            rec = sd / "prompt_record.json"
            if not rec.exists():
                return "(not yet built)"
            try:
                d = json.loads(rec.read_text())
                sys_txt = _markup_escape(d.get('system', ''))
                usr_txt = _markup_escape(d.get('prompt', ''))
                return (f"[bold blue]System:[/bold blue] {sys_txt}\n\n"
                        f"[bold blue]User:[/bold blue]\n{usr_txt}")
            except Exception as exc:
                return f"(parse error: {_markup_escape(str(exc))})"
        elif p == "proposal":
            return _read_file_tail(sd / "proposal.txt")
        elif p == "worker_input":
            return _read_file_tail(sd / "worker_prompt.txt")
        elif p == "worker_log":
            return _read_file_tail(sd / "worker.log")
        elif p == "eval_log":
            # eval.log lives in workspace/ while running, copied to sample dir on completion
            path = sd / "eval.log"
            if not path.exists():
                path = sd / "workspace" / "eval.log"
            return _read_file_tail(path)
        elif p == "result":
            baseline, thresh = BASELINES.get(
                (self._task_name, self._subtask), (0.0, 0.0)
            )
            return _format_result(sd, s, baseline, thresh)
        elif p == "abstract":
            return _extract_abstract(sd / "proposal.txt")
        elif p == "worker_steps":
            return _extract_worker_steps(sd / "worker.log")
        elif p == "final_code":
            return _extract_final_code(sd)
        return "(unknown panel)"


# ── README content ───────────────────────────────────────────────────────────

README_MD = """\
# Benchmark Evaluation

## Pipeline overview

```
Frontline papers → Build prompt → Generate proposal (checkpoint or Claude API)
  → SlotPool assigns (machine, GPU) → Worker (Claude Code implements get_lr())
  → MLS-Bench eval (200 epochs) → result.json
```

Each sample acquires a GPU slot from the shared `SlotPool` when its worker
starts, then releases it on completion.  Workers from different runs compete
for the same pool, so all idle GPUs are used automatically.

Every run auto-registers in `runs/benchmark/REGISTRY.jsonl` — the dashboard
picks it up instantly, no manual syncing needed.

---

## Launching a sweep (CLI)

```bash
cd /newcpfs/lxh/agentic-training/proposal_rl

# Full sweep — all checkpoints × all strategies × 20 samples
python benchmark/sweep.py

# Single checkpoint, two strategies, all machines
python benchmark/sweep.py \\
  --checkpoints "exp13 full_refs" \\
  --strategies top_k_refs,with_research_question \\
  --n-samples 20

# Restrict worker pool to specific machines / GPUs
python benchmark/sweep.py \\
  --checkpoints "exp12 research_q" \\
  --strategies with_research_question \\
  --machines lxh_agent_1,lxh_agent_2,lxh_agent_3 \\
  --gpus 0-7

# Dry run — print table without launching
python benchmark/sweep.py --dry-run
```

**Key flags**

| Flag | Default | Meaning |
|------|---------|---------|
| `--checkpoints` | `all` | Comma-separated exp prefixes or `all` |
| `--strategies` | `all` | Comma-separated strategy names or `all` |
| `--n-samples` | `20` | Proposals per (checkpoint × strategy) run |
| `--machines` | all 4 DSW machines | SSH aliases for worker pool |
| `--gpus` | `0-7` | GPU range or list, e.g. `0-3` or `0,2,4` |
| `--max-parallel` | `4` | Concurrent generation pipelines (CPU-bound) |

---

## Python API (single run)

```python
from pathlib import Path
from benchmark.pipeline import BenchmarkConfig, BenchmarkPipeline
from benchmark.slot_pool import SlotPool
from benchmark.tasks import get_task
import uuid

subtask    = "resnet20-cifar10"
checkpoint = "runs/exps/exp13_.../rl/final"
run_id     = uuid.uuid4().hex[:8]

# Pool of GPUs to draw from (workers assigned dynamically)
pool = SlotPool.from_machines(["lxh_agent_1"], gpus=range(4))

task = get_task("dl_lr_schedule", subtask=subtask)
cfg  = BenchmarkConfig(
    task_name  = "dl_lr_schedule",
    subtask    = subtask,
    checkpoint = checkpoint,
    strategy   = "top_k_refs",
    n_samples  = 20,
    run_id     = run_id,
    # is_api   = True, api_model = "claude-opus-4-6"  # for API mode
)
run_dir = Path("runs/benchmark") / f"dl_lr_schedule_{subtask}_{run_id}"
BenchmarkPipeline(cfg, task, run_dir, slot_pool=pool).start()
```

---

## Strategies

| Name | Description |
|------|-------------|
| `full_refs` | All reference papers in prompt |
| `top_k_refs` | Top-K most relevant references |
| `related_work` | Related-work section only |
| `with_research_question` | Adds an explicit research question |
| `top_k_related_work` | Top-K refs + related-work framing |

---

## Environment

| Variable | Purpose |
|----------|---------|
| `CUDA_VISIBLE_DEVICES` | Set per-sample by SlotPool slot |
| `BENCHMARK_HMAC_KEY` | Signs `result.json` (default: `benchmark-eval-secret`) |
| `ANTHROPIC_API_KEY` | Required for API-mode proposal generation |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` | `1` — speeds up worker startup |

**Conda env for training:** `/newcpfs/lxh/miniconda3/envs/loongflow_ml`

---

## Timings (resnet20-cifar10, 200 epochs)

| Phase | Time |
|-------|------|
| Proposal generation (per sample, sequential) | ~3 min |
| Worker training (per sample, 1 GPU) | ~20 min |
| **Total, N samples, K free GPUs** | **~3N/1 gen + 20 min** (if K ≥ N) |

With 20 samples and ≥ 20 free GPU slots: all workers run in parallel,
wall time ≈ 60 min generation + 20 min training = **~80 min**.

---

## Pass criteria

| Subtask | Baseline | Pass if Δ ≥ |
|---------|----------|-------------|
| resnet20-cifar10 | 92.71% | +0.30% |
| resnet56-cifar100 | 72.43% | +0.30% |
| mobilenetv2-fmnist | 94.83% | +0.30% |

---

## Dashboard columns

| Column | Meaning |
|--------|---------|
| Progress | `fin/total  ⏳Nw  ↻Ng` — finished, active workers (GPU), generating |
| Pass/Done | `passed/done  N✗` — passed evals / completed / errors |
| Mean Δ | Mean test-acc improvement over baseline across completed samples |

---

## Output layout

```
runs/benchmark/<run_id>/
  config.json           BenchmarkConfig (task, subtask, checkpoint, …)
  sample_00/
    state.json          live status: step, improvement, elapsed_s
    proposal.txt        raw model output
    worker_prompt.txt   full prompt sent to Claude Code
    worker.log          Claude Code CLI output
    editable_region.py  worker's get_lr() implementation
    eval.log            MLS-Bench training output
    result.json         {val_metric, _sig, test_acc_*}
  sample_01/ …
```
"""


# ── CSS ───────────────────────────────────────────────────────────────────────

DASHBOARD_CSS = """
Screen { layout: vertical; }

TabbedContent { height: 1fr; }
TabbedContent > ContentSwitcher { height: 1fr; }
TabPane { padding: 0; layout: vertical; height: 1fr; }

#status-bar {
    height: 1; padding: 0 1;
    background: $surface-darken-1; color: $text-muted;
}

#matrix-area { height: 1fr; }

DataTable { height: 1fr; }
DataTable > .datatable--header { background: $primary-darken-2; color: $text; }
DataTable > .datatable--cursor { background: $accent-darken-1; }

#detail-pane {
    height: 9;
    border-top: solid $primary-darken-2;
    padding: 0 1;
    layout: vertical;
}
#detail-pane-title {
    height: 1; color: $primary; text-style: bold;
}
#detail-pane-samples {
    height: 2; color: $text;
}
#detail-pane-hint {
    height: 1; color: $text-muted;
}
#detail-pane-stats {
    height: 2; color: $text-muted;
}

#readme-scroll { height: 1fr; padding: 1 3; }

/* Prompts tab */
#prompts-split { height: 1fr; layout: horizontal; }
#prompts-list-panel {
    width: 32;
    border-right: solid $primary-darken-2;
    layout: vertical;
}
#prompts-list-title {
    height: 1; background: $primary; color: $text;
    text-align: center; padding: 0 1;
}
#prompts-list { height: 1fr; }
#prompts-actions {
    height: 3; layout: horizontal; align: left middle;
    padding: 0 1; background: $surface-darken-1;
}
#build-all-btn  { min-width: 14; margin-right: 1; }
#prompts-status { color: $text-muted; }
#prompts-right  { width: 1fr; layout: vertical; }
#prompts-meta   { height: 2; padding: 0 1; color: $text-muted; }
#prompts-scroll { height: 1fr; }
#prompts-content { padding: 1 2; }
"""


# ── Dashboard Screen (main) ───────────────────────────────────────────────────

class DashboardApp(App):
    """Live matrix dashboard for all benchmark runs."""

    CSS = DASHBOARD_CSS
    TITLE = "Benchmark Dashboard"
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("q", "quit", "Quit"),
        Binding("r", "force_refresh", "Refresh"),
        Binding("b", "backfill", "Backfill"),
        Binding("enter", "open_detail", "Detail", priority=True),
    ]

    def __init__(self, runs_dir: Path) -> None:
        super().__init__()
        self._runs_dir = runs_dir
        self._records: list[RunRecord] = []
        # Latest run per (model_label, strategy)
        self._cell_run: dict[tuple[str, str], RunRecord] = {}
        # Cached samples per run_dir
        self._sample_cache: dict[str, list[SampleState]] = {}
        self._model_order: list[str] = []
        # Currently highlighted cell
        self._sel_model: str = ""
        self._sel_strategy: str = ""
        self._table_ready = False

    # ── Compose ────────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="tab-dashboard"):
            with TabPane("Dashboard", id="tab-dashboard"):
                yield Static("Loading…", id="status-bar")
                with Vertical(id="matrix-area"):
                    t = DataTable(id="matrix", show_cursor=True, zebra_stripes=True)
                    t.cursor_type = "cell"
                    yield t
                with Vertical(id="detail-pane"):
                    yield Static("Select a cell to see details", id="detail-pane-title")
                    yield Static("", id="detail-pane-samples")
                    yield Static("", id="detail-pane-stats")
                    yield Static("Press Enter to open full detail view", id="detail-pane-hint")
            with TabPane("How to Run", id="tab-readme"):
                with VerticalScroll(id="readme-scroll"):
                    yield Markdown(README_MD)
            with TabPane("Prompts", id="tab-prompts"):
                with Horizontal(id="prompts-split"):
                    with Vertical(id="prompts-list-panel"):
                        yield Static("Strategy", id="prompts-list-title")
                        yield ListView(id="prompts-list")
                        with Horizontal(id="prompts-actions"):
                            yield Button("Build All", id="build-all-btn", variant="primary")
                            yield Button("Force Rebuild", id="force-rebuild-btn", variant="warning")
                            yield Button("Rebuild Selected", id="rebuild-one-btn", variant="default")
                            yield Static("", id="prompts-status")
                    with Vertical(id="prompts-right"):
                        yield Static("", id="prompts-meta")
                        with VerticalScroll(id="prompts-scroll"):
                            yield Static("Select a strategy to view its prompt.", id="prompts-content", markup=True)
        yield Footer()

    # ── Mount ─────────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        self._setup_table()
        self._load_and_rebuild()
        self.set_interval(3.0, self._poll)
        self._setup_prompts_list()
        self._prompts_sel_strategy: str = ALL_STRATEGIES[0]
        self._building_prompts: bool = False

    def _setup_table(self) -> None:
        table = self.query_one("#matrix", DataTable)
        table.add_column("Checkpoint", key="checkpoint", width=36)
        table.add_column("Strategy",   key="strategy",   width=26)
        table.add_column("Subtask",    key="subtask",    width=20)
        table.add_column("Progress",   key="progress",   width=14)
        table.add_column("Pass/Done",  key="passdone",   width=12)
        table.add_column("Mean Δ",     key="meandelta",  width=12)
        self._table_ready = True

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_and_rebuild(self) -> None:
        self._records = load_registry(self._runs_dir)
        self._rebuild_cell_map()
        self._rebuild_table()
        self._update_status()

    def _rebuild_cell_map(self) -> None:
        """Group records by (model_label, strategy), keeping the latest per cell."""
        grouped: dict[tuple[str, str], list[RunRecord]] = {}
        for r in self._records:
            # Derive strategy from record: prefer explicit field, else infer from model_label
            strategy = r.strategy or _infer_strategy(r.model_label)
            key = (r.model_label, strategy)
            grouped.setdefault(key, []).append(r)

        self._cell_run = {
            key: sorted(runs, key=lambda r: r.started_at)[-1]
            for key, runs in grouped.items()
        }
        # All model labels, sorted
        self._model_order = sorted(
            {r.model_label for r in self._records},
            key=model_sort_key,
        )

    def _get_samples(self, run_dir: str) -> list[SampleState]:
        """Cached sample state read (refreshed on every poll)."""
        return _load_samples(Path(run_dir))

    # ── Table build/update ────────────────────────────────────────────────────

    def _row_keys(self) -> list[tuple[str, str]]:
        """All (model_label, strategy) pairs in display order."""
        pairs: list[tuple[str, str]] = []
        for model in self._model_order:
            for strategy in STRATEGY_COLS:
                if (model, strategy) in self._cell_run:
                    pairs.append((model, strategy))
            # Also include strategies not in STRATEGY_COLS (e.g. inferred old runs)
            for (m, s), _ in self._cell_run.items():
                if m == model and s not in STRATEGY_COLS and (m, s) not in pairs:
                    pairs.append((m, s))
        return pairs

    def _row_key_str(self, model: str, strategy: str) -> str:
        return f"{model}||{strategy}"

    def _rebuild_table(self) -> None:
        if not self._table_ready:
            return
        table = self.query_one("#matrix", DataTable)
        table.clear()
        prev_model = ""
        for model, strategy in self._row_keys():
            # Show checkpoint label only on first strategy sub-row; blank for subsequent
            ckpt_label = model if model != prev_model else "  ↳"
            prev_model = model
            rec = self._cell_run.get((model, strategy))
            subtask = rec.subtask if rec else ""
            prog, passdone, meandelta = self._row_cells(model, strategy)
            row_key = self._row_key_str(model, strategy)
            table.add_row(ckpt_label, strategy, subtask, prog, passdone, meandelta,
                          key=row_key)

    def _update_table_cells(self) -> None:
        if not self._table_ready or not self._model_order:
            return
        table = self.query_one("#matrix", DataTable)
        for model, strategy in self._row_keys():
            prog, passdone, meandelta = self._row_cells(model, strategy)
            row_key = self._row_key_str(model, strategy)
            for col_key, val in (("progress", prog), ("passdone", passdone), ("meandelta", meandelta)):
                try:
                    table.update_cell(row_key, col_key, val, update_width=False)
                except Exception:
                    pass

    def _row_cells(self, model: str, strategy: str) -> tuple[str, str, str]:
        """Return (progress, pass/done, mean_delta) strings for a (model, strategy) row."""
        rec = self._cell_run.get((model, strategy))
        if rec is None:
            return "—", "—", "—"
        samples = self._get_samples(rec.run_dir)
        if not samples:
            return "0/0", "—", "—"
        total = rec.n_samples or len(samples)
        done  = [s for s in samples if s.step == PipelineStep.DONE]
        errs  = [s for s in samples if s.step == PipelineStep.ERROR]
        # "active" = in worker or eval (truly consuming a GPU slot right now)
        active = [s for s in samples if s.step in (
            PipelineStep.RUNNING_WORKER, PipelineStep.RUNNING_EVAL)]
        # "generating" = model inference in flight (not yet dispatched to worker)
        generating = [s for s in samples if s.step in (
            PipelineStep.BUILDING_PROMPT, PipelineStep.GENERATING)]
        passed = [s for s in done if s.passed]
        n_fin  = len(done) + len(errs)

        # Progress: "fin/total  ⏳Nw Ng" where Nw=worker-active, Ng=generating
        parts = [f"{n_fin}/{total}"]
        if active:
            parts.append(f"⏳{len(active)}w")
        if generating:
            parts.append(f"↻{len(generating)}g")
        prog = "  ".join(parts)

        # Pass/Done: "passed/done  Nerr err"
        if done or errs:
            pd = f"{len(passed)}/{len(done)}"
            if errs:
                pd += f"  {len(errs)}✗"
            passdone = pd
        else:
            passdone = "—"

        # Mean delta over completed samples
        imps = [s.improvement for s in done if s.improvement is not None]
        if imps:
            mean = sum(imps) / len(imps)
            sign = "+" if mean >= 0 else ""
            meandelta = f"{sign}{mean:.2f}%"
        else:
            meandelta = "—"
        return prog, passdone, meandelta

    # kept for backward compat (used by detail pane)
    def _cell_text(self, model: str, strategy: str) -> str:
        rec = self._cell_run.get((model, strategy))
        if rec is None:
            return "—"
        samples = self._get_samples(rec.run_dir)
        return _cell_summary(samples)

    # ── Polling ───────────────────────────────────────────────────────────────

    def _poll(self) -> None:
        new_records = load_registry(self._runs_dir)
        if len(new_records) != len(self._records):
            # New runs registered — full rebuild
            self._records = new_records
            self._rebuild_cell_map()
            self._rebuild_table()
        else:
            self._update_table_cells()
        self._update_status()
        self._update_detail_pane()

    def action_force_refresh(self) -> None:
        self._load_and_rebuild()
        self.notify("Refreshed")

    def action_backfill(self) -> None:
        n = backfill_registry(self._runs_dir)
        self.notify(f"Backfilled {n} run(s) from disk")
        if n > 0:
            self._load_and_rebuild()

    # ── Prompts tab ───────────────────────────────────────────────────────────

    _TASK_FOR_PROMPTS = "dl_lr_schedule"

    def _prompt_status_icon(self, strategy: str) -> str:
        entry = load_cached(self._TASK_FOR_PROMPTS, strategy, self._runs_dir)
        return "✓" if entry else "○"

    def _setup_prompts_list(self) -> None:
        try:
            lv = self.query_one("#prompts-list", ListView)
            lv.clear()
            for s in ALL_STRATEGIES:
                icon = self._prompt_status_icon(s)
                lv.append(ListItem(Static(f" {icon}  {s}"), id=f"ps-{s}"))
        except NoMatches:
            pass

    def _refresh_prompts_list(self) -> None:
        for s in ALL_STRATEGIES:
            icon = self._prompt_status_icon(s)
            try:
                item = self.query_one(f"#ps-{s}", ListItem)
                item.query_one(Static).update(f" {icon}  {s}")
            except NoMatches:
                pass

    def _show_strategy_prompt(self, strategy: str) -> None:
        entry = load_cached(self._TASK_FOR_PROMPTS, strategy, self._runs_dir)
        try:
            meta_widget = self.query_one("#prompts-meta", Static)
            content_widget = self.query_one("#prompts-content", Static)
        except NoMatches:
            return

        if entry is None:
            meta_widget.update(f"{strategy}  —  not built yet")
            content_widget.update(
                "Prompt not built yet.\n\n"
                "Click [bold]Build All[/bold] to synthesize all 5 strategy prompts.\n"
                "This makes LLM API calls and may take ~2–5 minutes."
            )
            return

        n_refs = entry.get("n_refs", "?")
        n_fl   = entry.get("n_frontlines", 3)
        built  = entry.get("built_at", "")[:10]
        meta_widget.update(
            f"{strategy}  |  {n_refs} refs ({n_fl} frontlines + {n_refs - n_fl} from ref lists)"
            f"  |  built {built}"
        )

        system = entry.get("system", "")
        prompt = entry.get("prompt", "")
        # Show system prompt compactly, then full user prompt
        content_widget.update(
            f"[bold cyan]── SYSTEM ──[/bold cyan]\n{system}\n\n"
            f"[bold cyan]── USER ({strategy}) ──[/bold cyan]\n{prompt}"
        )

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.list_view.id != "prompts-list":
            return
        if event.item is None:
            return
        item_id = getattr(event.item, "id", "") or ""
        if item_id.startswith("ps-"):
            self._prompts_sel_strategy = item_id[3:]
            self._show_strategy_prompt(self._prompts_sel_strategy)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "build-all-btn":
            self._start_build_all(force=False)
        elif event.button.id == "force-rebuild-btn":
            self._start_build_all(force=True)
        elif event.button.id == "rebuild-one-btn":
            self._start_build_one(self._prompts_sel_strategy, force=True)

    def _start_build_all(self, force: bool = False) -> None:
        if self._building_prompts:
            self.notify("Build already running…", severity="warning")
            return
        self._building_prompts = True
        try:
            label = "⏳ rebuilding…" if force else "⏳ building…"
            self.query_one("#prompts-status", Static).update(label)
            self.query_one("#build-all-btn", Button).disabled = True
            self.query_one("#force-rebuild-btn", Button).disabled = True
        except NoMatches:
            pass

        runs_dir = self._runs_dir

        def _build():
            from benchmark.prompt_cache import build_all as _build_all
            _build_all(
                self._TASK_FOR_PROMPTS,
                runs_dir,
                force=force,
                on_progress=lambda s, e: self.post_message(PromptBuilt(s, e)),
            )

        threading.Thread(target=_build, daemon=True).start()

    def _start_build_one(self, strategy: str, force: bool = True) -> None:
        if self._building_prompts:
            self.notify("Build already running…", severity="warning")
            return
        self._building_prompts = True
        try:
            self.query_one("#prompts-status", Static).update(f"⏳ rebuilding {strategy}…")
            for btn_id in ("build-all-btn", "force-rebuild-btn", "rebuild-one-btn"):
                self.query_one(f"#{btn_id}", Button).disabled = True
        except NoMatches:
            pass

        runs_dir = self._runs_dir
        task_name = self._TASK_FOR_PROMPTS

        def _build():
            from benchmark.prompt_cache import build_one as _build_one
            entry = _build_one(task_name, strategy, runs_dir, force=force)
            self.post_message(PromptBuilt(strategy, entry))

        threading.Thread(target=_build, daemon=True).start()

    def on_prompt_built(self, msg: PromptBuilt) -> None:
        self._refresh_prompts_list()
        # If the user is currently viewing this strategy, refresh the content
        if msg.strategy == self._prompts_sel_strategy:
            self._show_strategy_prompt(msg.strategy)
        # Check if all done
        all_done = all(
            load_cached(self._TASK_FOR_PROMPTS, s, self._runs_dir) is not None
            for s in ALL_STRATEGIES
        )
        # Re-enable buttons once this strategy's build completes
        # (for single-strategy builds, done immediately; for build-all, done after last)
        all_done = all(
            load_cached(self._TASK_FOR_PROMPTS, s, self._runs_dir) is not None
            for s in ALL_STRATEGIES
        )
        if msg.entry is not None or all_done:
            self._building_prompts = False
            status = "✓ all built" if all_done else f"✓ {msg.strategy}"
            try:
                self.query_one("#prompts-status", Static).update(status)
                for btn_id in ("build-all-btn", "force-rebuild-btn", "rebuild-one-btn"):
                    self.query_one(f"#{btn_id}", Button).disabled = False
            except NoMatches:
                pass
            if all_done:
                self.notify("All 5 strategy prompts built and cached.")
            else:
                self.notify(f"{msg.strategy} rebuilt.")

    # ── Status bar ────────────────────────────────────────────────────────────

    def _update_status(self) -> None:
        all_samples = [
            s
            for rec in self._cell_run.values()
            for s in self._get_samples(rec.run_dir)
        ]
        n_runs = len(self._cell_run)
        n_done = sum(1 for s in all_samples if s.step == PipelineStep.DONE)
        n_err  = sum(1 for s in all_samples if s.step == PipelineStep.ERROR)
        n_pass = sum(1 for s in all_samples if s.passed)
        n_run  = sum(1 for s in all_samples
                     if s.step not in (PipelineStep.DONE, PipelineStep.ERROR,
                                       PipelineStep.PENDING))
        total  = len(all_samples)
        try:
            self.query_one("#status-bar", Static).update(
                f"{n_runs} cells | {total} samples: "
                f"{n_done} done  {n_pass} passed  {n_err} errors  {n_run} running"
            )
        except NoMatches:
            pass

    # ── Cell navigation ───────────────────────────────────────────────────────

    def on_data_table_cell_highlighted(
        self, event: DataTable.CellHighlighted
    ) -> None:
        try:
            row_key = str(event.cell_key.row_key.value)
        except Exception:
            return
        if "||" not in row_key:
            return
        model, strategy = row_key.split("||", 1)
        self._sel_model = model
        self._sel_strategy = strategy
        self._update_detail_pane()

    def _update_detail_pane(self) -> None:
        if not self._sel_model:
            return
        rec = self._cell_run.get((self._sel_model, self._sel_strategy))

        title = f"{self._sel_model}  ×  {self._sel_strategy}"
        if rec:
            title += f"  [dim](run {rec.run_id} · {rec.subtask} · {rec.machine})[/dim]"
        try:
            self.query_one("#detail-pane-title", Static).update(title)
        except NoMatches:
            pass

        if rec is None:
            for wid in ("#detail-pane-samples", "#detail-pane-stats"):
                try:
                    self.query_one(wid, Static).update("No run registered for this cell.")
                except NoMatches:
                    pass
            return

        samples = self._get_samples(rec.run_dir)

        # Sample chips
        chips = "  ".join(_sample_label(s) for s in samples) if samples else "(no samples yet)"
        try:
            self.query_one("#detail-pane-samples", Static).update(chips)
        except NoMatches:
            pass

        # Stats
        done = [s for s in samples if s.step == PipelineStep.DONE]
        imps = [s.improvement for s in done if s.improvement is not None]
        if imps:
            mean = sum(imps) / len(imps)
            sign = "+" if mean >= 0 else ""
            passed = sum(1 for s in done if s.passed)
            baseline, thresh = BASELINES.get(
                (rec.task, rec.subtask), (0.0, 0.0)
            )
            stats = (f"baseline={baseline:.2f}%  pass_threshold=+{thresh:.2f}%  "
                     f"mean_Δ={sign}{mean:.2f}%  passed={passed}/{len(done)}")
        else:
            stats = ""
        try:
            self.query_one("#detail-pane-stats", Static).update(stats)
        except NoMatches:
            pass

    # ── Open detail screen ────────────────────────────────────────────────────

    def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        """DataTable fires CellSelected on Enter — use it to open detail view."""
        self.action_open_detail()

    def action_open_detail(self) -> None:
        if not self._sel_model or not self._sel_strategy:
            self.notify("Navigate to a row first", severity="warning")
            return
        rec = self._cell_run.get((self._sel_model, self._sel_strategy))
        if rec is None:
            self.notify("No run for this row yet", severity="warning")
            return
        self.push_screen(DetailScreen(
            run_dir=Path(rec.run_dir),
            model_label=f"{self._sel_model} [{self._sel_strategy}]",
            subtask=rec.subtask,
            task_name=rec.task,
        ))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", default=str(_RUNS_DIR),
                        help="Path to the runs/ directory (default: auto-detect)")
    parser.add_argument("--backfill", action="store_true",
                        help="Scan existing run dirs and add missing entries to registry, then exit")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)

    if args.backfill:
        n = backfill_registry(runs_dir)
        print(f"Backfilled {n} run(s) into {runs_dir}/benchmark/REGISTRY.jsonl")
        return

    app = DashboardApp(runs_dir=runs_dir)
    app.run()


if __name__ == "__main__":
    main()
