#!/usr/bin/env python3
"""
Interactive TUI demo for the proposal model.

Usage:
  python scripts/demo_tui.py
  python scripts/demo_tui.py --config configs/base.yaml

Navigation:
  ↑/↓          Move through experiment list (auto-sets strategy to match checkpoint)
  Enter        Generate proposal with highlighted experiment
  a / /        Focus arXiv ID input
  Escape       Return focus to experiment list
  Tab          Cycle focus between panels (for scrolling)
  c            Toggle compare mode
  p            Toggle prompt summary / full
  t            Synthesize / refresh TeX-grounded target preview
  s            Save to JSON
  r            Clear and re-generate
  q / Ctrl+C   Quit

Strategy selector:
  Shown below the arXiv ID bar. Changes automatically when you navigate the
  experiment list (defaults to the strategy used to train that checkpoint).
  You can also click/change it manually — changing strategy re-builds the
  prompt and caches the result under (arxiv_id, strategy).
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

import yaml
import yaml as _yaml_mod
from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, Label, ListItem, ListView, Select, Static

# All known prompt-builder strategies (must match train/prompt_builder.py REGISTRY)
ALL_STRATEGIES = [
    "full_refs",
    "top_k_refs",
    "related_work",
    "with_research_question",
    "top_k_related_work",
]
DEFAULT_STRATEGY = "full_refs"

TEX_TARGET_FIELDS = {
    "target_source",
    "tex_status",
    "paper_dir",
    "tex_dir",
    "tex_file_count",
    "tex_section_count",
    "tex_sections",
    "cot_impl_proposal",
    "target_impl_proposal",
    "target_quality",
    "tex_impl_leakage_score",
    "synthesis_model",
    "synthesis_temperature",
    "synthesized_at",
    "skip_reason",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_exp_strategy(exp_dir: Path) -> str | None:
    """Read the prompt_builder strategy from a saved experiment config_base.yaml."""
    cfg_path = exp_dir / "config_base.yaml"
    if not cfg_path.exists():
        return None
    try:
        with open(cfg_path) as f:
            cfg = _yaml_mod.safe_load(f)
        return cfg.get("prompt_builder", {}).get("strategy")
    except Exception:
        return None


def _scan_experiments(runs_dir: Path) -> list[dict]:
    exps: list[dict] = [
        {"label": "claude-opus-4-6",   "checkpoint": None, "is_api": True,
         "model": "claude-opus-4-6",   "strategy": DEFAULT_STRATEGY},
        {"label": "claude-sonnet-4-6", "checkpoint": None, "is_api": True,
         "model": "claude-sonnet-4-6", "strategy": DEFAULT_STRATEGY},
    ]
    finals = sorted(
        (runs_dir / "exps").glob("*/rl/final"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    seen: set[str] = set()
    for ckpt in finals:
        if not (ckpt / "config.json").exists():
            continue  # skip empty/unmerged finals
        exp_name = ckpt.parent.parent.name
        parts = exp_name.rsplit("_", 2)
        base = "_".join(parts[:-2]) if len(parts) > 2 else exp_name
        if base not in seen:
            seen.add(base)
            exp_dir = ckpt.parent.parent
            strategy = _read_exp_strategy(exp_dir) or DEFAULT_STRATEGY
            exps.append({"label": base, "checkpoint": str(ckpt),
                         "is_api": False, "model": None, "full_name": exp_name,
                         "strategy": strategy})
    return exps


def _short_prompt(prompt: str, n_refs: int = 8) -> str:
    lines = prompt.split("\n")
    out, count = [], 0
    for line in lines:
        out.append(line)
        if line.strip().startswith(f"[{count + 1}]"):
            count += 1
        if count >= n_refs:
            remaining = sum(1 for l in lines if l.strip().startswith("["))
            out.append(f"\n... [{remaining - n_refs} more references — press 'p' for full] ...")
            break
    return "\n".join(out)


def _default_target_cache_files(runs_dir: Path, primary: Path) -> list[Path]:
    dataset = runs_dir / "dataset"
    files = [
        dataset / "train_tex_impl.jsonl",
        dataset / "val_tex_impl.jsonl",
        dataset / "test_tex_impl.jsonl",
        dataset / "train_tex_impl.skipped.jsonl",
        dataset / "val_tex_impl.skipped.jsonl",
        dataset / "test_tex_impl.skipped.jsonl",
        dataset / "demo_tex_impl_targets.jsonl",
        primary,
    ]
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


# ── CSS ───────────────────────────────────────────────────────────────────────

APP_CSS = """
Screen {
    layout: horizontal;
}

/* ── Sidebar ── */
#sidebar {
    width: 28;
    border: solid $primary;
    padding: 0 1;
}
#sidebar-title {
    background: $primary;
    color: $text;
    text-align: center;
    height: 1;
    padding: 0 1;
}
#exp-list {
    height: 1fr;
}

/* ── Main column ── */
#main {
    width: 1fr;
    layout: vertical;
}

/* ── Global form-widget text fix ─────────────────────────────────────────────
   Important points:
   1. Input has no .input--value component; value inherits Input color.
   2. Select focus lives on Select, not SelectCurrent.
   3. Override :ansi explicitly because Textual's widget DEFAULT_CSS may use it.
   ──────────────────────────────────────────────────────────────────────────── */

/* arXiv input: force the value text to inherit a visible color */
#arxiv-input,
#arxiv-input:ansi {
    background: #1c1c2e !important;
    color: #cdd6f4 !important;
    border: tall #1e9afe !important;
}

#arxiv-input:focus,
#arxiv-input:ansi:focus {
    background: #1c1c2e !important;
    color: #ffffff !important;
    border: tall #a6e3a1 !important;
}

/* These are real Input component classes in Textual */
#arxiv-input > .input--placeholder,
#arxiv-input:ansi > .input--placeholder {
    color: #585b70 !important;
}

#arxiv-input > .input--suggestion,
#arxiv-input:ansi > .input--suggestion {
    color: #6c7086 !important;
}

#arxiv-input > .input--cursor,
#arxiv-input:ansi > .input--cursor {
    background: #1e9afe !important;
    color: #ffffff !important;
}

#arxiv-input > .input--selection,
#arxiv-input:ansi > .input--selection {
    background: #45475a !important;
    color: #ffffff !important;
}

/* Strategy Select: the focus is on #strategy-select, not on SelectCurrent */
#strategy-select > SelectCurrent,
#strategy-select > SelectCurrent:ansi {
    background: #1c1c2e !important;
    color: #cdd6f4 !important;
    border: tall #1e9afe !important;
}

#strategy-select:focus > SelectCurrent,
#strategy-select:focus > SelectCurrent:ansi {
    border: tall #a6e3a1 !important;
}

/* Selected strategy label */
#strategy-select > SelectCurrent Static#label,
#strategy-select > SelectCurrent.-has-value Static#label,
#strategy-select > SelectCurrent:ansi Static#label,
#strategy-select > SelectCurrent:ansi.-has-value Static#label {
    background: transparent !important;
    color: #cdd6f4 !important;
}

/* Select arrow */
#strategy-select > SelectCurrent .arrow,
#strategy-select > SelectCurrent:ansi .arrow {
    background: transparent !important;
    color: #cdd6f4 !important;
}

/* ── Paper bar ── */
#paper-bar {
    height: 5;
    border: solid $primary-darken-2;
    padding: 0 1;
    layout: horizontal;
    align: left middle;
}
#paper-bar Label {
    width: auto;
    margin-right: 1;
}
#arxiv-input {
    width: 28;
    margin-right: 1;
    border: tall #1e9afe;
}
#arxiv-input:focus {
    border: tall #a6e3a1;
}
#paper-title {
    color: $text-muted;
    width: 1fr;
    overflow: hidden;
}

/* ── Strategy bar ── */
#strategy-bar {
    height: 5;
    border: solid $primary-darken-3;
    padding: 0 1;
    layout: horizontal;
    align: left middle;
}
#strategy-bar Label {
    width: auto;
    margin-right: 1;
    color: $text-muted;
}
#strategy-select {
    width: 28;
    margin-right: 1;
}
#strategy-hint {
    color: $text-muted;
    width: 1fr;
    overflow: hidden;
}

/* ── Prompt panel ── */
#prompt-scroll {
    height: 22%;
    border: solid $primary-darken-2;
}
#prompt-content {
    padding: 1 2;
    color: $text-muted;
}

/* ── Target preview area ── */
#target-area {
    height: 30%;
    layout: horizontal;
}
.target-wrap {
    width: 1fr;
    border: solid $primary-darken-3;
    layout: vertical;
}
#target-tex-wrap {
    margin-left: 1;
}
.target-header {
    height: 1;
    background: $panel;
    color: $text;
    padding: 0 1;
}
.target-scroll {
    height: 1fr;
}
.target-body {
    padding: 1 2;
    color: $text-muted;
}

/* ── Response area ── */
#response-area {
    height: 1fr;
    layout: horizontal;
}
.resp-wrap {
    width: 1fr;
    border: solid $accent;
    layout: vertical;
}
.resp-wrap.generating { border: solid $warning; }
.resp-wrap.done       { border: solid $success; }
.resp-wrap.error      { border: solid $error; }

.resp-header {
    height: 1;
    background: $panel;
    color: $text;
    padding: 0 1;
}
.resp-scroll {
    height: 1fr;
}
.resp-body {
    padding: 1 2;
}

#response-right-wrap {
    display: none;
    margin-left: 1;
}
#response-area.compare #response-right-wrap {
    display: block;
}

/* ── Status bar (below Footer) ── */
#status {
    dock: bottom;
    height: 1;
    background: $boost;
    color: $text-muted;
    padding: 0 1;
}
"""


# ── App ───────────────────────────────────────────────────────────────────────

class ProposalDemoApp(App):
    CSS = APP_CSS
    TITLE = "Proposal Model Demo"

    # No 'enter' binding here — ListView's own enter binding fires ListView.Selected
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("q",      "quit", "Quit"),
        Binding("a,/",    "focus_arxiv",     "arXiv ID"),
        Binding("escape", "blur_arxiv",      "Back",    show=False),
        Binding("c",      "toggle_compare",  "Compare"),
        Binding("p",      "toggle_prompt",   "Prompt"),
        Binding("t",      "synthesize_target", "Synth"),
        Binding("s",      "save",            "Save"),
        Binding("r",      "regenerate",      "Re-gen"),
    ]

    _compare_mode: bool = False
    _prompt_summary: bool = True
    # Cache: (arxiv_id, strategy) → record dict with pre-built prompt
    _record_cache: dict[tuple[str, str], dict]
    _target_cache: dict[str, dict]
    _generating: bool = False
    _target_generating: bool = False

    def __init__(
        self,
        cfg: dict,
        runs_dir: Path,
        target_cache_path: Path | None = None,
        target_model: str | None = None,
        target_max_tokens: int | None = None,
        target_temperature: float = 0.3,
        max_tex_chars: int = 24000,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._cfg = cfg
        self._runs_dir = runs_dir
        self._experiments = _scan_experiments(runs_dir)
        self._record_cache = {}
        self._target_cache_path = target_cache_path or (runs_dir / "dataset" / "demo_tex_impl_targets.jsonl")
        self._target_model = target_model or cfg.get("cot", {}).get("model", "claude-sonnet-4-6")
        self._target_max_tokens = target_max_tokens or cfg.get("cot", {}).get("max_tokens", 8192)
        self._target_temperature = target_temperature
        self._max_tex_chars = max_tex_chars
        self._target_cache = self._load_target_cache()

    @property
    def _current_strategy(self) -> str:
        try:
            sel = self.query_one("#strategy-select", Select)
            v = sel.value
            return v if v and v != Select.BLANK else DEFAULT_STRATEGY
        except NoMatches:
            return DEFAULT_STRATEGY

    @property
    def _current_record(self) -> dict | None:
        """Return cached record for the current (arxiv_id, strategy), if any."""
        try:
            inp = self.query_one("#arxiv-input", Input)
            arxiv_id = inp.value.strip()
            if not arxiv_id:
                return None
        except NoMatches:
            return None
        return self._record_cache.get((arxiv_id, self._current_strategy))

    def _load_target_cache(self) -> dict[str, dict]:
        from data.synthesize_tex_targets import load_synthesis_cache

        return load_synthesis_cache(
            _default_target_cache_files(self._runs_dir, self._target_cache_path)
        )

    def _attach_cached_target(self, record: dict) -> dict:
        cached = self._target_cache.get(record.get("arxiv_id", ""))
        if not cached:
            return record
        merged = dict(record)
        for key in TEX_TARGET_FIELDS:
            if key in cached:
                merged[key] = cached[key]
        return merged

    # ── Layout ────────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Label("Experiments", id="sidebar-title")
                items = [ListItem(Label(e["label"]), id=f"exp-{i}")
                         for i, e in enumerate(self._experiments)]
                yield ListView(*items, id="exp-list")
            with Vertical(id="main"):
                with Horizontal(id="paper-bar"):
                    yield Label("arXiv ID:")
                    yield Input(placeholder="press 'a', type ID, Enter",
                                id="arxiv-input")
                    yield Label("", id="paper-title")
                with Horizontal(id="strategy-bar"):
                    yield Label("Strategy:")
                    yield Select(
                        [(s, s) for s in ALL_STRATEGIES],
                        value=DEFAULT_STRATEGY,
                        id="strategy-select",
                        allow_blank=False,
                    )
                    yield Label("", id="strategy-hint")
                with VerticalScroll(id="prompt-scroll"):
                    yield Static("[dim]No paper loaded — press 'a', enter an arXiv ID, press Enter.[/dim]",
                                 id="prompt-content")
                with Horizontal(id="target-area"):
                    with Vertical(id="target-abstract-wrap", classes="target-wrap"):
                        yield Label("Original abstract target", id="target-abstract-header",
                                    classes="target-header")
                        with VerticalScroll(classes="target-scroll"):
                            yield Static(
                                "[dim]Load a paper to preview the current abstract target.[/dim]",
                                id="target-abstract-content", classes="target-body")
                    with Vertical(id="target-tex-wrap", classes="target-wrap"):
                        yield Label("TeX synthesis target", id="target-tex-header",
                                    classes="target-header")
                        with VerticalScroll(classes="target-scroll"):
                            yield Static(
                                "[dim]Load a paper, then press 't' to synthesize from TeX.[/dim]",
                                id="target-tex-content", classes="target-body")
                with Horizontal(id="response-area"):
                    with Vertical(id="response-left-wrap", classes="resp-wrap"):
                        yield Label("· ready", id="response-left-header",
                                    classes="resp-header")
                        with VerticalScroll(classes="resp-scroll"):
                            yield Static(
                                "[dim]Load a paper then press Enter on an experiment.[/dim]",
                                id="response-left-content", classes="resp-body")
                    with Vertical(id="response-right-wrap", classes="resp-wrap"):
                        yield Label("· compare", id="response-right-header",
                                    classes="resp-header")
                        with VerticalScroll(classes="resp-scroll"):
                            yield Static("[dim]Enable compare mode with 'c'.[/dim]",
                                         id="response-right-content", classes="resp-body")
        # Footer first → sits above status bar
        yield Footer()
        yield Static("Ready.", id="status")

    def on_mount(self) -> None:
        try:
            lv = self.query_one("#exp-list", ListView)
            if len(self._experiments) > 2:
                lv.index = 2  # skip API baselines; highlight first local checkpoint
        except NoMatches:
            pass
        self._update_status(
            "Ready — press 'a' to enter arXiv ID, Enter to load; "
            "then press Enter on an experiment to generate."
        )

    # ── Guard: suppress app bindings while typing in Input ────────────────────

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        passthrough = {"quit", "blur_arxiv", "focus_next", "focus_previous"}
        if isinstance(self.focused, Input) and action not in passthrough:
            return False
        return True

    # ── Strategy auto-update when experiment is highlighted ───────────────────

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """Update strategy selector to match the highlighted experiment."""
        if event.item is None:
            return
        lv = self.query_one("#exp-list", ListView)
        idx = lv.index
        if idx is None or idx >= len(self._experiments):
            return
        exp = self._experiments[idx]
        strategy = exp.get("strategy", DEFAULT_STRATEGY)
        try:
            sel = self.query_one("#strategy-select", Select)
            sel.value = strategy
            hint = f"[dim]default for {exp['label'][:30]}[/dim]"
            self.query_one("#strategy-hint", Label).update(hint)
        except NoMatches:
            pass

    # ── Paper loading ─────────────────────────────────────────────────────────

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "arxiv-input":
            return
        arxiv_id = event.value.strip()
        if not arxiv_id:
            return
        strategy = self._current_strategy
        cached = self._record_cache.get((arxiv_id, strategy))
        if cached is not None:
            self._refresh_prompt_panel()
            self._refresh_target_panel()
            n_refs = len(cached.get("refs", []))
            self._update_status(
                f"[green]Cached:[/green] {arxiv_id} / {strategy} ({n_refs} refs) — "
                "press Enter on an experiment to generate."
            )
        else:
            self._load_paper(arxiv_id, strategy)
        # Hand focus back to the list so Enter immediately triggers generation
        try:
            self.query_one("#exp-list", ListView).focus()
        except NoMatches:
            pass

    def on_select_changed(self, event: Select.Changed) -> None:
        """Re-load prompt when strategy changes and a paper is already entered."""
        if event.select.id != "strategy-select":
            return
        try:
            arxiv_id = self.query_one("#arxiv-input", Input).value.strip()
        except NoMatches:
            return
        if not arxiv_id:
            return
        strategy = str(event.value)
        cached = self._record_cache.get((arxiv_id, strategy))
        if cached is not None:
            self._refresh_prompt_panel()
            self._refresh_target_panel()
            n_refs = len(cached.get("refs", []))
            self._update_status(
                f"[green]Cached:[/green] {arxiv_id} / {strategy} ({n_refs} refs)"
            )
        else:
            self._load_paper(arxiv_id, strategy)

    def _load_paper(self, arxiv_id: str, strategy: str) -> None:
        self._update_status(f"Searching dataset for {arxiv_id} [{strategy}]...")

        def _do() -> None:
            _scripts = str(Path(__file__).resolve().parent)
            if _scripts not in sys.path:
                sys.path.insert(0, _scripts)
            from probe import load_record_from_arxiv_store, load_record_from_dataset
            from train.prompt_builder import get_builder

            record = load_record_from_dataset(arxiv_id, self._runs_dir)
            if record is not None:
                n_refs = len(record.get("refs", []))
                self.call_from_thread(
                    self._update_status,
                    f"Found in dataset ({n_refs} refs) — building prompt [{strategy}]...",
                )
                # Always rebuild prompt with the requested strategy
                pb_cfg = self._cfg.get("prompt_builder", {}).copy()
                pb_cfg["runs_dir"] = str(self._runs_dir)
                pb_cfg["strategy"] = strategy
                builder = get_builder({**self._cfg, "prompt_builder": pb_cfg})
                record = dict(record)  # don't mutate the original
                record["system"] = builder.system()
                record["prompt"] = builder.build(record)
            else:
                arxiv_root = Path(self._cfg.get("arxiv_root", ""))
                if arxiv_root.exists():
                    self.call_from_thread(
                        self._update_status,
                        f"Not in dataset — fetching from arxiv store [{strategy}]...",
                    )
                    pb_cfg = self._cfg.get("prompt_builder", {}).copy()
                    pb_cfg["runs_dir"] = str(self._runs_dir)
                    pb_cfg["strategy"] = strategy
                    builder = get_builder({**self._cfg, "prompt_builder": pb_cfg})
                    record = load_record_from_arxiv_store(arxiv_id, arxiv_root, builder)
                else:
                    record = None

            self.call_from_thread(self._on_paper_loaded, record, arxiv_id, strategy)

        threading.Thread(target=_do, daemon=True).start()

    def _on_paper_loaded(self, record: dict | None, arxiv_id: str, strategy: str) -> None:
        if record is None:
            self._update_status(f"[red]Paper '{arxiv_id}' not found.[/red]")
            return
        record = self._attach_cached_target(record)
        self._record_cache[(arxiv_id, strategy)] = record
        title = record.get("title", "")
        n_refs = len(record.get("refs", []))
        try:
            self.query_one("#paper-title", Label).update(f"[dim]{title[:70]}[/dim]")
        except NoMatches:
            pass
        self._refresh_prompt_panel()
        self._refresh_target_panel()
        self._update_status(
            f"[green]Ready:[/green] {arxiv_id} / {strategy}  {n_refs} refs  "
            f"\"{title[:40]}\" — press Enter on an experiment to generate."
        )

    def _refresh_prompt_panel(self) -> None:
        record = self._current_record
        if record is None:
            return
        prompt = record.get("prompt", "")
        if self._prompt_summary:
            prompt = _short_prompt(prompt)
        system = record.get("system", "")
        strategy = self._current_strategy
        sys_display = system[:300] + "…" if self._prompt_summary and len(system) > 300 else system
        text = (f"[bold blue]System:[/bold blue] {sys_display}\n\n"
                f"[bold blue]User[/bold blue] [dim]({strategy}):[/dim]\n{prompt}")
        try:
            self.query_one("#prompt-content", Static).update(text)
        except NoMatches:
            pass

    def _abstract_target_text(self, record: dict) -> str:
        parts = [
            f"[bold blue]Paper[/bold blue]\n{escape(record.get('title', '') or '(untitled)')}",
            f"[bold blue]Abstract[/bold blue]\n{escape(record.get('abstract', '') or '(missing abstract)')}",
        ]
        if record.get("target_proposal"):
            parts.append(
                "[bold blue]Existing abstract-CoT target_proposal[/bold blue]\n"
                + escape(record["target_proposal"])
            )
        return "\n\n".join(parts)

    def _tex_target_text(self, record: dict) -> str:
        status = record.get("tex_status")
        target = record.get("target_impl_proposal")
        if target:
            quality = record.get("target_quality") or {}
            metrics = []
            if record.get("synthesis_model"):
                metrics.append(f"model={record['synthesis_model']}")
            if record.get("tex_section_count") is not None:
                metrics.append(f"sections={record.get('tex_section_count')}")
            if record.get("tex_impl_leakage_score") is not None:
                metrics.append(f"abstract_sim={record.get('tex_impl_leakage_score')}")
            if quality.get("word_count") is not None:
                metrics.append(f"words={quality.get('word_count')}")
            header = "  ".join(metrics) or "cached"
            return (
                f"[bold green]TeX-grounded synthesis[/bold green] [dim]{escape(header)}[/dim]\n\n"
                f"{escape(target)}"
            )

        if status and status != "ok":
            reason = record.get("skip_reason") or status
            tex_dir = record.get("tex_dir") or record.get("paper_dir") or ""
            return (
                f"[bold yellow]Skipped for TeX synthesis[/bold yellow]\n"
                f"reason: {escape(str(reason))}\n"
                f"path: {escape(str(tex_dir))}\n\n"
                "This paper remains visible through the original abstract target, "
                "but it will not enter the TeX-grounded target dataset."
            )

        return (
            "[dim]No TeX synthesis cached for this paper.\n"
            "Press 't' to synthesize from the local TeX source. "
            "If no TeX exists, this panel will record the skip reason.[/dim]"
        )

    def _refresh_target_panel(self) -> None:
        record = self._current_record
        if record is None:
            return
        try:
            self.query_one("#target-abstract-content", Static).update(
                self._abstract_target_text(record)
            )
            self.query_one("#target-tex-content", Static).update(
                self._tex_target_text(record)
            )
        except NoMatches:
            pass

    # ── Experiment selection → generation ─────────────────────────────────────

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Enter pressed on a list item — trigger generation."""
        self._do_generate()

    def _selected_experiment(self) -> dict | None:
        try:
            lv = self.query_one("#exp-list", ListView)
            if lv.index is None:
                return None
            return self._experiments[lv.index]
        except (NoMatches, IndexError):
            return None

    def _do_generate(self) -> None:
        if self._generating:
            self._update_status("[yellow]Generation already in progress…[/yellow]")
            return
        record = self._current_record
        if record is None:
            self._update_status(
                "[yellow]No paper loaded — press 'a', enter arXiv ID, press Enter first.[/yellow]"
            )
            return
        exp = self._selected_experiment()
        if exp is None:
            self._update_status("[yellow]No experiment selected.[/yellow]")
            return

        # In compare mode, if left already has output send to right
        panel = "left"
        if self._compare_mode:
            try:
                lw = self.query_one("#response-left-wrap")
                if "done" in lw.classes:
                    panel = "right"
            except NoMatches:
                pass

        self._run_generation(exp, record, panel)

    # ── Actions ───────────────────────────────────────────────────────────────

    def action_generate(self) -> None:
        self._do_generate()

    def action_regenerate(self) -> None:
        # Clear both panels and regenerate
        for side in ("left", "right"):
            try:
                self.query_one(f"#response-{side}-wrap").remove_class("done", "error", "generating")
                self.query_one(f"#response-{side}-header", Label).update(f"· ready")
                self.query_one(f"#response-{side}-content", Static).update("")
            except NoMatches:
                pass
        self._do_generate()

    def action_focus_arxiv(self) -> None:
        try:
            self.query_one("#arxiv-input", Input).focus()
        except NoMatches:
            pass

    def action_blur_arxiv(self) -> None:
        try:
            self.query_one("#exp-list", ListView).focus()
        except NoMatches:
            pass

    def action_toggle_compare(self) -> None:
        self._compare_mode = not self._compare_mode
        try:
            area = self.query_one("#response-area")
            if self._compare_mode:
                area.add_class("compare")
                self._update_status("Compare mode ON — select a different experiment and press Enter.")
            else:
                area.remove_class("compare")
                self._update_status("Compare mode OFF.")
        except NoMatches:
            pass

    def action_toggle_prompt(self) -> None:
        self._prompt_summary = not self._prompt_summary
        self._refresh_prompt_panel()
        mode = "summary (8 refs)" if self._prompt_summary else "full"
        self._update_status(f"Prompt display: {mode}")

    def action_synthesize_target(self) -> None:
        if self._target_generating:
            self._update_status("[yellow]TeX synthesis already in progress...[/yellow]")
            return
        record = self._current_record
        if record is None:
            self._update_status("[yellow]Load a paper before synthesizing a TeX target.[/yellow]")
            return

        self._target_generating = True
        try:
            self.query_one("#target-tex-header", Label).update("TeX synthesis target  [dim]running[/dim]")
            self.query_one("#target-tex-content", Static).update(
                "[dim]Synthesizing from local TeX source...[/dim]"
            )
        except NoMatches:
            pass
        self._update_status(
            f"Synthesizing TeX target for {record.get('arxiv_id')} with {self._target_model}..."
        )

        t0 = time.time()

        def _worker() -> None:
            try:
                from data.synthesize_tex_targets import append_jsonl, synthesize_tex_target

                result = synthesize_tex_target(
                    record,
                    self._cfg,
                    model=self._target_model,
                    max_tokens=self._target_max_tokens,
                    temperature=self._target_temperature,
                    max_tex_chars=self._max_tex_chars,
                    connect_timeout=2.0,
                    chunk_timeout=180.0,
                )
                elapsed = time.time() - t0
                self.call_from_thread(
                    self._update_status,
                    f"Synthesizing TeX target for {record.get('arxiv_id')}... done ({elapsed:.0f}s)",
                )
                append_jsonl(self._target_cache_path, result)
                self.call_from_thread(self._on_target_done, result)
            except Exception as exc:
                msg = str(exc) or repr(exc)
                self.call_from_thread(self._on_target_error, msg)

        threading.Thread(target=_worker, daemon=True).start()

    def action_save(self) -> None:
        record = self._current_record
        if record is None:
            self._update_status("[yellow]Nothing to save — load a paper first.[/yellow]")
            return
        out: dict = {
            "arxiv_id": record.get("arxiv_id"),
            "title": record.get("title"),
            "strategy": self._current_strategy,
            "system": record.get("system"),
            "prompt": record.get("prompt"),
            "abstract": record.get("abstract"),
            "target_source": record.get("target_source"),
            "tex_status": record.get("tex_status"),
            "target_impl_proposal": record.get("target_impl_proposal"),
            "cot_impl_proposal": record.get("cot_impl_proposal"),
            "target_quality": record.get("target_quality"),
            "tex_impl_leakage_score": record.get("tex_impl_leakage_score"),
            "responses": {},
        }
        for side in ("left", "right"):
            try:
                label_w = self.query_one(f"#response-{side}-header", Label)
                body_w  = self.query_one(f"#response-{side}-content", Static)
                label = str(label_w.renderable)
                body  = str(body_w.renderable)
                if body and not body.startswith("[dim]"):
                    out["responses"][label] = body
            except NoMatches:
                pass
        arxiv_id = record.get("arxiv_id", "unknown")
        strategy = self._current_strategy
        save_path = (self._runs_dir / "demo_saves"
                     / f"{arxiv_id}_{strategy}_{int(time.time())}.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        self._update_status(f"Saved to {save_path}")

    # ── Generation worker ─────────────────────────────────────────────────────

    def _run_generation(self, exp: dict, record: dict, panel: str) -> None:
        try:
            wrap    = self.query_one(f"#response-{panel}-wrap")
            header  = self.query_one(f"#response-{panel}-header", Label)
            content = self.query_one(f"#response-{panel}-content", Static)
        except NoMatches:
            return

        wrap.remove_class("done", "error")
        wrap.add_class("generating")
        header.update(f"⟳ {exp['label']}")
        content.update("")
        self._generating = True
        self._update_status(f"Generating with {exp['label']} — please wait…")

        def _worker() -> None:
            try:
                t0 = time.time()
                _scripts = str(Path(__file__).resolve().parent)
                if _scripts not in sys.path:
                    sys.path.insert(0, _scripts)
                if exp["is_api"]:
                    from probe import generate_baseline
                    response, elapsed = generate_baseline(record, exp["model"], 2048)
                else:
                    import torch
                    from probe import generate, load_model_and_tokenizer
                    self.call_from_thread(
                        self._update_status,
                        f"Loading model weights for {exp['label']}…",
                    )
                    model, tokenizer, device = load_model_and_tokenizer(
                        Path(exp["checkpoint"]), None, self._cfg
                    )
                    self.call_from_thread(
                        self._update_status,
                        f"Running inference for {exp['label']}…",
                    )
                    response = generate(model, tokenizer, record, 2048, device)
                    elapsed = time.time() - t0

                # Optional FAS scoring
                metrics = f"{elapsed:.1f}s"
                try:
                    from eval.fas import extract_proposal_text, get_fas_evaluator, load_index
                    index_file = self._runs_dir / "eval" / "test_index.npz"
                    if index_file.exists():
                        fas_index = load_index(index_file)
                        fas_eval  = get_fas_evaluator(self._cfg)
                        proposal  = extract_proposal_text(response)
                        fas = fas_eval.score(proposal, record.get("arxiv_id", ""), fas_index)
                        recall = "✓" if fas.get("recall_at_k") else "✗"
                        metrics = (
                            f"FAS={fas['FAS']:.3f}  recall={recall}  "
                            f"sim={fas.get('mean_sim', 0):.3f}  {elapsed:.1f}s"
                        )
                except Exception:
                    pass

                self.call_from_thread(self._on_done, panel, exp["label"], response, metrics)
            except Exception as exc:
                self.call_from_thread(self._on_error, panel, str(exc))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_done(self, panel: str, label: str, response: str, metrics: str) -> None:
        try:
            wrap    = self.query_one(f"#response-{panel}-wrap")
            header  = self.query_one(f"#response-{panel}-header", Label)
            content = self.query_one(f"#response-{panel}-content", Static)
            wrap.remove_class("generating", "error")
            wrap.add_class("done")
            header.update(f"✓ {label}  [dim]{metrics}[/dim]")
            content.update(response)
        except NoMatches:
            pass
        self._generating = False
        self._update_status(f"[green]Done.[/green]  {metrics}")

    def _on_error(self, panel: str, error: str) -> None:
        try:
            wrap    = self.query_one(f"#response-{panel}-wrap")
            header  = self.query_one(f"#response-{panel}-header", Label)
            content = self.query_one(f"#response-{panel}-content", Static)
            wrap.remove_class("generating", "done")
            wrap.add_class("error")
            header.update("✗ Error")
            content.update(f"[red]{error}[/red]")
        except NoMatches:
            pass
        self._generating = False
        self._update_status(f"[red]Error: {error[:100]}[/red]")

    def _on_target_done(self, result: dict) -> None:
        arxiv_id = result.get("arxiv_id", "")
        if arxiv_id:
            self._target_cache[arxiv_id] = result
        current = self._current_record
        if current is not None and current.get("arxiv_id") == arxiv_id:
            merged = dict(current)
            for key in TEX_TARGET_FIELDS:
                if key in result:
                    merged[key] = result[key]
            self._record_cache[(arxiv_id, self._current_strategy)] = merged
        try:
            self.query_one("#target-tex-header", Label).update("TeX synthesis target")
        except NoMatches:
            pass
        self._target_generating = False
        self._refresh_target_panel()
        if result.get("tex_status") == "ok" and result.get("target_impl_proposal"):
            self._update_status(
                f"[green]TeX target cached:[/green] {arxiv_id} -> {self._target_cache_path}"
            )
        else:
            self._update_status(
                f"[yellow]TeX target skipped:[/yellow] {arxiv_id} ({result.get('tex_status')})"
            )

    def _on_target_error(self, error: str) -> None:
        try:
            self.query_one("#target-tex-header", Label).update("TeX synthesis target  [red]error[/red]")
            self.query_one("#target-tex-content", Static).update(f"[red]{escape(error)}[/red]")
        except NoMatches:
            pass
        self._target_generating = False
        self._update_status(f"[red]TeX synthesis error: {error[:100]}[/red]")

    def _update_status(self, text: str) -> None:
        try:
            self.query_one("#status", Static).update(text)
        except NoMatches:
            pass


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--target-cache",
        default=None,
        help="Append/read TeX target previews here (default: runs/dataset/demo_tex_impl_targets.jsonl)",
    )
    parser.add_argument("--target-model", default=None, help="Claude model for TeX target synthesis")
    parser.add_argument("--target-max-tokens", type=int, default=None)
    parser.add_argument("--target-temperature", type=float, default=0.3)
    parser.add_argument("--max-tex-chars", type=int, default=24000)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    runs_dir = Path(cfg["runs_dir"])

    ProposalDemoApp(
        cfg=cfg,
        runs_dir=runs_dir,
        target_cache_path=Path(args.target_cache) if args.target_cache else None,
        target_model=args.target_model,
        target_max_tokens=args.target_max_tokens,
        target_temperature=args.target_temperature,
        max_tex_chars=args.max_tex_chars,
    ).run()


if __name__ == "__main__":
    main()
