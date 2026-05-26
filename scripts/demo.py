#!/usr/bin/env python3
"""
Demo tool: print prompt and generated proposal for a checkpoint + arxiv_id.

Usage:
  # Auto-discover latest checkpoint by experiment name prefix
  python scripts/demo.py 2601.12345 --exp exp13_full_refs_sft_rl

  # Explicit checkpoint path
  python scripts/demo.py 2601.12345 --checkpoint runs/exps/exp13_.../rl/final

  # Prompt only, no model loading
  python scripts/demo.py 2601.12345 --prompt-only

  # Save result to JSON
  python scripts/demo.py 2601.12345 --exp exp13 --save out.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from probe import (
    generate,
    generate_baseline,
    load_model_and_tokenizer,
    load_record_from_arxiv_store,
    load_record_from_dataset,
)
from train.prompt_builder import get_builder

console = Console()


def _resolve_checkpoint(exp_prefix: str, runs_dir: Path) -> Path:
    """Find the latest rl/final checkpoint matching an experiment name prefix."""
    candidates = sorted(
        (runs_dir / "exps").glob(f"{exp_prefix}*/rl/final"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            (runs_dir / "exps").glob(f"*{exp_prefix}*/rl/final"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    if not candidates:
        console.print(f"[red]No checkpoint found for experiment prefix '{exp_prefix}'[/red]")
        console.print(f"  Searched: {runs_dir / 'exps' / (exp_prefix + '*/rl/final')}")
        sys.exit(1)
    if len(candidates) > 1:
        console.print(
            f"[yellow]Multiple checkpoints found for '{exp_prefix}', using latest:[/yellow]\n"
            f"  {candidates[0]}"
        )
    return candidates[0]


def _load_record(arxiv_id: str, runs_dir: Path, cfg: dict, strategy: str | None) -> dict:
    pb_cfg = cfg.get("prompt_builder", {}).copy()
    pb_cfg["runs_dir"] = str(runs_dir)
    if strategy:
        pb_cfg["strategy"] = strategy
    builder = get_builder({**cfg, "prompt_builder": pb_cfg})

    record = load_record_from_dataset(arxiv_id, runs_dir)
    if record and strategy and record.get("refs"):
        record["system"] = builder.system()
        record["prompt"] = builder.build(record)
    if record is None:
        arxiv_root = Path(cfg.get("arxiv_root", ""))
        if arxiv_root.exists():
            record = load_record_from_arxiv_store(arxiv_id, arxiv_root, builder)
    if record is None:
        console.print(f"[red]Could not find paper '{arxiv_id}' in dataset or arxiv store.[/red]")
        sys.exit(1)
    return record


def _print_record_header(record: dict) -> None:
    console.print()
    console.print(Rule(f"[bold cyan]{record.get('arxiv_id', '?')}[/bold cyan]"))
    console.print(f"  [bold]Title:[/bold]   {record.get('title', '?')[:100]}")
    console.print(f"  [bold]Created:[/bold] {str(record.get('created', '?'))[:10]}")
    n_refs = len(record.get("refs", []))
    if n_refs:
        console.print(f"  [bold]Refs:[/bold]    {n_refs}")
    console.print()


def _print_prompt(record: dict) -> None:
    console.print(Panel(
        Text(record["system"], style="dim"),
        title="[bold]System Prompt[/bold]",
        border_style="blue",
        expand=True,
    ))
    console.print(Panel(
        record["prompt"],
        title="[bold]User Prompt[/bold]",
        border_style="blue",
        expand=True,
    ))


def _print_response(label: str, response: str, elapsed: float | None = None) -> None:
    title = f"[bold green]{label}[/bold green]"
    if elapsed is not None:
        title += f" [dim]({elapsed:.1f}s)[/dim]"
    console.print(Panel(response, title=title, border_style="green", expand=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arxiv_id", help="arXiv ID (e.g. 2601.12345)")
    parser.add_argument("--exp", metavar="PREFIX",
                        help="Experiment name prefix (auto-discovers latest rl/final checkpoint)")
    parser.add_argument("--checkpoint", metavar="PATH",
                        help="Explicit path to model checkpoint (overrides --exp)")
    parser.add_argument("--baseline", action="store_true",
                        help="Also run claude-opus-4-6 baseline for comparison")
    parser.add_argument("--baseline-model", default="claude-opus-4-6")
    parser.add_argument("--prompt-only", action="store_true",
                        help="Print prompt only; skip model loading and generation")
    parser.add_argument("--strategy", default=None,
                        help="Prompt-builder strategy override (default: from config)")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--save", metavar="PATH",
                        help="Save result to JSON file")
    args = parser.parse_args()

    if not args.checkpoint and not args.exp and not args.prompt_only and not args.baseline:
        parser.error("one of --exp, --checkpoint, --baseline, or --prompt-only is required")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    runs_dir = Path(cfg["runs_dir"])

    # Resolve checkpoint
    checkpoint = None
    checkpoint_label = "prompt-only"
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        checkpoint_label = checkpoint.name
    elif args.exp:
        checkpoint = _resolve_checkpoint(args.exp, runs_dir)
        checkpoint_label = args.exp

    record = _load_record(args.arxiv_id, runs_dir, cfg, args.strategy)
    _print_record_header(record)
    _print_prompt(record)

    if args.prompt_only:
        return

    save_data: dict = {
        "arxiv_id": record.get("arxiv_id"),
        "title": record.get("title"),
        "system": record["system"],
        "prompt": record["prompt"],
        "checkpoint": str(checkpoint) if checkpoint else None,
        "responses": {},
    }

    if args.baseline:
        import time
        console.print(f"[dim]Running {args.baseline_model}...[/dim]")
        t0 = time.time()
        # Suppress anthropic client debug output
        os.environ.setdefault("ANTHROPIC_LOG", "error")
        response, elapsed = generate_baseline(record, args.baseline_model, args.max_new_tokens)
        _print_response(f"Baseline: {args.baseline_model}", response, elapsed)
        save_data["responses"][f"baseline:{args.baseline_model}"] = response

    if checkpoint:
        import time
        console.print(f"[dim]Loading model from {checkpoint}...[/dim]")
        model, tokenizer, device = load_model_and_tokenizer(checkpoint, None, cfg)
        console.print("[dim]Generating...[/dim]")
        t0 = time.time()
        response = generate(model, tokenizer, record, args.max_new_tokens, device)
        elapsed = time.time() - t0
        _print_response(checkpoint_label, response, elapsed)
        save_data["responses"][checkpoint_label] = response

    if args.save:
        Path(args.save).write_text(json.dumps(save_data, indent=2, ensure_ascii=False))
        console.print(f"\n[dim]Saved to {args.save}[/dim]")


if __name__ == "__main__":
    main()
