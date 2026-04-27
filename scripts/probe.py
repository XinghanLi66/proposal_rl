#!/usr/bin/env python3
"""
Manually inspect a checkpoint's proposal for a single paper.

Given a checkpoint and a paper (by arxiv_id, metadata.json path, or dataset
record path), reconstructs the exact training prompt and runs generation.

Usage examples:
  # By arxiv_id (looks up in dataset splits)
  python scripts/probe.py --checkpoint runs/grpo/final --paper 2501.12345

  # CoT synthesis via Claude API (same prompt as SFT training data)
  python scripts/probe.py --cot --paper 2601.18346

  # Both: compare checkpoint output side-by-side with Claude CoT
  python scripts/probe.py --checkpoint runs/grpo/final --cot --paper 2601.18346

  # From a dataset JSONL record (pre-built prompt, fastest)
  python scripts/probe.py --checkpoint runs/grpo/final --record runs/dataset/test.jsonl --index 0

  # Print prompt only, no generation
  python scripts/probe.py --paper 2501.12345 --prompt-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.fas import extract_proposal_text, get_fas_evaluator, load_index
from train.prompt_builder import get_builder, REGISTRY as STRATEGY_REGISTRY

# ── CoT synthesis (mirrors data/synthesize_cot.py exactly) ───────────────────

SYNTHESIS_SYSTEM = """\
You are a research advisor helping synthesize novel research directions. \
You will be given a reference list and the actual abstract of a paper that cites those references. \
Your job is to generate a plausible, forward-looking research proposal that is INSPIRED by the \
reference list — as if a researcher was about to write the paper — but does NOT reveal or copy \
the specific contributions of the actual paper.

Rules:
1. Do NOT mention the specific method names, algorithm names, or numerical results from the actual paper.
2. Do NOT use phrases that uniquely identify the actual paper (e.g., "we propose X" where X is the paper's method name).
3. The proposal should describe a DIRECTION plausible from the references, not the paper's actual solution.
4. Use the exact XML structure specified below.
5. The <thinking> section should reflect a researcher's analytical process."""

SYNTHESIS_USER = """\
=== REFERENCE LIST ===
{ref_block}

=== ACTUAL PAPER ABSTRACT (for context — do NOT copy or reveal) ===
{abstract}

=== YOUR TASK ===
Write a research proposal as if you are the researcher who read the reference list above \
and is about to write the paper described in the abstract — but WITHOUT revealing the paper's \
specific contributions, method names, or results.

Use this exact format:

<thinking>
[Step 1: What themes and methods appear across these references?]
[Step 2: What open problems or limitations do they collectively suggest?]
[Step 3: What would be a natural next research direction?]
[Step 4: What approach might address this gap?]
</thinking>
<proposal>
<problem>What core problem should this work address? (1-3 sentences)</problem>
<gap>What specific gap in the literature motivates this? (2-4 sentences)</gap>
<key_insight>What key insight or hypothesis would drive the approach? (2-3 sentences)</key_insight>
<approach>How might the proposed method work at a high level? (3-5 sentences, no specific names)</approach>
<expected_contributions>What would be the main scientific contributions? (2-4 bullet points)</expected_contributions>
</proposal>"""


def _extract_ref_block(prompt: str) -> str:
    """Extract just the numbered reference list from a pre-built prompt."""
    m = re.search(r"Below are \d+ papers.*?\n\n(.*?)\n\nGenerate a structured", prompt, re.DOTALL)
    return m.group(1) if m else prompt[:3000]


def _word_cosine(a: str, b: str) -> float:
    def bow(text):
        words = re.findall(r'\b\w+\b', text.lower())
        v: dict[str, int] = {}
        for w in words:
            v[w] = v.get(w, 0) + 1
        return v
    va, vb = bow(a), bow(b)
    keys = set(va) | set(vb)
    dot = sum(va.get(k, 0) * vb.get(k, 0) for k in keys)
    na = sum(x**2 for x in va.values())**0.5
    nb = sum(x**2 for x in vb.values())**0.5
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


_PROXY_URL = "http://10.39.10.241:10001"


def _claude_client():
    import os
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY") or "123"
    return anthropic.Anthropic(api_key=api_key, base_url=_PROXY_URL, timeout=3000.0, max_retries=2)


def synthesize_cot(record: dict, model: str, max_tokens: int, temperature: float) -> tuple[str, float]:
    ref_block = _extract_ref_block(record.get("prompt", ""))
    abstract = record.get("abstract", "")
    t0 = time.time()
    with _claude_client().messages.stream(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=SYNTHESIS_SYSTEM,
        messages=[{"role": "user", "content": SYNTHESIS_USER.format(
            ref_block=ref_block, abstract=abstract,
        )}],
    ) as stream:
        text = stream.get_final_text()
    return text, time.time() - t0


def generate_baseline(record: dict, model: str, max_tokens: int) -> tuple[str, float]:
    t0 = time.time()
    with _claude_client().messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=record["system"],
        messages=[{"role": "user", "content": record["prompt"]}],
    ) as stream:
        text = stream.get_final_text()
    return text, time.time() - t0


# ── Record loading ─────────────────────────────────────────────────────────────

def load_record_from_dataset(arxiv_id: str, runs_dir: Path) -> dict | None:
    for split in ("test", "val", "train"):
        p = runs_dir / "dataset" / f"{split}.jsonl"
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get("arxiv_id") == arxiv_id:
                        print(f"  [found in {split} split]")
                        return d
                except Exception:
                    pass
    return None


def load_record_from_metadata(meta_path: Path, runs_dir: Path, builder) -> dict | None:
    meta = json.loads(meta_path.read_text())
    arxiv_id = meta.get("arxiv_id")
    if not arxiv_id:
        print("ERROR: no arxiv_id in metadata.json", file=sys.stderr)
        return None

    refs = None
    for split in ("test", "val", "train"):
        p = runs_dir / "dataset" / f"refs_{split}.jsonl"
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get("arxiv_id") == arxiv_id:
                        refs = d.get("refs", [])
                        print(f"  [refs found in refs_{split}.jsonl: {len(refs)} refs]")
                        break
                except Exception:
                    pass
        if refs is not None:
            break

    if not refs:
        print(f"WARNING: no refs found for {arxiv_id} — prompt will be empty", file=sys.stderr)
        refs = []

    record = {
        "arxiv_id": arxiv_id,
        "title": meta.get("title", ""),
        "abstract": meta.get("abstract", ""),
        "refs": refs,
    }
    record["system"] = builder.system()
    record["prompt"] = builder.build(record)
    return record


def load_record_from_jsonl(jsonl_path: Path, index: int) -> dict | None:
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i == index:
                try:
                    return json.loads(line)
                except Exception as e:
                    print(f"ERROR parsing line {index}: {e}", file=sys.stderr)
                    return None
    print(f"ERROR: index {index} out of range in {jsonl_path}", file=sys.stderr)
    return None


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model_and_tokenizer(checkpoint: Path, base_model: str | None, cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    base_path = base_model or cfg.get("model_name_or_path")
    candidates = [
        base_path,
        "/newcpfs/user/yuanqianhao/hf_models/Qwen/Qwen2.5-7B-Instruct",
        "/newcpfs/user/gaochaochen/huimu/CodePrMP/models/Qwen2.5-7B-Instruct",
    ]
    resolved_base = None
    for p in candidates:
        if p and Path(p).exists() and (Path(p) / "config.json").exists():
            resolved_base = p
            break

    tok_path = resolved_base or str(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        if resolved_base is None:
            raise ValueError("no base model path")
        print(f"Loading base: {resolved_base}")
        base = AutoModelForCausalLM.from_pretrained(
            resolved_base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        print(f"Loading LoRA adapter: {checkpoint}")
        model = PeftModel.from_pretrained(base, str(checkpoint)).merge_and_unload()
        print("Loaded as LoRA checkpoint (merged)")
    except Exception as e:
        print(f"LoRA load failed ({e}), trying full model...")
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        print("Loaded as full model checkpoint")

    model.eval()
    return model, tokenizer, device


# ── Generation ────────────────────────────────────────────────────────────────

def generate(model, tokenizer, record: dict, max_new_tokens: int, device: str) -> str:
    messages = [
        {"role": "system", "content": record["system"]},
        {"role": "user",   "content": record["prompt"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=None,
                        help="Path to model checkpoint (LoRA adapter or full HF model)")
    parser.add_argument("--base-model", default=None,
                        help="Base model path (for LoRA checkpoints if not in config)")
    parser.add_argument("--cot", action="store_true",
                        help="Synthesize a CoT proposal via Claude API (same prompt as SFT training)")
    parser.add_argument("--cot-model", default=None,
                        help="Claude model for CoT synthesis (default: from config)")
    parser.add_argument("--baseline", action="store_true",
                        help="Run claude-opus-4-6 on the inference prompt as a baseline")
    parser.add_argument("--baseline-model", default="claude-opus-4-6")
    parser.add_argument("--no-fas", action="store_true", help="Skip FAS evaluation")
    parser.add_argument("--index-split", default="test", choices=["test", "val"])
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--strategy", default=None,
                        choices=list(STRATEGY_REGISTRY),
                        help="Prompt-builder strategy (default: read from config, or full_refs). "
                             "LLM-based strategies (top_k_refs, related_work, "
                             "with_research_question, top_k_related_work) call the API.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--paper", help="arxiv_id or path to metadata.json")
    group.add_argument("--record", help="Path to a dataset JSONL file; use with --index")

    parser.add_argument("--index", type=int, default=0, help="Line index in --record JSONL")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--prompt-only", action="store_true")
    args = parser.parse_args()

    if not args.checkpoint and not args.cot and not args.baseline and not args.prompt_only:
        parser.error("at least one of --checkpoint, --cot, --baseline, or --prompt-only is required")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    runs_dir = Path(cfg["runs_dir"])

    # Build prompt builder — CLI --strategy overrides config
    pb_cfg = cfg.get("prompt_builder", {}).copy()
    pb_cfg["runs_dir"] = str(runs_dir)
    if args.strategy:
        pb_cfg["strategy"] = args.strategy
    builder = get_builder({**cfg, "prompt_builder": pb_cfg})
    print(f"Prompt strategy: {pb_cfg.get('strategy', 'full_refs')}  [{type(builder).__name__}]")

    # Load record
    record = None
    if args.record:
        print(f"Loading record #{args.index} from {args.record}")
        record = load_record_from_jsonl(Path(args.record), args.index)
        # Optionally rebuild prompt with the requested strategy (if explicitly set)
        if args.strategy and record is not None and record.get("refs"):
            record["system"] = builder.system()
            record["prompt"] = builder.build(record)
    else:
        p = Path(args.paper)
        if p.exists() and p.suffix == ".json":
            print(f"Loading from metadata.json: {p}")
            record = load_record_from_metadata(p, runs_dir, builder)
        else:
            print(f"Looking up arxiv_id={args.paper} in dataset...")
            record = load_record_from_dataset(args.paper, runs_dir)
            # Rebuild prompt with the requested strategy if refs are available
            if args.strategy and record is not None and record.get("refs"):
                record["system"] = builder.system()
                record["prompt"] = builder.build(record)

    if record is None:
        print("ERROR: could not load paper record.", file=sys.stderr)
        sys.exit(1)

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Paper  : {record.get('arxiv_id', '?')}")
    print(f"  Title  : {record.get('title', '?')[:80]}")
    print(f"  Created: {record.get('created', '?')[:10]}")
    n_refs = len(record.get("refs", []))
    if n_refs:
        print(f"  Refs   : {n_refs}")
    print(sep)

    print("\n── SYSTEM PROMPT ──────────────────────────────────────────────────────")
    print(record["system"])
    print("\n── USER PROMPT ────────────────────────────────────────────────────────")
    print(record["prompt"])
    print(sep)

    if args.prompt_only:
        print("\n[--prompt-only: skipping generation]")
        return

    # Load FAS evaluator and index once (shared across all outputs)
    fas_evaluator = fas_index = None
    if not args.no_fas:
        index_file = runs_dir / "eval" / f"{args.index_split}_index.npz"
        if index_file.exists():
            print(f"\nLoading FAS index ({args.index_split})...")
            fas_index = load_index(index_file)
            fas_evaluator = get_fas_evaluator(cfg)
            print(f"  FAS strategy: {type(fas_evaluator).__name__}")
        else:
            print(f"  [FAS index not found at {index_file}, skipping FAS]")

    def eval_and_print_fas(label: str, output_text: str) -> None:
        if fas_evaluator is None:
            return
        proposal_text = extract_proposal_text(output_text)
        fas = fas_evaluator.score(proposal_text, record.get("arxiv_id", ""), fas_index)
        hit = "✓ hit" if fas["recall_at_k"] else "✗ miss"
        print(f"  FAS={fas['FAS']:.4f}  recall@{fas['topk']}={hit}  "
              f"{'mean_sim' if 'mean_sim' in fas else 'max_judge'}="
              f"{fas.get('mean_sim', fas.get('max_judge_score', 0)):.4f}  [{label}]")

    # ── Baseline: claude-opus-4-6 on the inference prompt ─────────────────────
    if args.baseline:
        print(f"\nRunning baseline ({args.baseline_model})...")
        baseline_output, baseline_t = generate_baseline(record, args.baseline_model, args.max_new_tokens)
        print(f"\n── BASELINE ({args.baseline_model})  [{baseline_t:.1f}s] ──────────────────────")
        print(baseline_output)
        eval_and_print_fas(f"baseline:{args.baseline_model}", baseline_output)
        print(sep)

    # ── CoT synthesis via Claude API ──────────────────────────────────────────
    if args.cot:
        cot_cfg = cfg.get("cot", {})
        cot_model = args.cot_model or cot_cfg.get("model", "claude-sonnet-4-6")
        max_tokens = cot_cfg.get("max_tokens", 2048)
        temperature = cot_cfg.get("temperature", 0.8)
        print(f"\nSynthesizing CoT via {cot_model}...")
        cot_output, cot_t = synthesize_cot(record, cot_model, max_tokens, temperature)
        proposal_text = extract_proposal_text(cot_output)
        leakage = _word_cosine(proposal_text, record.get("abstract", ""))
        print(f"\n── COT SYNTHESIS ({cot_model})  [{cot_t:.1f}s] ────────────────────────")
        print(cot_output)
        print(f"\n  leakage score: {leakage:.4f}{'  ⚠ flagged' if leakage > 0.85 else ''}")
        eval_and_print_fas(f"cot:{cot_model}", cot_output)
        print(sep)

    # ── Checkpoint generation ─────────────────────────────────────────────────
    if args.checkpoint:
        print(f"\nCheckpoint: {args.checkpoint}")
        model, tokenizer, device = load_model_and_tokenizer(
            Path(args.checkpoint), args.base_model, cfg
        )
        print("\nGenerating...")
        output = generate(model, tokenizer, record, args.max_new_tokens, device)
        print(f"\n── MODEL OUTPUT ({Path(args.checkpoint).name}) ──────────────────────────────────")
        print(output)
        eval_and_print_fas(Path(args.checkpoint).name, output)
        print(sep)


if __name__ == "__main__":
    main()
