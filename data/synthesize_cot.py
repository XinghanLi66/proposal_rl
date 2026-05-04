#!/usr/bin/env python3
"""
Synthesize Chain-of-Thought research proposals via the Claude API.

For each record in the base dataset (train.jsonl), we call Claude with:
  - the reference list (already formatted as `prompt`)
  - a carefully designed prompt that prevents leakage of the actual paper

Output appends two fields to each record:
  - "target_proposal": a clean leakage-stripped proposal (used as SFT target)
  - "cot_proposal":    same but with <thinking> chain of thought (longer SFT target)
  - "leakage_score":   cosine similarity between target_proposal and actual abstract
                       (records with score > 0.85 are flagged)

Resumable: already-synthesized arxiv_ids are skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

import anthropic
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---- Prompt templates --------------------------------------------------------

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

# ---- Leakage detection -------------------------------------------------------

def simple_cosine(a: str, b: str) -> float:
    """Very lightweight word-overlap cosine for quick leakage screening."""
    def bow(text):
        words = re.findall(r'\b\w+\b', text.lower())
        vec = {}
        for w in words:
            vec[w] = vec.get(w, 0) + 1
        return vec

    va, vb = bow(a), bow(b)
    keys = set(va) | set(vb)
    if not keys:
        return 0.0
    dot = sum(va.get(k, 0) * vb.get(k, 0) for k in keys)
    na = sum(v ** 2 for v in va.values()) ** 0.5
    nb = sum(v ** 2 for v in vb.values()) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def extract_proposal_text(response: str) -> str:
    """Extract just the <proposal>...</proposal> block for leakage check."""
    m = re.search(r"<proposal>(.*?)</proposal>", response, re.DOTALL)
    if m:
        # Strip XML tags for text comparison
        return re.sub(r"<[^>]+>", " ", m.group(1)).strip()
    return response


# ---- Async API client --------------------------------------------------------

def _extract_ref_text(prompt: str) -> str:
    """Extract the core reference block from a formatted prompt string."""
    # USER_TEMPLATE: "Below are N papers ... \n\n{ref_block}\n\nGenerate a structured"
    m = re.search(r"Below are \d+ papers.*?\n\n(.*?)\n\nGenerate a structured", prompt, re.DOTALL)
    if m:
        return m.group(1)
    # RELATED_WORK_TEMPLATE: "... studying the following area ...\n\n{narrative}\n\nBased on"
    m = re.search(r"studying the following area[^\n]*\n\n(.*?)\n\nBased on this", prompt, re.DOTALL)
    if m:
        return m.group(1)
    return prompt[:3000]


async def synthesize_one(
    client: anthropic.AsyncAnthropic,
    record: dict,
    model: str,
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    ref_block = record.get("prompt", "")
    ref_text = _extract_ref_text(ref_block)

    abstract = record.get("abstract", "")

    async with semaphore:
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=SYNTHESIS_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": SYNTHESIS_USER.format(ref_block=ref_text, abstract=abstract),
                }],
            )
            text = response.content[0].text
        except Exception as e:
            log.warning(f"API error for {record['arxiv_id']}: {e}")
            return None

    proposal_text = extract_proposal_text(text)
    leakage = simple_cosine(proposal_text, abstract)

    return {
        **record,
        "cot_proposal": text,
        "target_proposal": proposal_text,
        "leakage_score": round(leakage, 4),
        "leakage_flagged": leakage > 0.85,
    }


async def run_synthesis(
    records: list[dict],
    output_file: Path,
    done_ids: set[str],
    model: str,
    max_tokens: int,
    temperature: float,
    concurrency: int,
) -> None:
    pending = [r for r in records if r["arxiv_id"] not in done_ids]
    log.info(f"Synthesizing {len(pending)} records (skip={len(done_ids)})")

    # Uses ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN from environment (local service)
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(concurrency)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    written = flagged = errors = 0
    start = time.time()

    with open(output_file, "a") as out_f:
        tasks = [
            asyncio.create_task(
                synthesize_one(client, r, model, max_tokens, temperature, semaphore)
            )
            for r in pending
        ]
        for i, fut in enumerate(asyncio.as_completed(tasks)):
            result = await fut
            if result is None:
                errors += 1
            else:
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
                written += 1
                if result["leakage_flagged"]:
                    flagged += 1

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                eta = (len(pending) - i - 1) / rate if rate > 0 else 0
                log.info(
                    f"Progress {i+1}/{len(pending)} | written={written} "
                    f"flagged={flagged} errors={errors} | "
                    f"{rate:.1f}/s | ETA {eta/60:.0f}m"
                )

    log.info(f"Done. written={written} flagged={flagged} errors={errors}")


def load_done(output_file: Path) -> set[str]:
    done = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["arxiv_id"])
                except Exception:
                    pass
    return done


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--input", help="Override input JSONL (default: runs/dataset/train.jsonl)")
    parser.add_argument("--output", help="Override output JSONL (default: runs/dataset/train_cot.jsonl)")
    parser.add_argument("--strategy", default=None,
                        help="Rebuild prompts from refs using this strategy before synthesis. "
                             "Allows per-experiment CoT synthesis aligned with the training prompt format.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_AUTH_TOKEN") and not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError(
            "No Claude credentials found. "
            "Export ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL (local service) "
            "or ANTHROPIC_API_KEY (Anthropic API)."
        )

    cfg = load_config(args.config)
    runs_dir = Path(cfg["runs_dir"])

    input_file = Path(args.input) if args.input else runs_dir / "dataset" / "train.jsonl"
    output_file = Path(args.output) if args.output else runs_dir / "dataset" / "train_cot.jsonl"

    cot_cfg = cfg.get("cot", {})
    model = cot_cfg.get("model", "claude-sonnet-4-6")
    max_tokens = cot_cfg.get("max_tokens", 2048)
    temperature = cot_cfg.get("temperature", 0.8)
    concurrency = cot_cfg.get("concurrency", 8)

    records = []
    with open(input_file) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    if args.limit:
        records = records[:args.limit]

    log.info(f"Loaded {len(records)} records from {input_file}")

    # Optionally rebuild prompts using a different strategy so synthesis input
    # aligns with the RL training prompt format for this experiment.
    if args.strategy:
        import sys
        _repo = str(Path(__file__).resolve().parent.parent)
        if _repo not in sys.path:
            sys.path.insert(0, _repo)
        from train.prompt_builder import get_builder
        pb_cfg = cfg.get("prompt_builder", {}).copy()
        pb_cfg["strategy"] = args.strategy
        pb_cfg["runs_dir"] = str(runs_dir)
        builder = get_builder({**cfg, "prompt_builder": pb_cfg})
        log.info(f"Rebuilding prompts with strategy={args.strategy} ({type(builder).__name__})")
        rebuilt = []
        for r in records:
            rebuilt.append({**r, "system": builder.system(), "prompt": builder.build(r)})
        records = rebuilt
        log.info(f"Rebuilt {len(records)} prompts")

    done_ids = load_done(output_file)

    asyncio.run(run_synthesis(
        records, output_file, done_ids,
        model, max_tokens, temperature, concurrency,
    ))


if __name__ == "__main__":
    main()
