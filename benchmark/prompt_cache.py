"""
Task-level prompt cache for the benchmark pipeline.

Prompts are task-wide — all samples with the same (task, strategy) get the
exact same prompt, so we build it once and cache it to disk.

Ref list construction:
  [frontline_paper_1, frontline_paper_2, frontline_paper_3]   ← the 3 frontlines first
  + union of all their S2 reference lists (deduped, frontlines excluded)

The combined list is fed to whichever prompt builder strategy is selected.
k = 5 for all LLM-based strategies (top_k_refs, top_k_related_work, etc.).

Cache layout:
  <runs_dir>/benchmark/prompt_cache/<task_name>__combined_refs.json   ← S2 refs (shared)
  <runs_dir>/benchmark/prompt_cache/<task_name>__<strategy>.json      ← per-strategy prompt

The inner LLM calls made by prompt_builder (selection indices, related-work
narratives) are cached separately in
  <runs_dir>/dataset/prompt_cache/
keyed by the synthetic arxiv_id  "task:<task_name>:combined:p<n_pinned>".

with_research_question strategy:
  Built by taking the full_refs prompt and inserting the pre-written task
  research question before the proposal format block — no LLM call needed.
  The question is defined in TASK_RESEARCH_QUESTIONS below.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

# Canonical strategy list — import this everywhere instead of redefining it.
ALL_STRATEGIES: list[str] = [
    "full_refs",
    "top_k_refs",
    "related_work",
    "with_research_question",
    "top_k_related_work",
]

_ARXIV_ROOT = _REPO.parent / "data" / "arxiv" / "papers"

# Marker that separates the ref block from the proposal format instructions.
# Used to insert the research question in the right place.
_PROPOSAL_MARKER = "Generate a structured research proposal using this exact format:"

# Manually-written, benchmark-specific research questions keyed by task name.
# with_research_question is built as: full_refs prompt + this question inserted
# before the proposal format — identical ref list to full_refs, no LLM call.
TASK_RESEARCH_QUESTIONS: dict[str, str] = {
    "dl_lr_schedule": (
        "Warmup-then-cosine-decay schedules, SGDR restarts, and the 1cycle policy each "
        "outperform constant-rate Adam, yet their relative advantages are poorly understood "
        "and they are rarely compared or combined on character-level language models. "
        "What learning rate schedule — or principled combination of warmup, cyclical component, "
        "and decay strategy — yields the greatest reduction in bits-per-character for a "
        "character-level autoregressive model trained from scratch with a fixed compute budget, "
        "and what training dynamics (gradient norms, loss curvature, effective step size) "
        "explain its advantage over the naive constant-rate baseline?"
    ),
}


def _fetch_paper_s2(arxiv_id: str) -> dict:
    """Fetch a single paper's title/abstract/year from the S2 API."""
    import httpx
    url = f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_id}"
    try:
        resp = httpx.get(url, params={"fields": "title,year,abstract"}, timeout=15.0)
        resp.raise_for_status()
        data = resp.json()
        return {
            "title":    (data.get("title") or "").strip(),
            "abstract": (data.get("abstract") or "").strip(),
            "year":     data.get("year"),
        }
    except Exception:
        return {}


# ── Cache I/O ─────────────────────────────────────────────────────────────────

def _cache_dir(runs_dir: Path) -> Path:
    return runs_dir / "benchmark" / "prompt_cache"


def _cache_path(runs_dir: Path, task_name: str, strategy: str) -> Path:
    return _cache_dir(runs_dir) / f"{task_name}__{strategy}.json"


def _combined_refs_path(runs_dir: Path, task_name: str) -> Path:
    return _cache_dir(runs_dir) / f"{task_name}__combined_refs.json"


def load_cached(task_name: str, strategy: str, runs_dir: Path) -> dict | None:
    """Return cached prompt entry or None if not built yet."""
    p = _cache_path(runs_dir, task_name, strategy)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _save(entry: dict, runs_dir: Path) -> None:
    d = _cache_dir(runs_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = _cache_path(runs_dir, entry["task_name"], entry["strategy"])
    p.write_text(json.dumps(entry, indent=2, ensure_ascii=False))


# ── Ref list construction ─────────────────────────────────────────────────────

def _load_combined_refs(runs_dir: Path, task_name: str) -> tuple[list[dict], list[str]] | None:
    """Load cached combined_refs from disk, or None if not cached."""
    p = _combined_refs_path(runs_dir, task_name)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        return d["combined_refs"], d["frontline_ids"]
    except Exception:
        return None


def _save_combined_refs(
    runs_dir: Path, task_name: str, combined_refs: list[dict], frontline_ids: list[str]
) -> None:
    d = _cache_dir(runs_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = _combined_refs_path(runs_dir, task_name)
    p.write_text(json.dumps(
        {"task_name": task_name, "frontline_ids": frontline_ids, "combined_refs": combined_refs,
         "built_at": datetime.now(timezone.utc).isoformat()},
        indent=2, ensure_ascii=False,
    ))


def _fetch_combined_refs(task, arxiv_root: Path) -> tuple[list[dict], list[str]]:
    """
    Fetch combined_refs fresh from S2. Called only when the combined_refs
    cache does not exist.

    combined_refs = [frontline_1, frontline_2, frontline_3] + union(their refs)
    Each entry: {arxiv_id, title, abstract, year}
    """
    from probe import _fetch_refs_s2, _find_metadata_in_store

    frontline_ids: list[str] = task.eval_papers()
    frontline_entries: list[dict] = []
    union_refs: dict[str, dict] = {}  # arxiv_id → ref dict

    for arxiv_id in frontline_ids:
        # Load metadata: local arxiv store first, S2 API as fallback
        meta: dict = {}
        meta_path = _find_metadata_in_store(arxiv_id, arxiv_root)
        if meta_path:
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                pass

        if not meta.get("title"):
            print(f"  [frontline {arxiv_id} not in local store — fetching from S2]")
            meta = _fetch_paper_s2(arxiv_id)

        created = meta.get("created") or ""
        year_str = created[:4]
        year = meta.get("year") or (int(year_str) if year_str.isdigit() else None)

        frontline_entries.append({
            "arxiv_id": arxiv_id,
            "title":    meta.get("title") or f"[{arxiv_id}]",
            "abstract": meta.get("abstract", ""),
            "year":     year,
        })

        refs = _fetch_refs_s2(arxiv_id, arxiv_root=arxiv_root)
        for ref in refs:
            ref_id = ref.get("arxiv_id")
            if ref_id and ref_id not in union_refs:
                union_refs[ref_id] = ref

    # Remove frontline papers from the union (they're listed first)
    for fl in frontline_entries:
        union_refs.pop(fl["arxiv_id"], None)

    combined = frontline_entries + list(union_refs.values())
    return combined, frontline_ids


def _build_combined_refs(
    task, arxiv_root: Path, runs_dir: Path | None = None, force: bool = False
) -> tuple[list[dict], list[str]]:
    """
    Return (combined_refs, frontline_ids), loading from disk cache when available.
    Set force=True to re-fetch from S2 regardless.
    """
    if runs_dir is not None and not force:
        cached = _load_combined_refs(runs_dir, task.name)
        if cached is not None:
            return cached

    combined, frontline_ids = _fetch_combined_refs(task, arxiv_root)

    if runs_dir is not None:
        _save_combined_refs(runs_dir, task.name, combined, frontline_ids)

    return combined, frontline_ids


# ── with_research_question: inject pre-written question into full_refs prompt ──

def _build_with_research_question(
    task_name: str, full_refs_entry: dict, runs_dir: Path
) -> dict:
    """
    Build with_research_question by inserting the pre-written task research
    question into the full_refs prompt. Identical ref list to full_refs.
    """
    question = TASK_RESEARCH_QUESTIONS[task_name]
    full_prompt = full_refs_entry["prompt"]

    insertion = (
        f'The researcher has identified the following open question as their primary motivation:\n'
        f'"{question}"\n\n'
    )

    if _PROPOSAL_MARKER in full_prompt:
        prompt = full_prompt.replace(_PROPOSAL_MARKER, insertion + _PROPOSAL_MARKER, 1)
    else:
        # Fallback: append before end
        prompt = full_prompt.rstrip() + "\n\n" + insertion

    return {
        "task_name":     task_name,
        "strategy":      "with_research_question",
        "system":        full_refs_entry["system"],
        "prompt":        prompt,
        "n_refs":        full_refs_entry["n_refs"],
        "n_frontlines":  full_refs_entry["n_frontlines"],
        "frontline_ids": full_refs_entry["frontline_ids"],
        "built_at":      datetime.now(timezone.utc).isoformat(),
    }


# ── Prompt building ───────────────────────────────────────────────────────────

def build_prompt(task_name: str, strategy: str, runs_dir: Path) -> dict:
    """
    Build the prompt for (task_name, strategy). May make LLM API calls.
    Returns a cache entry dict.
    """
    from benchmark.tasks import get_task
    from train.prompt_builder import get_builder

    task = get_task(task_name)

    # with_research_question for benchmark tasks: reuse full_refs ref block +
    # inject pre-written question. No LLM call, no WithResearchQuestionBuilder.
    if strategy == "with_research_question" and task_name in TASK_RESEARCH_QUESTIONS:
        full_refs_entry = get_or_build(task_name, "full_refs", runs_dir)
        return _build_with_research_question(task_name, full_refs_entry, runs_dir)

    combined_refs, frontline_ids = _build_combined_refs(task, _ARXIV_ROOT, runs_dir)
    n_pinned = len(frontline_ids)

    # Synthetic arxiv_id used as cache key for inner LLM caches
    # (top-k selection indices, related-work narratives).
    synthetic_id = f"task:{task_name}:combined:p{n_pinned}"

    record = {
        "arxiv_id":    synthetic_id,
        "title":       "",
        "abstract":    "",
        "refs":        combined_refs,
        "pinned_count": n_pinned,
    }

    cfg = {
        "prompt_builder": {
            "strategy": strategy,
            "top_k":    5,
        },
        "runs_dir": str(runs_dir),
    }
    builder = get_builder(cfg)

    return {
        "task_name":     task_name,
        "strategy":      strategy,
        "system":        builder.system(),
        "prompt":        builder.build(record),
        "n_refs":        len(combined_refs),
        "n_frontlines":  len(frontline_ids),
        "frontline_ids": frontline_ids,
        "built_at":      datetime.now(timezone.utc).isoformat(),
    }


def get_or_build(task_name: str, strategy: str, runs_dir: Path) -> dict:
    """Return cached entry, or build+cache it (may make LLM calls)."""
    cached = load_cached(task_name, strategy, runs_dir)
    if cached is not None:
        return cached
    entry = build_prompt(task_name, strategy, runs_dir)
    _save(entry, runs_dir)
    return entry


def build_one(
    task_name: str,
    strategy: str,
    runs_dir: Path,
    force: bool = False,
) -> dict | None:
    """Build (and cache) a single strategy. Returns entry or None on error."""
    if not force:
        cached = load_cached(task_name, strategy, runs_dir)
        if cached is not None:
            return cached
    try:
        entry = build_prompt(task_name, strategy, runs_dir)
        _save(entry, runs_dir)
        return entry
    except Exception:
        import traceback
        traceback.print_exc()
        return None


def build_all(
    task_name: str,
    runs_dir: Path,
    force: bool = False,
    on_progress: Callable[[str, dict | None], None] | None = None,
) -> dict[str, dict | None]:
    """
    Build and cache all strategies for task_name.

    on_progress(strategy, entry_or_None) is called after each attempt.
    Returns dict[strategy → entry] (None on error).
    """
    results: dict[str, dict | None] = {}
    for strategy in ALL_STRATEGIES:
        entry = build_one(task_name, strategy, runs_dir, force=force)
        results[strategy] = entry
        if on_progress:
            on_progress(strategy, entry)
    return results
