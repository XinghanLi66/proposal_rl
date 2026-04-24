"""
Conditioning prompt builder strategies.

Each strategy takes a dataset record (which must have a `refs` list) and
returns the user-facing prompt string passed to the model.

LLM-based strategies (top_k_refs, related_work, with_research_question) call
Claude to transform the reference list before building the prompt.  Results are
cached to disk (runs_dir/dataset/prompt_cache/) so each paper is only processed
once.  Falls back gracefully if the API is unavailable.

Add a new strategy by subclassing PromptBuilder, implementing build(),
and adding an entry to REGISTRY.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from abc import ABC, abstractmethod
from pathlib import Path

log = logging.getLogger(__name__)

# Suppress httpx's per-request INFO lines ("HTTP Request: POST … 200 OK") — they
# flood the terminal when building prompts for large datasets.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

_PROXY_URL = "http://10.39.10.241:10001"
_DEFAULT_MODEL = "claude-sonnet-4-6"

# ── Shared templates (must match data/build_dataset.py) ──────────────────────

SYSTEM_PROMPT = """\
You are a research scientist with deep expertise in machine learning and AI. \
You will be given a list of papers (title and abstract each) that a researcher has been reading. \
Your task is to generate a structured research proposal for a novel research direction \
that these papers collectively suggest — as if you were the researcher proposing new work \
inspired by this body of literature."""

_PROPOSAL_FORMAT = """\
Generate a structured research proposal using this exact format:
<thinking>
[Analyze what themes, methods, and open problems span these references. \
Identify the most compelling gap or opportunity. \
Think step-by-step before writing the proposal.]
</thinking>
<proposal>
<problem>What core research problem should be addressed?</problem>
<gap>What gap in the existing literature motivates this work?</gap>
<key_insight>What key insight or hypothesis drives the proposed approach?</key_insight>
<approach>How would the proposed method work at a high level?</approach>
<expected_contributions>What are the expected scientific contributions?</expected_contributions>
</proposal>"""

# Standard numbered-list template (full_refs and top_k_refs)
USER_TEMPLATE = """\
Below are {n_refs} papers from a researcher's reading list. \
Based on these references, propose a novel research direction.

{ref_block}

""" + _PROPOSAL_FORMAT

# Related-work narrative template
RELATED_WORK_TEMPLATE = """\
A researcher has been studying the following area of the literature:

{related_work}

Based on this background, propose a novel research direction.

""" + _PROPOSAL_FORMAT

# Research-question augmented template
WITH_QUESTION_TEMPLATE = """\
Below are {n_refs} papers from a researcher's reading list. \
Based on these references, propose a novel research direction.

{ref_block}

The researcher has identified the following open question as their primary motivation:
"{research_question}"

""" + _PROPOSAL_FORMAT


def _ref_entry(idx: int, ref: dict, abstract_chars: int = 400) -> str:
    abstract = (ref.get("abstract") or "").strip()
    if len(abstract) > abstract_chars:
        abstract = abstract[:abstract_chars] + "..."
    return (
        f"[{idx}] {ref.get('title', 'Unknown')} ({ref.get('year') or 'n.d.'})\n"
        f"Abstract: {abstract or '(no abstract available)'}"
    )


def _compact_ref_list(refs: list[dict]) -> str:
    """One line per ref — used in LLM selection prompts."""
    return "\n".join(
        f"[{i+1}] {r.get('title', 'Unknown')} ({r.get('year') or 'n.d.'})"
        for i, r in enumerate(refs)
    )


# ── Claude helpers ────────────────────────────────────────────────────────────

def _claude_client():
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY") or "123"
    return anthropic.Anthropic(
        api_key=api_key, base_url=_PROXY_URL, timeout=120.0, max_retries=2
    )


def _call_claude(client, model: str, prompt: str, max_tokens: int = 512) -> str:
    """Call Claude via the proxy using raw httpx so we can decode with errors='replace'.

    The proxy at _PROXY_URL sometimes returns non-UTF-8 bytes in the SSE stream;
    the Anthropic SDK's streaming path raises UnicodeDecodeError on every such
    response.  Bypassing the SDK and decoding ourselves avoids that.
    """
    import httpx as _httpx
    import json as _json
    base = str(client.base_url).rstrip("/")
    log.debug("POST %s/v1/messages  model=%s  max_tokens=%d", base, model, max_tokens)
    resp = _httpx.post(
        f"{base}/v1/messages",
        headers={
            "x-api-key": client.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    # Decode with errors='replace' so stray Latin-1 bytes become U+FFFD
    # rather than raising UnicodeDecodeError.
    data = _json.loads(resp.content.decode("utf-8", errors="replace"))
    return data["content"][0]["text"].strip()


# ── File-backed prompt cache ──────────────────────────────────────────────────

class _Cache:
    """
    Simple JSONL file cache keyed by arxiv_id.
    Entries are appended on write and loaded fully on init.
    Thread-unsafe — fine for single-process data loading.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict[str, str] = {}
        if path.exists():
            with open(path) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        self._data[d["key"]] = d["value"]
                    except Exception:
                        pass
            log.info(f"[Cache] {len(self._data)} entries loaded from {path.name}")

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        if key in self._data:
            return
        self._data[key] = value
        self._path.parent.mkdir(parents=True, exist_ok=True)
        import fcntl
        with open(self._path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(json.dumps({"key": key, "value": value}) + "\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


# ── Base class ────────────────────────────────────────────────────────────────

class PromptBuilder(ABC):
    def __init__(self, cfg: dict) -> None:
        self.max_refs = cfg.get("max_refs", 40)

    @abstractmethod
    def build(self, record: dict) -> str:
        """Return the user-turn prompt string for this record."""

    def system(self) -> str:
        return SYSTEM_PROMPT


# ── FullRefsBuilder ───────────────────────────────────────────────────────────

class FullRefsBuilder(PromptBuilder):
    """All resolved references, shuffled and truncated to max_refs. (default)"""

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.seed = cfg.get("shuffle_seed", 42)

    def build(self, record: dict) -> str:
        refs = list(record.get("refs", []))
        random.Random(self.seed).shuffle(refs)
        refs = refs[: self.max_refs]
        ref_block = "\n\n".join(_ref_entry(i + 1, r) for i, r in enumerate(refs))
        return USER_TEMPLATE.format(n_refs=len(refs), ref_block=ref_block)


# ── TopKRefsBuilder ───────────────────────────────────────────────────────────

class TopKRefsBuilder(PromptBuilder):
    """
    LLM selects the K most important references from the full list,
    then builds the standard numbered-list prompt with only those K refs.

    Falls back to the first K refs if the API call fails.
    """

    _SELECT_PROMPT = """\
You are given a list of research papers. Select the {k} most central and \
important references for inspiring a novel research direction. Prioritise \
methodological breadth, impact, and recency.

{compact_list}

Reply with ONLY a comma-separated list of exactly {k} 1-based indices, e.g.: 3,7,12"""

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.k = cfg.get("top_k", 5)
        self.model = cfg.get("model", _DEFAULT_MODEL)
        runs_dir = cfg.get("runs_dir", ".")
        cache_file = Path(runs_dir) / "dataset" / "prompt_cache" / f"top_k_{self.k}.jsonl"
        self._cache = _Cache(cache_file)
        self._client = None

    def _client_(self):
        if self._client is None:
            self._client = _claude_client()
        return self._client

    def build(self, record: dict) -> str:
        refs = list(record.get("refs", []))
        if not refs:
            return USER_TEMPLATE.format(n_refs=0, ref_block="(no references)")

        arxiv_id = record.get("arxiv_id", "")
        cached = self._cache.get(arxiv_id) if arxiv_id else None

        if cached is not None:
            indices = json.loads(cached)
        else:
            indices = self._select_indices(refs, arxiv_id)

        selected = [refs[i] for i in indices if i < len(refs)]
        ref_block = "\n\n".join(_ref_entry(i + 1, r) for i, r in enumerate(selected))
        return USER_TEMPLATE.format(n_refs=len(selected), ref_block=ref_block)

    def _select_indices(self, refs: list[dict], arxiv_id: str) -> list[int]:
        k = min(self.k, len(refs))
        truncated = refs[: self.max_refs]
        prompt = self._SELECT_PROMPT.format(
            k=k, compact_list=_compact_ref_list(truncated)
        )
        try:
            text = _call_claude(self._client_(), self.model, prompt, max_tokens=64)
            raw = [int(x) for x in re.findall(r"\d+", text)]
            indices = [i - 1 for i in raw if 1 <= i <= len(truncated)][:k]
            if not indices:
                raise ValueError(f"no valid indices in response: {text!r}")
            if arxiv_id:  # only cache on success
                self._cache.set(arxiv_id, json.dumps(indices))
        except Exception as exc:
            log.warning(f"TopKRefsBuilder: LLM call failed ({exc}), using first {k} refs")
            indices = list(range(k))
        return indices


# ── RelatedWorkBuilder ────────────────────────────────────────────────────────

class RelatedWorkBuilder(PromptBuilder):
    """
    LLM synthesizes a related-work narrative from the full reference list,
    then conditions the model on that narrative instead of a numbered list.

    The narrative is cached so the LLM is called only once per paper.
    """

    _SYNTHESIS_PROMPT = """\
Synthesize a concise related work section (3-5 paragraphs) covering the following \
{n} research papers. Focus on thematic connections, methodological trends, and open \
problems across the collection — do NOT summarise each paper individually.

{ref_block}

Write the related work section:"""

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.model = cfg.get("model", _DEFAULT_MODEL)
        runs_dir = cfg.get("runs_dir", ".")
        self._cache = _Cache(
            Path(runs_dir) / "dataset" / "prompt_cache" / "related_work.jsonl"
        )
        self._client = None

    def _client_(self):
        if self._client is None:
            self._client = _claude_client()
        return self._client

    def build(self, record: dict) -> str:
        refs = list(record.get("refs", []))
        if not refs:
            return USER_TEMPLATE.format(n_refs=0, ref_block="(no references)")

        arxiv_id = record.get("arxiv_id", "")
        related_work = self._cache.get(arxiv_id) if arxiv_id else None

        if related_work is None:
            related_work = self._synthesize(refs, arxiv_id)

        return RELATED_WORK_TEMPLATE.format(related_work=related_work)

    def _synthesize(self, refs: list[dict], arxiv_id: str) -> str:
        truncated = refs[: self.max_refs]
        ref_block = "\n\n".join(_ref_entry(i + 1, r) for i, r in enumerate(truncated))
        prompt = self._SYNTHESIS_PROMPT.format(n=len(truncated), ref_block=ref_block)
        log.info("related_work: synthesizing %s (%d refs)", arxiv_id or "?", len(truncated))
        try:
            text = _call_claude(self._client_(), self.model, prompt, max_tokens=1024)
            if arxiv_id:  # only cache on success
                self._cache.set(arxiv_id, text)
            return text
        except Exception as exc:
            log.warning("RelatedWorkBuilder: LLM call failed (%s), falling back to numbered ref list", exc)
            # Fallback: numbered list with abstracts — coherent and keeps the template intact
            return ref_block


# ── WithResearchQuestionBuilder ───────────────────────────────────────────────

class WithResearchQuestionBuilder(PromptBuilder):
    """
    LLM generates a focused open research question from the reference list,
    then augments the standard numbered-list prompt with that question.

    The question is cached so the LLM is called only once per paper.
    """

    _QUESTION_PROMPT = """\
You are given a list of {n} research papers. Identify the single most compelling \
open research question or direction that naturally emerges from the collective gaps \
and limitations of this body of work.

{compact_list}

State the research question in 1-3 sentences. Be specific and forward-looking:"""

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.model = cfg.get("model", _DEFAULT_MODEL)
        runs_dir = cfg.get("runs_dir", ".")
        self._cache = _Cache(
            Path(runs_dir) / "dataset" / "prompt_cache" / "research_question.jsonl"
        )
        self._client = None

    def _client_(self):
        if self._client is None:
            self._client = _claude_client()
        return self._client

    def build(self, record: dict) -> str:
        refs = list(record.get("refs", []))
        if not refs:
            return USER_TEMPLATE.format(n_refs=0, ref_block="(no references)")

        arxiv_id = record.get("arxiv_id", "")
        research_question = self._cache.get(arxiv_id) if arxiv_id else None

        if research_question is None:
            research_question = self._generate_question(refs, arxiv_id)

        # Build the standard ref block alongside the question
        random.Random(42).shuffle(refs)
        truncated = refs[: self.max_refs]
        ref_block = "\n\n".join(_ref_entry(i + 1, r) for i, r in enumerate(truncated))
        return WITH_QUESTION_TEMPLATE.format(
            n_refs=len(truncated),
            ref_block=ref_block,
            research_question=research_question,
        )

    def _generate_question(self, refs: list[dict], arxiv_id: str) -> str:
        truncated = refs[: self.max_refs]
        prompt = self._QUESTION_PROMPT.format(
            n=len(truncated), compact_list=_compact_ref_list(truncated)
        )
        try:
            text = _call_claude(self._client_(), self.model, prompt, max_tokens=128)
            if arxiv_id:  # only cache on success
                self._cache.set(arxiv_id, text)
            return text
        except Exception as exc:
            log.warning(f"WithResearchQuestionBuilder: LLM call failed ({exc}), omitting question")
            return "What novel research direction is most motivated by this body of work?"


# ── TopKRelatedWorkBuilder ────────────────────────────────────────────────────

class TopKRelatedWorkBuilder(PromptBuilder):
    """
    Two-stage LLM builder:
      1. LLM selects the top-K most important references (like TopKRefsBuilder).
      2. LLM synthesizes a related-work narrative from just those K refs (like RelatedWorkBuilder).

    Caches both the index selection and the resulting narrative, keyed by arxiv_id.
    """

    _SELECT_PROMPT = TopKRefsBuilder._SELECT_PROMPT

    _SYNTHESIS_PROMPT = RelatedWorkBuilder._SYNTHESIS_PROMPT

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.k = cfg.get("top_k", 5)
        self.model = cfg.get("model", _DEFAULT_MODEL)
        runs_dir = cfg.get("runs_dir", ".")
        cache_dir = Path(runs_dir) / "dataset" / "prompt_cache"
        self._idx_cache = _Cache(cache_dir / f"top_k_{self.k}.jsonl")
        self._rw_cache  = _Cache(cache_dir / "top_k_related_work.jsonl")
        self._client = None

    def _client_(self):
        if self._client is None:
            self._client = _claude_client()
        return self._client

    def build(self, record: dict) -> str:
        refs = list(record.get("refs", []))
        if not refs:
            return USER_TEMPLATE.format(n_refs=0, ref_block="(no references)")

        arxiv_id = record.get("arxiv_id", "")

        # Stage 1: top-K index selection
        cached_idx = self._idx_cache.get(arxiv_id) if arxiv_id else None
        if cached_idx is not None:
            indices = json.loads(cached_idx)
        else:
            indices = self._select_indices(refs, arxiv_id)
        selected = [refs[i] for i in indices if i < len(refs)]

        # Stage 2: narrative synthesis from selected refs
        cache_key = arxiv_id + f"_topk{self.k}" if arxiv_id else ""
        related_work = self._rw_cache.get(cache_key) if cache_key else None
        if related_work is None:
            related_work = self._synthesize(selected, cache_key)

        return RELATED_WORK_TEMPLATE.format(related_work=related_work)

    def _select_indices(self, refs: list[dict], arxiv_id: str) -> list[int]:
        k = min(self.k, len(refs))
        truncated = refs[: self.max_refs]
        prompt = self._SELECT_PROMPT.format(k=k, compact_list=_compact_ref_list(truncated))
        try:
            text = _call_claude(self._client_(), self.model, prompt, max_tokens=64)
            raw = [int(x) for x in re.findall(r"\d+", text)]
            indices = [i - 1 for i in raw if 1 <= i <= len(truncated)][:k]
            if not indices:
                raise ValueError(f"no valid indices in response: {text!r}")
            if arxiv_id:  # only cache on success
                self._idx_cache.set(arxiv_id, json.dumps(indices))
        except Exception as exc:
            log.warning(f"TopKRelatedWorkBuilder: selection failed ({exc}), using first {k} refs")
            indices = list(range(k))
        return indices

    def _synthesize(self, refs: list[dict], cache_key: str) -> str:
        ref_block = "\n\n".join(_ref_entry(i + 1, r) for i, r in enumerate(refs))
        prompt = self._SYNTHESIS_PROMPT.format(n=len(refs), ref_block=ref_block)
        log.info("top_k_related_work: synthesizing %s (%d refs)", cache_key or "?", len(refs))
        try:
            text = _call_claude(self._client_(), self.model, prompt, max_tokens=1024)
            if cache_key:  # only cache on success
                self._rw_cache.set(cache_key, text)
            return text
        except Exception as exc:
            log.warning("TopKRelatedWorkBuilder: synthesis failed (%s), falling back to numbered ref list", exc)
            return ref_block


# ── Registry and factory ──────────────────────────────────────────────────────

REGISTRY: dict[str, type[PromptBuilder]] = {
    "full_refs":              FullRefsBuilder,
    "top_k_refs":             TopKRefsBuilder,
    "related_work":           RelatedWorkBuilder,
    "with_research_question": WithResearchQuestionBuilder,
    "top_k_related_work":     TopKRelatedWorkBuilder,
}


def get_builder(cfg: dict) -> PromptBuilder:
    pb_cfg = cfg.get("prompt_builder", {}).copy()
    # Pass runs_dir so LLM-based builders can locate their cache files
    pb_cfg.setdefault("runs_dir", cfg.get("runs_dir", "."))
    strategy = pb_cfg.get("strategy", "full_refs")
    cls = REGISTRY.get(strategy)
    if cls is None:
        raise ValueError(
            f"Unknown prompt_builder strategy: {strategy!r}. "
            f"Available: {list(REGISTRY)}"
        )
    return cls(pb_cfg)
