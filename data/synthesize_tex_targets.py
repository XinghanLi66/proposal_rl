#!/usr/bin/env python3
"""
Synthesize implementation-ready proposal targets from full TeX sources.

This is the TeX-grounded successor to data/synthesize_cot.py. It keeps the
original abstract for comparison, but the new training target is synthesized
from extracted full-paper TeX evidence so it can describe concrete methods,
training/data recipes, and evaluation plans instead of only abstract-level
intent.

Default behavior is conservative: records without usable TeX are skipped from
the main output and written to a *.skipped.jsonl manifest. There is no abstract
fallback for TeX targets.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable

import anthropic
import yaml

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data.synthesize_cot import _extract_ref_text, extract_proposal_text, simple_cosine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_PROXY_URL = "http://10.39.10.241:10001"


TEX_SYNTHESIS_SYSTEM = """\
You are creating high-quality supervised training targets for a research-proposal model.
You will receive a paper's reference-conditioned prompt, original abstract, and selected
evidence extracted from the paper's full TeX source.

Your job is to write an implementation-ready prospective research proposal: it should
sound like a researcher planning the work before writing the paper, but it may use the
concrete method, training, implementation, and evaluation details found in the TeX evidence.

Rules:
1. Do not merely paraphrase the abstract.
2. Ground the proposal in the TeX evidence, especially method, system, experiment,
   ablation, and result sections.
3. Prefer actionable details: data, models, algorithms, losses, baselines, metrics,
   implementation steps, and failure modes when available.
4. Do not invent numeric results that are not supported by the evidence.
5. Use the exact XML structure requested by the user prompt."""


TEX_SYNTHESIS_USER = """\
=== PAPER METADATA ===
arxiv_id: {arxiv_id}
title: {title}

=== ORIGINAL ABSTRACT (old target for comparison; do not merely paraphrase) ===
{abstract}

=== REFERENCE-CONDITIONED PROMPT ===
{ref_block}

=== FULL-TEX EVIDENCE EXCERPTS ===
{tex_excerpt}

=== TASK ===
Write a prospective, implementation-ready research target for training. It should
explain what a model should propose if it had inferred this paper from the references.
Use this exact format:

<thinking>
[Briefly identify the concrete problem, method ingredients, implementation recipe,
evaluation design, and risks that are supported by the TeX evidence.]
</thinking>
<proposal>
<problem>Concrete research problem, not just a broad topic.</problem>
<gap>Specific limitation in prior work or current practice.</gap>
<core_idea>Main technical idea or hypothesis.</core_idea>
<implementation_plan>Actionable implementation steps, including models, algorithms, training/inference pipeline, and data flow where available.</implementation_plan>
<algorithm_or_system>Important algorithmic/system components, losses, constraints, or design choices.</algorithm_or_system>
<training_or_data_recipe>Datasets, preprocessing, supervision, simulation, optimization, or experimental setup needed to reproduce the idea.</training_or_data_recipe>
<evaluation_plan>Baselines, metrics, ablations, and stress tests.</evaluation_plan>
<expected_results>Expected qualitative or quantitative outcomes supported by the evidence.</expected_results>
<risks_and_limitations>Likely limitations, assumptions, and failure modes.</risks_and_limitations>
</proposal>"""


SECTION_RE = re.compile(
    r"\\(?P<level>section|subsection|subsubsection|paragraph)\*?"
    r"(?:\[[^\]]*\])?\{(?P<title>[^{}]{1,220})\}",
    re.DOTALL,
)
PSEUDO_HEADING_RE = re.compile(
    r"(?m)^\s*(?:\([a-z]\)\s*)?(?:\\noindent\s*)?"
    r"\\textit\{(?P<title>[A-Z][^{}]{2,100}\.?)\}\s*(?:\\\\)?"
)
COMMENT_RE = re.compile(r"(?<!\\)%.*")
WHITESPACE_RE = re.compile(r"\s+")

DROP_ENVS = (
    "figure",
    "figure*",
    "table",
    "table*",
    "comment",
    "thebibliography",
)

CONTROL_COMMANDS = (
    "cite",
    "citet",
    "citep",
    "citealp",
    "citeauthor",
    "citeyear",
    "ref",
    "eqref",
    "cref",
    "Cref",
    "autoref",
    "label",
    "url",
    "email",
)

STYLE_COMMANDS = (
    "textit",
    "textbf",
    "emph",
    "underline",
    "textsc",
    "textrm",
    "mathrm",
    "mathbf",
    "mathit",
    "mbox",
)


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_records(input_file: Path, limit: int | None = None, arxiv_id: str | None = None) -> list[dict]:
    records: list[dict] = []
    with open(input_file) as f:
        for line in f:
            try:
                record = json.loads(line)
            except Exception:
                continue
            if arxiv_id and record.get("arxiv_id") != arxiv_id:
                continue
            records.append(record)
            if limit and len(records) >= limit:
                break
    return records


def load_done(output_file: Path) -> set[str]:
    done: set[str] = set()
    if not output_file.exists():
        return done
    with open(output_file) as f:
        for line in f:
            try:
                aid = json.loads(line).get("arxiv_id")
                if aid:
                    done.add(aid)
            except Exception:
                pass
    return done


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def load_synthesis_cache(paths: Iterable[Path]) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    for path in paths:
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                aid = row.get("arxiv_id")
                if aid and (row.get("target_impl_proposal") or row.get("tex_status")):
                    existing = cache.get(aid)
                    if row.get("target_impl_proposal") or not (existing and existing.get("target_impl_proposal")):
                        cache[aid] = row
    return cache


def _safe_arxiv_dir_name(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "__")


def _canonical_candidates(arxiv_id: str, arxiv_root: Path) -> list[Path]:
    candidates: list[Path] = []
    safe = _safe_arxiv_dir_name(arxiv_id)
    m = re.match(r"^(\d{2})(\d{2})\.", arxiv_id)
    if m:
        year = "20" + m.group(1)
        month = m.group(2)
        candidates.append(arxiv_root / year / month / safe)
        if safe != arxiv_id:
            candidates.append(arxiv_root / year / month / arxiv_id)
    return candidates


def find_paper_dir(arxiv_id: str, arxiv_root: Path) -> Path | None:
    for candidate in _canonical_candidates(arxiv_id, arxiv_root):
        if candidate.exists():
            return candidate

    safe = _safe_arxiv_dir_name(arxiv_id)
    for year_dir in sorted(arxiv_root.iterdir() if arxiv_root.exists() else []):
        if not year_dir.is_dir():
            continue
        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            for name in (safe, arxiv_id):
                candidate = month_dir / name
                if candidate.exists():
                    return candidate
    return None


def _tex_files(tex_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(tex_dir.rglob("*.tex")):
        if not path.is_file():
            continue
        stem = path.stem.lower()
        if stem in {"refs", "references", "bibliography", "biblio"}:
            continue
        files.append(path)
    return files


def _strip_comments(text: str) -> str:
    return "\n".join(COMMENT_RE.sub("", line) for line in text.splitlines())


def _drop_env(text: str, env: str) -> str:
    pattern = re.compile(
        rf"\\begin\{{{re.escape(env)}\}}.*?\\end\{{{re.escape(env)}\}}",
        re.DOTALL,
    )
    return pattern.sub(" ", text)


def _prepare_tex(raw: str) -> str:
    text = _strip_comments(raw)
    for env in DROP_ENVS:
        text = _drop_env(text, env)
    text = re.sub(r"\\begin\{abstract\}", r"\\section*{Abstract}", text)
    text = re.sub(r"\\end\{abstract\}", "\n", text)
    text = PSEUDO_HEADING_RE.sub(lambda m: "\n\\section*{" + m.group("title").rstrip(".") + "}\n", text)
    return text


def clean_latex_text(text: str) -> str:
    text = _strip_comments(text)
    text = re.sub(r"\\href\{[^{}]*\}\{([^{}]*)\}", r"\1", text)
    for cmd in CONTROL_COMMANDS:
        text = re.sub(rf"\\{cmd}\*?(?:\[[^\]]*\])?\{{[^{{}}]*\}}", " ", text)
    for cmd in STYLE_COMMANDS:
        for _ in range(4):
            text = re.sub(rf"\\{cmd}\*?(?:\[[^\]]*\])?\{{([^{{}}]*)\}}", r"\1", text)
    for _ in range(4):
        text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", text)
    text = re.sub(r"\\.", " ", text)
    text = text.replace("~", " ")
    text = text.replace("$", " ")
    text = text.replace("{", " ").replace("}", " ")
    text = text.replace("&", " and ")
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def clean_heading(title: str) -> str:
    cleaned = clean_latex_text(title).strip(" .:-")
    return cleaned or "Untitled"


def classify_section(heading: str) -> str:
    h = heading.lower()
    if "abstract" in h:
        return "abstract"
    if "data availability" in h or "acknowledg" in h:
        return "other"
    if any(k in h for k in ("intro", "background", "motivation", "overview")):
        return "problem"
    if any(k in h for k in ("method", "approach", "algorithm", "model", "framework", "system", "calculation", "simulation", "protocol")):
        return "method"
    if any(k in h for k in ("implementation", "training", "dataset", "data", "loss", "objective")):
        return "implementation"
    if any(k in h for k in ("experiment", "evaluation", "setup", "baseline", "benchmark", "ablation")):
        return "evaluation"
    if any(k in h for k in ("result", "analysis", "finding")):
        return "results"
    if any(k in h for k in ("discussion", "limitation", "future", "conclusion")):
        return "discussion"
    if "appendix" in h or "supplement" in h:
        return "appendix"
    return "other"


def extract_tex_sections_from_text(raw: str, source: str) -> list[dict]:
    prepared = _prepare_tex(raw)
    markers = list(SECTION_RE.finditer(prepared))
    if not markers:
        cleaned = clean_latex_text(prepared)
        if len(cleaned) < 200:
            return []
        return [{
            "heading": Path(source).name,
            "kind": "other",
            "text": cleaned,
            "source": source,
            "order": 0,
        }]

    sections: list[dict] = []
    for idx, marker in enumerate(markers):
        start = marker.end()
        end = markers[idx + 1].start() if idx + 1 < len(markers) else len(prepared)
        heading = clean_heading(marker.group("title"))
        text = clean_latex_text(prepared[start:end])
        if len(text) < 120:
            continue
        sections.append({
            "heading": heading,
            "kind": classify_section(heading),
            "text": text,
            "source": source,
            "order": idx,
        })
    return sections


def extract_tex_sections(tex_files: list[Path], paper_dir: Path) -> list[dict]:
    sections: list[dict] = []
    for path in tex_files:
        try:
            raw = path.read_text(errors="replace")
        except Exception:
            continue
        rel = str(path.relative_to(paper_dir))
        sections.extend(extract_tex_sections_from_text(raw, rel))
    for i, section in enumerate(sections):
        section["global_order"] = i
    return sections


KIND_PRIORITY = {
    "abstract": 0,
    "problem": 1,
    "method": 2,
    "implementation": 2,
    "evaluation": 3,
    "results": 3,
    "discussion": 4,
    "appendix": 5,
    "other": 9,
}


def select_tex_sections(sections: list[dict], max_chars: int) -> list[dict]:
    ranked = sorted(
        sections,
        key=lambda s: (KIND_PRIORITY.get(s.get("kind", "other"), 9), s.get("global_order", 0)),
    )
    selected: list[dict] = []
    used = 0
    per_section_cap = max(2500, min(6500, max_chars // 4))
    for section in ranked:
        if used >= max_chars:
            break
        text = section.get("text", "")
        if not text:
            continue
        budget = min(per_section_cap, max_chars - used)
        clipped = text[:budget].strip()
        if len(clipped) < 120:
            continue
        selected.append({
            "heading": section.get("heading", ""),
            "kind": section.get("kind", "other"),
            "source": section.get("source", ""),
            "text": clipped,
        })
        used += len(clipped)
    return selected


def format_tex_excerpt(sections: list[dict]) -> str:
    chunks = []
    for section in sections:
        chunks.append(
            f"### {section.get('heading', 'Untitled')} [{section.get('kind', 'other')}]\n"
            f"source: {section.get('source', '')}\n"
            f"{section.get('text', '')}"
        )
    return "\n\n".join(chunks)


def extract_tex_context(arxiv_id: str, arxiv_root: Path, max_tex_chars: int = 24000) -> dict:
    paper_dir = find_paper_dir(arxiv_id, arxiv_root)
    if paper_dir is None:
        return {
            "tex_status": "missing_paper_dir",
            "paper_dir": None,
            "tex_dir": None,
            "tex_sections": [],
            "tex_excerpt": "",
        }

    tex_dir = paper_dir / "tex"
    if not tex_dir.exists():
        return {
            "tex_status": "missing_tex_dir",
            "paper_dir": str(paper_dir),
            "tex_dir": str(tex_dir),
            "tex_sections": [],
            "tex_excerpt": "",
        }

    files = _tex_files(tex_dir)
    if not files:
        return {
            "tex_status": "missing_tex_files",
            "paper_dir": str(paper_dir),
            "tex_dir": str(tex_dir),
            "tex_sections": [],
            "tex_excerpt": "",
        }

    sections = extract_tex_sections(files, paper_dir)
    selected = select_tex_sections(sections, max_tex_chars)
    if not selected:
        return {
            "tex_status": "empty_tex_after_parse",
            "paper_dir": str(paper_dir),
            "tex_dir": str(tex_dir),
            "tex_sections": [],
            "tex_excerpt": "",
            "tex_file_count": len(files),
        }

    return {
        "tex_status": "ok",
        "paper_dir": str(paper_dir),
        "tex_dir": str(tex_dir),
        "tex_file_count": len(files),
        "tex_section_count": len(sections),
        "tex_sections": selected,
        "tex_excerpt": format_tex_excerpt(selected),
    }


def score_target_quality(text: str) -> dict:
    proposal = extract_proposal_text(text)
    lower = proposal.lower()
    words = re.findall(r"\b\w+\b", proposal)
    required = (
        "problem",
        "gap",
        "core_idea",
        "implementation_plan",
        "algorithm_or_system",
        "training_or_data_recipe",
        "evaluation_plan",
        "expected_results",
        "risks_and_limitations",
    )
    return {
        "word_count": len(words),
        "has_all_required_tags": all(f"<{tag}>" in text and f"</{tag}>" in text for tag in required),
        "has_implementation_terms": any(
            k in lower for k in ("implement", "train", "dataset", "algorithm", "baseline", "metric")
        ),
        "has_evaluation_terms": any(k in lower for k in ("evaluate", "baseline", "ablation", "metric", "benchmark")),
    }


_REACHABLE_PROXY_CACHE: dict[str, bool] = {}


def _probe_url(url: str, timeout: float = 1.0) -> bool:
    """Return True if url TCP-connects within timeout seconds (cached per process)."""
    import socket
    import urllib.parse
    if url in _REACHABLE_PROXY_CACHE:
        return _REACHABLE_PROXY_CACHE[url]
    try:
        p = urllib.parse.urlparse(url)
        host = p.hostname or ""
        port = p.port or 80
        with socket.create_connection((host, port), timeout=timeout):
            result = True
    except Exception:
        result = False
    _REACHABLE_PROXY_CACHE[url] = result
    return result


def _proxy_credentials() -> tuple[str, str]:
    api_key = (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        or "123"
    )
    env_url = os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("ANTHROPIC_PROXY_URL")
    if env_url:
        env_url = env_url.rstrip("/")
        # If env URL is not the default proxy, verify reachability before trusting it.
        # This prevents a dead env proxy (e.g. 10.39.17.12) from blocking the healthy default.
        if env_url != _PROXY_URL and not _probe_url(env_url):
            log.warning("env proxy %s unreachable; using default %s", env_url, _PROXY_URL)
            env_url = _PROXY_URL
        base_url = env_url
    else:
        base_url = _PROXY_URL
    return api_key, base_url


def _proxy_url_candidates(primary: str) -> list[str]:
    candidates: list[str] = []
    for url in (primary, _PROXY_URL):
        url = url.rstrip("/")
        if url and url not in candidates:
            candidates.append(url)
    return candidates


def _claude_client(timeout: float = 300.0, max_retries: int = 2) -> anthropic.AsyncAnthropic:
    api_key, base_url = _proxy_credentials()
    return anthropic.AsyncAnthropic(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)


async def _call_streaming(
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    system: str,
    messages: list[dict],
    connect_timeout: float = 10.0,
    chunk_timeout: float = 60.0,
    max_retries: int = 2,
) -> str:
    """POST to the Bedrock-wrapped proxy with stream=true, decode EventStream lines.

    The proxy wraps AWS Bedrock's application/vnd.amazon.eventstream format:
    each response line is JSON with chunk.bytes = base64(Anthropic SSE event).
    Streaming keeps the per-chunk read timeout from firing on long generations.
    """
    import base64
    import httpx

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Accept-Encoding": "identity",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": messages,
        "stream": True,
    }
    timeout = httpx.Timeout(connect=connect_timeout, read=chunk_timeout, write=10.0, pool=5.0)
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        if attempt > 0:
            await asyncio.sleep(3.0 * attempt)
        try:
            text_chunks: list[str] = []
            async with httpx.AsyncClient(timeout=timeout) as http:
                async with http.stream("POST", f"{base_url}/v1/messages", headers=headers, json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        b64 = obj.get("chunk", {}).get("bytes", "")
                        if not b64:
                            continue
                        try:
                            evt = json.loads(base64.b64decode(b64))
                        except Exception:
                            continue
                        if evt.get("type") == "content_block_delta":
                            delta = evt.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text_chunks.append(delta.get("text", ""))
            return "".join(text_chunks)
        except (httpx.ConnectTimeout, httpx.ConnectError, httpx.ReadTimeout) as exc:
            last_exc = exc
            log.warning("proxy transport error (attempt %d/%d): %r", attempt + 1, max_retries + 1, exc)
            continue
        except Exception:
            raise
    raise RuntimeError(
        f"streaming proxy request failed after {max_retries + 1} attempts: {last_exc!r}"
    )


def _call_streaming_sync(
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    system: str,
    messages: list[dict],
    connect_timeout: float = 10.0,
    chunk_timeout: float = 60.0,
    max_retries: int = 2,
) -> str:
    """Synchronous version of _call_streaming — use in plain threads (no asyncio needed)."""
    import base64
    import httpx

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Accept-Encoding": "identity",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": messages,
        "stream": True,
    }
    timeout = httpx.Timeout(connect=connect_timeout, read=chunk_timeout, write=10.0, pool=5.0)
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        if attempt > 0:
            time.sleep(3.0 * attempt)
        try:
            text_chunks: list[str] = []
            with httpx.Client(timeout=timeout) as http:
                with http.stream("POST", f"{base_url}/v1/messages", headers=headers, json=payload) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        b64 = obj.get("chunk", {}).get("bytes", "")
                        if not b64:
                            continue
                        try:
                            evt = json.loads(base64.b64decode(b64))
                        except Exception:
                            continue
                        if evt.get("type") == "content_block_delta":
                            delta = evt.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text_chunks.append(delta.get("text", ""))
            return "".join(text_chunks)
        except (httpx.ConnectTimeout, httpx.ConnectError, httpx.ReadTimeout) as exc:
            last_exc = exc
            log.warning("proxy transport error (attempt %d/%d): %r", attempt + 1, max_retries + 1, exc)
            continue
        except Exception:
            raise
    raise RuntimeError(
        f"streaming proxy request failed after {max_retries + 1} attempts: {last_exc!r}"
    )


def make_skip_record(record: dict, context: dict) -> dict:
    return {
        "arxiv_id": record.get("arxiv_id"),
        "title": record.get("title"),
        "target_source": "tex_implementation",
        "tex_status": context.get("tex_status", "unknown"),
        "skip_reason": context.get("tex_status", "unknown"),
        "paper_dir": context.get("paper_dir"),
        "tex_dir": context.get("tex_dir"),
    }


def build_result_record(record: dict, context: dict, text: str, model: str, temperature: float) -> dict:
    proposal_text = extract_proposal_text(text)
    leakage = simple_cosine(proposal_text, record.get("abstract", ""))
    return {
        **record,
        "target_source": "tex_implementation",
        "tex_status": context.get("tex_status", "ok"),
        "paper_dir": context.get("paper_dir"),
        "tex_dir": context.get("tex_dir"),
        "tex_file_count": context.get("tex_file_count", 0),
        "tex_section_count": context.get("tex_section_count", 0),
        "tex_sections": context.get("tex_sections", []),
        "cot_impl_proposal": text,
        "target_impl_proposal": proposal_text,
        "target_quality": score_target_quality(text),
        "tex_impl_leakage_score": round(leakage, 4),
        "synthesis_model": model,
        "synthesis_temperature": temperature,
        "synthesized_at": int(time.time()),
    }


async def synthesize_tex_target_async(
    record: dict,
    cfg: dict,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.3,
    max_tex_chars: int = 24000,
    client: anthropic.AsyncAnthropic | None = None,  # kept for API compat; unused
    connect_timeout: float = 10.0,
    chunk_timeout: float = 60.0,
) -> dict:
    arxiv_root = Path(cfg["arxiv_root"])
    context = extract_tex_context(record.get("arxiv_id", ""), arxiv_root, max_tex_chars=max_tex_chars)
    if context.get("tex_status") != "ok":
        return make_skip_record(record, context)

    cot_cfg = cfg.get("cot", {})
    model = model or cot_cfg.get("model", "claude-sonnet-4-6")
    max_tokens = max_tokens or cot_cfg.get("max_tokens", 8192)
    api_key, base_url = _proxy_credentials()

    ref_text = _extract_ref_text(record.get("prompt", ""))
    messages = [{
        "role": "user",
        "content": TEX_SYNTHESIS_USER.format(
            arxiv_id=record.get("arxiv_id", ""),
            title=record.get("title", ""),
            abstract=record.get("abstract", ""),
            ref_block=ref_text,
            tex_excerpt=context["tex_excerpt"],
        ),
    }]
    last_exc: Exception | None = None
    text = ""
    for candidate_url in _proxy_url_candidates(base_url):
        try:
            text = await _call_streaming(
                base_url=candidate_url,
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=TEX_SYNTHESIS_SYSTEM,
                messages=messages,
                connect_timeout=connect_timeout,
                chunk_timeout=chunk_timeout,
                max_retries=0 if candidate_url != _PROXY_URL else 2,
            )
            break
        except RuntimeError as exc:
            last_exc = exc
            if candidate_url == _PROXY_URL:
                raise
            log.warning("streaming proxy %s failed; falling back to %s", candidate_url, _PROXY_URL)
    else:
        raise RuntimeError(f"streaming proxy request failed: {last_exc!r}")
    return build_result_record(record, context, text, model, temperature)


def synthesize_tex_target(
    record: dict,
    cfg: dict,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.3,
    max_tex_chars: int = 24000,
    connect_timeout: float = 10.0,
    chunk_timeout: float = 60.0,
) -> dict:
    """Synchronous streaming synthesis via the proxy; safe to call from a TUI worker thread."""
    arxiv_root = Path(cfg["arxiv_root"])
    context = extract_tex_context(record.get("arxiv_id", ""), arxiv_root, max_tex_chars=max_tex_chars)
    if context.get("tex_status") != "ok":
        return make_skip_record(record, context)

    cot_cfg = cfg.get("cot", {})
    resolved_model = model or cot_cfg.get("model", "claude-sonnet-4-6")
    resolved_max_tokens = max_tokens or cot_cfg.get("max_tokens", 8192)
    api_key, base_url = _proxy_credentials()

    ref_text = _extract_ref_text(record.get("prompt", ""))
    messages = [{
        "role": "user",
        "content": TEX_SYNTHESIS_USER.format(
            arxiv_id=record.get("arxiv_id", ""),
            title=record.get("title", ""),
            abstract=record.get("abstract", ""),
            ref_block=ref_text,
            tex_excerpt=context["tex_excerpt"],
        ),
    }]
    last_exc: Exception | None = None
    text = ""
    for candidate_url in _proxy_url_candidates(base_url):
        try:
            text = _call_streaming_sync(
                base_url=candidate_url,
                api_key=api_key,
                model=resolved_model,
                max_tokens=resolved_max_tokens,
                temperature=temperature,
                system=TEX_SYNTHESIS_SYSTEM,
                messages=messages,
                connect_timeout=connect_timeout,
                chunk_timeout=chunk_timeout,
                max_retries=0 if candidate_url != _PROXY_URL else 2,
            )
            break
        except RuntimeError as exc:
            last_exc = exc
            if candidate_url == _PROXY_URL:
                raise
            log.warning("streaming proxy %s failed; falling back to %s", candidate_url, _PROXY_URL)
    else:
        raise RuntimeError(f"streaming proxy request failed: {last_exc!r}")
    return build_result_record(record, context, text, resolved_model, temperature)


def rebuild_prompt_if_needed(record: dict, cfg: dict, strategy: str | None) -> dict:
    if not strategy:
        return record
    from train.prompt_builder import get_builder

    pb_cfg = cfg.get("prompt_builder", {}).copy()
    pb_cfg["runs_dir"] = str(cfg["runs_dir"])
    pb_cfg["strategy"] = strategy
    builder = get_builder({**cfg, "prompt_builder": pb_cfg})
    rebuilt = dict(record)
    rebuilt["system"] = builder.system()
    rebuilt["prompt"] = builder.build(record)
    return rebuilt


async def run_synthesis(
    records: list[dict],
    cfg: dict,
    output_file: Path,
    skipped_file: Path,
    model: str,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    max_tex_chars: int,
) -> None:
    done_ids = load_done(output_file) | load_done(skipped_file)
    pending = [r for r in records if r.get("arxiv_id") not in done_ids]
    log.info("Synthesizing %d records (skip existing=%d)", len(pending), len(done_ids))

    client = _claude_client()
    semaphore = asyncio.Semaphore(concurrency)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    skipped_file.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = errors = 0
    start = time.time()

    async def _one(record: dict) -> dict | None:
        async with semaphore:
            try:
                return await synthesize_tex_target_async(
                    record,
                    cfg,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    max_tex_chars=max_tex_chars,
                    client=client,
                )
            except Exception as exc:
                log.warning("synthesis error for %s: %s", record.get("arxiv_id"), exc)
                return None

    tasks = [asyncio.create_task(_one(r)) for r in pending]
    for i, fut in enumerate(asyncio.as_completed(tasks), 1):
        row = await fut
        if row is None:
            errors += 1
        elif row.get("tex_status") == "ok" and row.get("target_impl_proposal"):
            append_jsonl(output_file, row)
            written += 1
        else:
            append_jsonl(skipped_file, row)
            skipped += 1

        if i % 20 == 0 or i == len(pending):
            elapsed = max(time.time() - start, 1e-6)
            rate = i / elapsed
            eta = (len(pending) - i) / rate if rate else 0.0
            log.info(
                "Progress %d/%d | written=%d skipped=%d errors=%d | %.2f/min | ETA %.1fm",
                i,
                len(pending),
                written,
                skipped,
                errors,
                rate * 60,
                eta / 60,
            )

    log.info("Done. written=%d skipped=%d errors=%d", written, skipped, errors)


def dry_run(records: list[dict], cfg: dict, max_tex_chars: int, require_tex: bool) -> None:
    arxiv_root = Path(cfg["arxiv_root"])
    ok = missing = 0
    for record in records:
        context = extract_tex_context(record.get("arxiv_id", ""), arxiv_root, max_tex_chars=max_tex_chars)
        status = context.get("tex_status")
        if status == "ok":
            ok += 1
            sections = context.get("tex_sections", [])
            headings = ", ".join(f"{s['heading']}[{s['kind']}]" for s in sections[:6])
            print(
                f"OK {record.get('arxiv_id')} files={context.get('tex_file_count')} "
                f"sections={context.get('tex_section_count')} selected={len(sections)}"
            )
            print(f"  {context.get('tex_dir')}")
            print(f"  {headings}")
        else:
            missing += 1
            if not require_tex:
                print(f"SKIP {record.get('arxiv_id')} status={status} dir={context.get('paper_dir')}")
    print(f"dry_run summary: ok={ok} missing_or_empty={missing} total={len(records)}")


def default_skipped_path(output_file: Path) -> Path:
    suffix = "".join(output_file.suffixes)
    if suffix:
        name = output_file.name.removesuffix(suffix)
        return output_file.with_name(f"{name}.skipped.jsonl")
    return output_file.with_name(output_file.name + ".skipped.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--input", help="Input dataset JSONL (default: runs/dataset/train.jsonl)")
    parser.add_argument("--output", help="Main TeX target JSONL (default: runs/dataset/train_tex_impl.jsonl)")
    parser.add_argument("--skipped-output", help="Skipped manifest JSONL (default: <output>.skipped.jsonl)")
    parser.add_argument("--strategy", default=None, help="Rebuild prompt with this prompt_builder strategy")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--arxiv-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-tex", action="store_true", help="In dry-run, hide missing-TeX records")
    parser.add_argument("--max-tex-chars", type=int, default=24000)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    runs_dir = Path(cfg["runs_dir"])
    input_file = Path(args.input) if args.input else runs_dir / "dataset" / "train.jsonl"
    output_file = Path(args.output) if args.output else runs_dir / "dataset" / "train_tex_impl.jsonl"
    skipped_file = Path(args.skipped_output) if args.skipped_output else default_skipped_path(output_file)

    records = load_records(input_file, limit=args.limit, arxiv_id=args.arxiv_id)
    records = [rebuild_prompt_if_needed(r, cfg, args.strategy) for r in records]
    log.info("Loaded %d records from %s", len(records), input_file)

    if args.dry_run:
        dry_run(records, cfg, max_tex_chars=args.max_tex_chars, require_tex=args.require_tex)
        return

    cot_cfg = cfg.get("cot", {})
    model = args.model or cot_cfg.get("model", "claude-sonnet-4-6")
    max_tokens = args.max_tokens or cot_cfg.get("max_tokens", 3072)
    asyncio.run(
        run_synthesis(
            records,
            cfg,
            output_file=output_file,
            skipped_file=skipped_file,
            model=model,
            max_tokens=max_tokens,
            temperature=args.temperature,
            concurrency=args.concurrency,
            max_tex_chars=args.max_tex_chars,
        )
    )


if __name__ == "__main__":
    main()
