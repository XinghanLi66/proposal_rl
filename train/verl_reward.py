#!/usr/bin/env python3
"""
verl reward function for proposal_rl.

verl calls compute_score(data_source, solution_str, ground_truth, extra_info)
and expects a float (or dict with "score" key).

data_source: one of "prs", "fas", "ppl" — set when building the Parquet dataset.
ground_truth: for PRS, the paper's abstract; for FAS/PPL, the same abstract.
extra_info: dict with optional "reward_type" override.

The sentence encoder and FAS index are initialised lazily on first call
(process-local globals, safe under Ray's one-worker-per-process model).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# verl loads this file via importlib.util.spec_from_file_location WITHOUT registering it in
# sys.modules, so exec_module runs the entire file fresh on every reward call, resetting all
# module-level globals (including _ppl_model) to None on each invocation.
# Fix: persist heavy objects in sys (a true process-level singleton that survives re-exec).
_PROC_CACHE_KEY = "_verl_reward_proc_cache"
if not hasattr(sys, _PROC_CACHE_KEY):
    setattr(sys, _PROC_CACHE_KEY, {})
_proc_cache: dict = getattr(sys, _PROC_CACHE_KEY)

import numpy as np
from sentence_transformers import SentenceTransformer

# ── shared constants ──────────────────────────────────────────────────────────

REQUIRED_TAGS = [
    "<problem>", "<gap>", "<key_insight>", "<approach>", "<expected_contributions>"
]

# Embed model path: env var or default HuggingFace id
_EMBED_MODEL = os.environ.get(
    "EMBED_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
# FAS index path: must be set when using FAS reward
_FAS_INDEX = os.environ.get("FAS_INDEX_FILE", "")

# PPL reward: path to the reference/SFT model used as the scoring model.
# Set REF_MODEL_PATH in train/rl.py to the sft/final checkpoint (same as actor base).
_REF_MODEL_PATH = os.environ.get("REF_MODEL_PATH", "")
_PPL_MAX_CTX_CHARS = int(os.environ.get("PPL_MAX_CTX_CHARS", "6000"))  # chars, not tokens
_PPL_MAX_ABS_CHARS = int(os.environ.get("PPL_MAX_ABS_CHARS", "800"))

# ── lazy globals (init on first call within each worker process) ──────────────

_encoder: SentenceTransformer | None = None
_fas_embs: np.ndarray | None = None
_fas_topk: int = 50

# PPL reward globals: backed by _proc_cache so they survive module re-exec by verl.
# (verl reloads this file on every reward call without caching it in sys.modules.)
_ppl_model = None      # local shadow; real value lives in _proc_cache
_ppl_tokenizer = None
_ppl_device: str = "cpu"

# ── rollout logging ───────────────────────────────────────────────────────────

_rollout_log_fh = None
_rollout_step = 0
_rollout_interval = 20
_rollout_per_step = 3


def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(_EMBED_MODEL)
    return _encoder


def _get_fas_embs() -> np.ndarray:
    global _fas_embs, _fas_topk
    if _fas_embs is None:
        index_file = _FAS_INDEX
        if not index_file or not Path(index_file).exists():
            raise RuntimeError(
                f"FAS index not found at {index_file!r}. "
                "Set FAS_INDEX_FILE env var before launching the worker."
            )
        data = np.load(index_file, allow_pickle=True)
        _fas_embs = data["embeddings"].astype(np.float32)
    return _fas_embs


# ── helpers ───────────────────────────────────────────────────────────────────

def _extract_proposal(text: str) -> str:
    m = re.search(r"<proposal>(.*?)</proposal>", text, re.DOTALL)
    if m:
        return re.sub(r"<[^>]+>", " ", m.group(1)).strip()
    return text


def _format_score(text: str) -> float:
    count = sum(1 for tag in REQUIRED_TAGS if tag in text)
    return count / len(REQUIRED_TAGS)


def _word_cosine(a: str, b: str) -> float:
    def bow(t: str):
        words = re.findall(r"\b\w+\b", t.lower())
        v: dict[str, int] = {}
        for w in words:
            v[w] = v.get(w, 0) + 1
        return v
    va, vb = bow(a), bow(b)
    keys = set(va) | set(vb)
    if not keys:
        return 0.0
    dot = sum(va.get(k, 0) * vb.get(k, 0) for k in keys)
    na = sum(v ** 2 for v in va.values()) ** 0.5
    nb = sum(v ** 2 for v in vb.values()) ** 0.5
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def _prs_score(proposal_text: str, abstract: str) -> float:
    enc = _get_encoder()
    embs = enc.encode(
        [proposal_text, abstract],
        batch_size=2,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    return float(np.dot(embs[0], embs[1]))


def _get_ppl_model():
    # Use _proc_cache (backed by sys) so the loaded model survives verl re-exec'ing this
    # module without registering it in sys.modules (which resets all module globals).
    if "ppl_model" not in _proc_cache:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_path = _REF_MODEL_PATH
        if not model_path or not Path(model_path).exists():
            raise RuntimeError(
                f"REF_MODEL_PATH not set or not found: {model_path!r}. "
                "Set REF_MODEL_PATH env var in train/rl.py to the sft/final checkpoint."
            )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # Load directly onto GPU if VRAM is available, else fall back to CPU.
        # low_cpu_mem_usage avoids the 2× peak-RAM spike during weight loading.
        if torch.cuda.is_available():
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    device_map="cuda:0",
                ).eval()
                device = "cuda:0"
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                log.warning("PPL model: GPU OOM, falling back to CPU (bfloat16)")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                ).eval()
                device = "cpu"
        else:
            # bfloat16 halves CPU RAM vs float32 (14 GB vs 28 GB) and is supported
            # by torch.autocast(device_type="cpu", dtype=torch.bfloat16).
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            ).eval()
            device = "cpu"
        _proc_cache["ppl_model"] = model
        _proc_cache["ppl_tokenizer"] = tokenizer
        _proc_cache["ppl_device"] = device
    return _proc_cache["ppl_model"], _proc_cache["ppl_tokenizer"]


def _ppl_score(proposal_text: str, abstract: str,
               max_ctx_chars: int = _PPL_MAX_CTX_CHARS,
               max_abs_chars: int = _PPL_MAX_ABS_CHARS) -> float:
    """Mean log-prob of abstract tokens given proposal context under the ref/SFT model.

    Measures P_policy(abstract | proposal): whether the model, given this proposal,
    would naturally generate the correct abstract. Higher = better proposal quality.
    Returns exp(-mean_CE / scale) ∈ (0, 1].
    """
    import torch
    model, tokenizer = _get_ppl_model()
    device = _proc_cache.get("ppl_device", "cpu")

    prop_text = proposal_text[:max_ctx_chars]
    abs_text = abstract[:max_abs_chars]

    # Tokenize proposal and abstract separately to know where abstract starts
    prop_ids = tokenizer.encode(prop_text, add_special_tokens=False)
    abs_ids = tokenizer.encode(abs_text, add_special_tokens=False)

    if not abs_ids:
        return 0.0

    # Concatenate: [BOS] + proposal_tokens + abstract_tokens
    # Truncate proposal tail if total exceeds model max length
    max_len = getattr(tokenizer, "model_max_length", 4096)
    max_len = min(max_len, 4096)  # cap at 4096 to keep memory bounded
    max_prop = max(0, max_len - len(abs_ids) - 1)
    prop_ids = prop_ids[-max_prop:]  # keep the end of the proposal (most relevant)

    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    input_ids = torch.tensor([bos + prop_ids + abs_ids], dtype=torch.long, device=device)

    n_abs = len(abs_ids)
    if input_ids.shape[1] <= 1:
        return 0.0

    # autocast: fp16 on CUDA, bfloat16 on CPU (both supported by torch.autocast).
    ppl_device = _proc_cache.get("ppl_device", "cpu")
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if ppl_device.startswith("cuda")
        else torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    )
    with torch.no_grad(), autocast_ctx:
        logits = model(input_ids).logits  # [1, T, V]

    # Score only the abstract token positions
    # prediction at position i uses logits[i-1]
    abs_token_ids = input_ids[0, -n_abs:]                  # [n_abs]
    pred_logits = logits[0, -(n_abs + 1):-1, :].float()   # [n_abs, V]
    log_probs = torch.nn.functional.log_softmax(pred_logits, dim=-1)
    token_lp = log_probs[range(n_abs), abs_token_ids]      # [n_abs]
    mean_ce = -token_lp.mean().item()

    # Qwen2.5-7B typical CE on in-domain text ≈ 2–3 nats; scale so ~2.5 nats → ~0.37
    # Using /3.0 keeps the same scale as before, range (0, 1]
    return float(min(1.0, max(0.0, np.exp(-mean_ce / 3.0))))


def _fas_score(proposal_text: str) -> float:
    enc = _get_encoder()
    emb = enc.encode(
        [proposal_text],
        batch_size=1,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    index_embs = _get_fas_embs()
    sims = emb @ index_embs.T  # [1, N]
    n = sims.shape[1]
    topk = min(_fas_topk, max(1, n - 1))
    top_sims = np.partition(-sims[0], topk)[:topk]
    return float((-top_sims).mean())


# ── verl reward entry point ───────────────────────────────────────────────────

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    antileak_threshold: float = 0.80,
) -> dict:
    """
    verl reward function.

    data_source: "prs" | "fas" | "ppl" — selects reward mode.
    solution_str: the model's raw completion text.
    ground_truth: the paper's abstract.
    extra_info: optional dict (legacy; "antileak_threshold" key also accepted here).
    antileak_threshold: passed directly by verl via reward_kwargs.

    Returns dict with "score" and per-component breakdowns.
    """
    global _rollout_step
    extra_info = extra_info or {}
    # extra_info key takes precedence if set
    antileak_threshold = float(extra_info.get("antileak_threshold", antileak_threshold))

    proposal_text = _extract_proposal(solution_str)
    fmt = _format_score(solution_str)

    if data_source == "prs":
        prs = _prs_score(proposal_text, ground_truth)
        score = 0.8 * prs + 0.2 * fmt
        result = {"score": score, "prs": prs, "format": fmt}

    elif data_source == "fas":
        fas = _fas_score(proposal_text)
        threshold = antileak_threshold
        sim = _word_cosine(proposal_text, ground_truth)
        antileak = 1.0 if sim <= threshold else max(0.0, 1.0 - 5.0 * (sim - threshold))
        score = 0.6 * fas + 0.2 * fmt + 0.2 * antileak
        result = {"score": score, "fas": fas, "format": fmt, "antileak": antileak}

    elif data_source == "ppl":
        # PPL reward: exp(-CE/3) of abstract tokens under GPT-2 given proposal context.
        # Measures how predictable the abstract is from the generated proposal.
        ppl = _ppl_score(proposal_text, ground_truth)
        score = 0.8 * ppl + 0.2 * fmt
        result = {"score": score, "ppl": ppl, "format": fmt}

    else:
        # fallback: format-only
        result = {"score": fmt, "format": fmt}

    # Rollout logging
    _rollout_step += 1
    if _rollout_log_fh is not None and _rollout_step % _rollout_interval == 0:
        try:
            _rollout_log_fh.write(json.dumps({
                "step":     _rollout_step,
                "stage":    "rl",
                "reward":   data_source,
                "output":   solution_str[:500],
                "abstract": ground_truth[:200],
                "ts":       time.time(),
                **result,
            }) + "\n")
        except Exception:
            pass

    return result


def open_rollout_log(path: str) -> None:
    global _rollout_log_fh
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _rollout_log_fh = open(path, "a", buffering=1)
