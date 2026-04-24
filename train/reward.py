"""
Reward functions for RL training.

Two reward modes, controlled by rl.reward_type in the config:

  prs  — Paper Recovery Score (default)
         Cosine similarity between the generated proposal and the actual
         abstract of the source paper.  No external index needed.
         reward = α·PRS + β·format

  fas  — Future Alignment Score (legacy)
         Embedding similarity to a held-out corpus of future papers.
         reward = α·FAS + β·format + γ·antileak

All reward functions follow the TRL 1.0.0 signature:
    fn(prompts, completions, **kwargs) -> list[float]
The `abstract` dataset column is passed via kwargs for both PRS and antileak.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

REQUIRED_TAGS = [
    "<problem>", "<gap>", "<key_insight>", "<approach>", "<expected_contributions>"
]


def extract_proposal_text(text: str) -> str:
    m = re.search(r"<proposal>(.*?)</proposal>", text, re.DOTALL)
    if m:
        return re.sub(r"<[^>]+>", " ", m.group(1)).strip()
    return text


def word_cosine(a: str, b: str) -> float:
    def bow(t):
        words = re.findall(r'\b\w+\b', t.lower())
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


# ── Shared encoder (used by PRS; FASReward has its own) ──────────────────────

_encoder: SentenceTransformer | None = None


def init_encoder(
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
    rollout_log_file: str | None = None,
) -> None:
    """Initialise the shared encoder for PRS mode."""
    global _encoder
    print(f"[Reward] Loading encoder: {embed_model}")
    _encoder = SentenceTransformer(embed_model, device=device)
    if rollout_log_file:
        _open_rollout_log(rollout_log_file)


# ── PRS reward ────────────────────────────────────────────────────────────────

def reward_prs(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """
    Paper Recovery Score.
    Cosine similarity between the generated proposal and the source abstract.
    abstract is passed via kwargs["abstract"] (dataset column).
    """
    global _rollout_step
    assert _encoder is not None, "Call init_encoder() before using reward_prs"

    abstracts = list(kwargs.get("abstract", [""] * len(completions)))
    proposal_texts = [extract_proposal_text(c) for c in completions]

    # Encode proposals and abstracts in one batch for efficiency
    all_embs = _encoder.encode(
        proposal_texts + abstracts,
        batch_size=min(len(completions) * 2, 32),
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    proposal_embs = all_embs[:len(completions)]
    abstract_embs = all_embs[len(completions):]

    # Pairwise cosine similarity (L2-normalised → dot product)
    scores = (proposal_embs * abstract_embs).sum(axis=1).tolist()

    # Rollout logging
    _rollout_step += 1
    if _rollout_log_fh is not None and _rollout_step % _rollout_interval == 0:
        for i in range(min(_rollout_per_step, len(completions))):
            _rollout_log_fh.write(json.dumps({
                "step":     _rollout_step,
                "stage":    "rl",
                "reward":   "prs",
                "prompt":   prompts[i][-300:] if prompts else "",
                "output":   completions[i],
                "score":    scores[i],
                "abstract": abstracts[i][:200],
                "ts":       time.time(),
            }) + "\n")

    return scores


# ── FAS reward (legacy) ───────────────────────────────────────────────────────

class FASReward:
    """Embedding-based future alignment reward. Pre-loads the val corpus index."""

    def __init__(
        self,
        index_file: str,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        topk: int = 50,
        device: str = "cpu",
    ):
        self.topk = topk
        print(f"[FASReward] Loading encoder: {embed_model}")
        self.encoder = SentenceTransformer(embed_model, device=device)
        print(f"[FASReward] Loading index: {index_file}")
        data = np.load(index_file, allow_pickle=True)
        self.index_embs = data["embeddings"].astype(np.float32)
        self.index_ids = data["arxiv_ids"].tolist()
        print(f"[FASReward] Index loaded: {self.index_embs.shape}")

    def score(self, texts: list[str]) -> list[float]:
        if not texts:
            return []
        embs = self.encoder.encode(
            texts,
            batch_size=min(len(texts), 32),
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        sims = embs @ self.index_embs.T
        n = sims.shape[1]
        topk = min(self.topk, n)
        top_sims = np.partition(-sims, topk, axis=1)[:, :topk]
        return (-top_sims).mean(axis=1).tolist()


_fas_reward: FASReward | None = None
_fas_weight: float = 0.6
_format_weight: float = 0.2
_antileak_weight: float = 0.2
_antileak_threshold: float = 0.80


def init_reward(
    index_file: str,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    topk: int = 50,
    fas_weight: float = 0.6,
    format_weight: float = 0.2,
    antileak_weight: float = 0.2,
    antileak_threshold: float = 0.80,
    rollout_log_file: str | None = None,
) -> None:
    """Initialise FAS mode rewards."""
    global _fas_reward, _fas_weight, _format_weight, _antileak_weight, _antileak_threshold
    _fas_reward = FASReward(index_file, embed_model, topk)
    _fas_weight = fas_weight
    _format_weight = format_weight
    _antileak_weight = antileak_weight
    _antileak_threshold = antileak_threshold
    if rollout_log_file:
        _open_rollout_log(rollout_log_file)


def reward_fas(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """FAS component: mean embedding similarity to val corpus top-K."""
    global _rollout_step
    assert _fas_reward is not None, "Call init_reward() before using reward_fas"
    texts = [extract_proposal_text(c) for c in completions]
    scores = _fas_reward.score(texts)

    _rollout_step += 1
    if _rollout_log_fh is not None and _rollout_step % _rollout_interval == 0:
        for i in range(min(_rollout_per_step, len(completions))):
            _rollout_log_fh.write(json.dumps({
                "step":   _rollout_step,
                "stage":  "rl",
                "reward": "fas",
                "prompt": prompts[i][-300:] if prompts else "",
                "output": completions[i],
                "score":  scores[i],
                "ts":     time.time(),
            }) + "\n")

    return scores


def reward_format(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """Format component: fraction of required XML tags present."""
    return [
        sum(1 for tag in REQUIRED_TAGS if tag in c) / len(REQUIRED_TAGS)
        for c in completions
    ]


def reward_antileak(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """Anti-leakage: penalises proposals with high word-overlap with the source abstract."""
    abstracts = kwargs.get("abstract", [""] * len(completions))
    scores = []
    for c, abstract in zip(completions, abstracts):
        sim = word_cosine(extract_proposal_text(c), abstract)
        reward = 1.0 if sim <= _antileak_threshold else max(0.0, 1.0 - 5.0 * (sim - _antileak_threshold))
        scores.append(reward)
    return scores


# ── Rollout logger ────────────────────────────────────────────────────────────

_rollout_log_fh = None
_rollout_step = 0
_rollout_interval = 20
_rollout_per_step = 3


def _open_rollout_log(path: str) -> None:
    global _rollout_log_fh
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _rollout_log_fh = open(path, "a", buffering=1)
