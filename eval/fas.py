"""
FAS (Future Alignment Score) evaluators.

Two strategies:
  embedding  — sentence encoder cosine similarity (fast, used for training reward)
  llm_judge  — embedding retrieval + LLM judge (aligns with arXiv:2603.27146)

Reference paper: text-embedding-3-large retrieval, GPT-4.1-mini judge, K=10, MAX aggregation.
Our llm_judge uses the configured embed_model for retrieval and Claude as judge.

Usage:
    index = load_index(path)
    evaluator = get_fas_evaluator(cfg)
    result = evaluator.score("proposal text", "2501.12345", index)
    # -> {"FAS": 0.63, "recall_at_k": 1.0, "mean_sim": 0.56}
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

_PROXY_URL = "http://10.39.10.241:10001"


def load_index(path: str | Path) -> dict:
    """Load a .npz FAS index. Returns dict with embeddings, arxiv_ids, abstracts."""
    data = np.load(path, allow_pickle=True)
    return {
        "embeddings": data["embeddings"].astype(np.float32),
        "arxiv_ids": data["arxiv_ids"].tolist(),
        "abstracts": data["abstracts"].tolist() if "abstracts" in data else [],
    }


def extract_proposal_text(text: str) -> str:
    """Extract clean text from <proposal>…</proposal>, stripping XML tags."""
    m = re.search(r"<proposal>(.*?)</proposal>", text, re.DOTALL)
    if m:
        return re.sub(r"<[^>]+>", " ", m.group(1)).strip()
    return text


class FASEvaluator(ABC):
    @abstractmethod
    def score(self, proposal_text: str, arxiv_id: str, index: dict) -> dict:
        """Score one proposal. Returns dict with at least 'FAS' and 'recall_at_k'."""

    def score_batch(
        self, proposal_texts: list[str], arxiv_ids: list[str], index: dict
    ) -> list[dict]:
        """Default: call score() per example. Override for batch efficiency."""
        return [self.score(t, aid, index) for t, aid in zip(proposal_texts, arxiv_ids)]


class EmbeddingFAS(FASEvaluator):
    """
    Fast embedding-based FAS.
    FAS = 0.5 * recall@topk + 0.5 * mean_cosine_sim(top-topk)
    """

    def __init__(self, cfg: dict) -> None:
        from sentence_transformers import SentenceTransformer
        self.topk = cfg.get("topk", 50)
        self.encoder = SentenceTransformer(
            cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")
        )

    def score(self, proposal_text: str, arxiv_id: str, index: dict) -> dict:
        return self.score_batch([proposal_text], [arxiv_id], index)[0]

    def score_batch(
        self, proposal_texts: list[str], arxiv_ids: list[str], index: dict
    ) -> list[dict]:
        embs = self.encoder.encode(
            proposal_texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

        index_embs = index["embeddings"]
        index_ids = index["arxiv_ids"]
        id_to_idx = {aid: i for i, aid in enumerate(index_ids)}
        topk = min(self.topk, index_embs.shape[0])

        sims = embs @ index_embs.T                                   # [B, N]
        top_idx = np.argpartition(-sims, topk, axis=1)[:, :topk]    # [B, topk]
        top_sims = np.take_along_axis(sims, top_idx, axis=1)        # [B, topk]

        results = []
        for i, arxiv_id in enumerate(arxiv_ids):
            true_idx = id_to_idx.get(arxiv_id)
            recall = 1.0 if (true_idx is not None and true_idx in top_idx[i]) else 0.0
            mean_sim = float(top_sims[i].mean())
            fas = 0.5 * recall + 0.5 * mean_sim
            best_j = top_idx[i][np.argmax(top_sims[i])]
            results.append({
                "FAS": round(fas, 4),
                "recall_at_k": recall,
                "mean_sim": round(mean_sim, 4),
                "topk": topk,
                "top1_id": index_ids[best_j],
            })
        return results


class LLMJudgeFAS(FASEvaluator):
    """
    LLM-judge FAS (aligns with arXiv:2603.27146).

    1. Retrieve top judge_topk papers by embedding similarity.
    2. Score each (proposal, abstract) pair with Claude (0-10).
    3. FAS = max_score / 10  (MAX aggregation, same as reference paper).

    Note: slow — N * judge_topk API calls per batch. Use embedding for training reward.
    """

    _JUDGE_PROMPT = (
        "Rate how well this research proposal predicts or aligns with the future paper's abstract.\n"
        "Consider whether the proposal anticipates the same research direction, problem, or approach.\n\n"
        "Research proposal:\n{proposal}\n\n"
        "Future paper abstract:\n{abstract}\n\n"
        "Score from 0 to 10 (integer only, 0=no alignment, 10=perfect alignment):"
    )

    def __init__(self, cfg: dict) -> None:
        from sentence_transformers import SentenceTransformer
        import os
        import anthropic
        self.judge_topk = cfg.get("judge_topk", 10)
        self.judge_model = cfg.get("judge_model", "claude-sonnet-4-6")
        self.encoder = SentenceTransformer(
            cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")
        )
        api_key = os.environ.get("ANTHROPIC_API_KEY") or "123"
        self.client = anthropic.Anthropic(
            api_key=api_key, base_url=_PROXY_URL, timeout=120.0, max_retries=2
        )

    def _judge(self, proposal: str, abstract: str) -> float:
        """Ask LLM to score one (proposal, abstract) pair. Returns value in [0, 1]."""
        prompt = self._JUDGE_PROMPT.format(
            proposal=proposal[:2000], abstract=abstract[:1000]
        )
        try:
            with self.client.messages.stream(
                model=self.judge_model,
                max_tokens=8,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                text = stream.get_final_text().strip()
            m = re.search(r"\d+", text)
            return min(int(m.group()), 10) / 10.0 if m else 0.0
        except Exception:
            return 0.0

    def score(self, proposal_text: str, arxiv_id: str, index: dict) -> dict:
        emb = self.encoder.encode(
            [proposal_text], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

        index_embs = index["embeddings"]
        index_ids = index["arxiv_ids"]
        abstracts = index.get("abstracts", [])
        id_to_idx = {aid: i for i, aid in enumerate(index_ids)}

        sims = (emb @ index_embs.T)[0]
        topk = min(self.judge_topk, len(sims))
        top_idx = np.argpartition(-sims, topk)[:topk]

        true_idx = id_to_idx.get(arxiv_id)
        recall = 1.0 if (true_idx is not None and true_idx in top_idx) else 0.0

        scores = [
            self._judge(proposal_text, abstracts[j] if j < len(abstracts) else "")
            for j in top_idx
        ]
        fas = max(scores) if scores else 0.0

        return {
            "FAS": round(fas, 4),
            "recall_at_k": recall,
            "max_judge_score": round(fas, 4),
            "mean_judge_score": round(float(np.mean(scores)) if scores else 0.0, 4),
            "topk": topk,
        }


def get_fas_evaluator(cfg: dict) -> FASEvaluator:
    """Factory: returns FASEvaluator specified by cfg['fas']['strategy']."""
    fas_cfg = cfg.get("fas", {})
    strategy = fas_cfg.get("strategy", "embedding")
    if strategy == "embedding":
        return EmbeddingFAS(fas_cfg)
    elif strategy == "llm_judge":
        return LLMJudgeFAS(fas_cfg)
    raise ValueError(
        f"Unknown fas.strategy: {strategy!r}. Choose 'embedding' or 'llm_judge'."
    )
