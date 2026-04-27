#!/usr/bin/env python3
"""
Compute Future Alignment Score (FAS) for a trained model checkpoint.

FAS strategy is controlled by cfg['fas']['strategy']:
  embedding  — fast cosine similarity (default)
  llm_judge  — Claude-scored alignment (slow, aligns with arXiv:2603.27146)

Also computes format score and anti-leakage score.
Saves a summary JSON and per-example results to the checkpoint directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import numpy as np
from sentence_transformers import SentenceTransformer

from eval.fas import extract_proposal_text, get_fas_evaluator, load_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REQUIRED_TAGS = ["<problem>", "<gap>", "<key_insight>", "<approach>", "<expected_contributions>"]


def format_score(text: str) -> float:
    return sum(1 for tag in REQUIRED_TAGS if tag in text) / len(REQUIRED_TAGS)


def word_cosine(a: str, b: str) -> float:
    def bow(t):
        words = re.findall(r'\b\w+\b', t.lower())
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


def compute_prs(proposal_texts: list[str], abstracts: list[str], encoder: SentenceTransformer) -> list[float]:
    """Paper Recovery Score: cosine sim between proposal embedding and source abstract embedding."""
    all_embs = encoder.encode(
        proposal_texts + abstracts,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    prop_embs = all_embs[:len(proposal_texts)]
    abst_embs = all_embs[len(proposal_texts):]
    return (prop_embs * abst_embs).sum(axis=1).tolist()


def generate_proposals(model, tokenizer, records, batch_size, max_new_tokens, device) -> list[str]:
    proposals = []
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        texts = []
        for rec in batch:
            messages = [
                {"role": "system", "content": rec["system"]},
                {"role": "user", "content": rec["prompt"]},
            ]
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=8192)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        for j in range(len(batch)):
            prompt_len = enc["input_ids"].shape[1]
            proposals.append(tokenizer.decode(out[j][prompt_len:], skip_special_tokens=True))

        log.info(f"Generated {min(i+batch_size, len(records))}/{len(records)}")
    return proposals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", help="Base model path for LoRA checkpoints")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=None, help="Override fas.topk from config")
    parser.add_argument("--output-dir", help="Override output directory")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    runs_dir = Path(cfg["runs_dir"])

    # Allow --topk to override config
    if args.topk:
        cfg.setdefault("fas", {})["topk"] = args.topk

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_path = args.base_model or cfg.get("model_name_or_path")
    tokenizer = AutoTokenizer.from_pretrained(base_path or args.checkpoint, trust_remote_code=True)
    tokenizer.padding_side = "left"

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, args.checkpoint).merge_and_unload()
        log.info("Loaded LoRA checkpoint")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        log.info("Loaded full model checkpoint")
    model.eval()

    # Load dataset
    dataset_file = runs_dir / "dataset" / f"{args.split}.jsonl"
    records = []
    with open(dataset_file) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    if args.limit:
        records = records[:args.limit]
    log.info(f"Evaluating on {len(records)} examples from {args.split}")

    # Generate proposals
    t0 = time.time()
    proposals = generate_proposals(model, tokenizer, records, args.batch_size, args.max_new_tokens, device)
    gen_time = time.time() - t0
    log.info(f"Generation done in {gen_time:.0f}s")

    proposal_texts = [extract_proposal_text(p) for p in proposals]
    arxiv_ids      = [r["arxiv_id"] for r in records]
    abstracts      = [r.get("abstract", "") for r in records]

    # FAS evaluation
    index_file = runs_dir / "eval" / f"{args.split}_index.npz"
    index = load_index(index_file)
    evaluator = get_fas_evaluator(cfg)
    log.info(f"FAS evaluator: {type(evaluator).__name__}")
    if type(evaluator).__name__ == "LLMJudgeFAS":
        log.warning(f"LLMJudgeFAS: {len(records)} × {evaluator.judge_topk} judge calls — slow")
    fas_results = evaluator.score_batch(proposal_texts, arxiv_ids, index)

    # PRS — reuse the encoder already inside the FAS evaluator when possible
    log.info("Computing PRS...")
    encoder = getattr(evaluator, "encoder", None) or SentenceTransformer(
        cfg.get("fas", {}).get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")
    )
    prs_scores = compute_prs(proposal_texts, abstracts, encoder)

    # Per-example metrics
    results = []
    format_scores, leakage_scores = [], []
    for i, rec in enumerate(records):
        fmt  = format_score(proposals[i])
        leak = word_cosine(proposal_texts[i], abstracts[i])
        format_scores.append(fmt)
        leakage_scores.append(leak)
        results.append({
            "arxiv_id": rec["arxiv_id"],
            **fas_results[i],
            "PRS": round(prs_scores[i], 4),
            "format_score": fmt,
            "leakage_score": round(leak, 4),
            "proposal": proposals[i],
        })

    fas_values      = [r["FAS"] for r in fas_results]
    recall_values   = [r["recall_at_k"] for r in fas_results]
    mean_sim_values = [r.get("mean_sim", r.get("max_judge_score", 0.0)) for r in fas_results]

    summary = {
        "checkpoint":          str(args.checkpoint),
        "split":               args.split,
        "fas_strategy":        cfg.get("fas", {}).get("strategy", "embedding"),
        "n_examples":          len(records),
        "topk":                fas_results[0]["topk"] if fas_results else 0,
        "FAS":                 round(float(np.mean(fas_values)), 4),
        "recall_at_k":         round(float(np.mean(recall_values)), 4),
        "mean_similarity":     round(float(np.mean(mean_sim_values)), 4),
        "PRS":                 round(float(np.mean(prs_scores)), 4),
        "format_score":        round(float(np.mean(format_scores)), 4),
        "leakage_score_mean":  round(float(np.mean(leakage_scores)), 4),
        "leakage_flagged_pct": round(float(np.mean([s > 0.85 for s in leakage_scores])), 4),
        "gen_time_s":          round(gen_time, 1),
    }

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint) / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out_dir / "per_example.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    log.info(f"\n{'='*50}")
    log.info(f"FAS ({summary['fas_strategy']}): {summary['FAS']:.4f}")
    log.info(f"Recall@{summary['topk']}:        {summary['recall_at_k']:.4f}")
    log.info(f"Mean sim:                        {summary['mean_similarity']:.4f}")
    log.info(f"PRS:                             {summary['PRS']:.4f}")
    log.info(f"Format score:                    {summary['format_score']:.4f}")
    log.info(f"Leakage mean:                    {summary['leakage_score_mean']:.4f}")
    log.info(f"Results saved → {out_dir}")


if __name__ == "__main__":
    main()
