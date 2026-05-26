"""
Canonical checkpoint registry for benchmark sweeps.

Each entry: (display_label, path_relative_to_runs/exps/, strategy)

Import this in both benchmark_tui.py and sweep.py so the list stays in sync.
Add new entries here when new rl/final checkpoints become available.
"""
from __future__ import annotations

# (label, path relative to runs/exps/, prompt strategy)
CHECKPOINT_REGISTRY: list[tuple[str, str, str]] = [
    # Base model — Qwen2.5-7B-Instruct, no fine-tuning
    ("qwen25 7b-instruct base",         "qwen25_7b_instruct_base",                             "top_k_refs"),
    # exp09 — top_k_refs, full-FT, PRS reward
    ("exp09 top_k_refs  ft-prs  [old]", "exp09_top_k_refs_sft_rl_20260510_193056/rl/final",  "top_k_refs"),
    ("exp09 top_k_refs  ft-prs  [new]", "exp09_top_k_refs_sft_rl_20260515_150827/rl/final",  "top_k_refs"),
    # exp10 — related_work, full-FT, PRS reward
    ("exp10 related_work ft-prs [old]", "exp10_related_work_sft_rl_20260511_032856/rl/final", "related_work"),
    ("exp10 related_work ft-prs [new]", "exp10_related_work_sft_rl_20260515_150844/rl/final", "related_work"),
    # exp11 — topk_rw hybrid, full-FT, PRS reward
    ("exp11 topk_rw      ft-prs  [old]","exp11_topk_rw_sft_rl_20260511_033056/rl/final",       "top_k_related_work"),
    ("exp11 topk_rw      ft-prs  [new]","exp11_topk_rw_sft_rl_20260514_044424/rl/final",       "top_k_related_work"),
    # exp12 — research_question, full-FT, PRS reward
    ("exp12 research_q   ft-prs",       "exp12_research_q_sft_rl_20260510_193056/rl/final",    "with_research_question"),
    # exp13 — full_refs, full-FT, PRS reward
    ("exp13 full_refs    ft-prs",       "exp13_full_refs_sft_rl_20260510_192856/rl/final",      "full_refs"),
    # exp14 — full_refs, LoRA, PRS reward
    ("exp14 full_refs    lora-prs",     "exp14_full_refs_lora_sft_rl_20260511_032856/rl/final", "full_refs"),
    # exp15 — top_k_refs, full-FT, FAS reward, init from exp09 SFT
    ("exp15 top_k_refs  ft-fas  [old]", "exp15_fas_from_exp09_sft_20260512_214236/rl/final",   "top_k_refs"),
    ("exp15 top_k_refs  ft-fas  [new]", "exp15_fas_from_exp09_sft_20260515_190523/rl/final",   "top_k_refs"),
    # exp16 — full_refs 20×800, full-FT, PRS reward
    ("exp16 full_refs   20x800-prs",    "exp16_full_refs_20x800_sft_rl_20260511_032856/rl/final", "full_refs"),
    # exp17 — top_k_refs, full-FT, PPL reward, init from exp09 SFT
    ("exp17 top_k_refs  ft-ppl  [old]", "exp17_top_k_refs_ppl_rl_20260511_102248/rl/final",    "top_k_refs"),
    ("exp17 top_k_refs  ft-ppl  [new]", "exp17_top_k_refs_ppl_rl_20260520_023410/rl/final",    "top_k_refs"),
]
