# proposal_rl

Training pipeline for **arXiv research proposal generation** via SFT + GRPO reinforcement learning. Given a paper's reference list, the model learns to generate a research proposal that aligns with the paper's actual abstract and future direction.

Built on top of [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) and [TRL](https://github.com/huggingface/trl).

---

## Overview

```
References  →  [Prompt Builder]  →  LLM  →  Proposal
                                              ↓
                                         [Reward]
                                    PRS: cosine_sim(proposal, abstract)
                                    FAS: sim(proposal, future corpus)
```

The pipeline has three stages:

1. **Data** — fetch reference metadata (Semantic Scholar), build train/val/test splits, synthesize chain-of-thought proposals with Claude
2. **SFT** — supervised fine-tuning on CoT proposals (`train_cot.jsonl`)
3. **RL** — GRPO/RLOO reinforcement learning with PRS or FAS reward signal

---

## Repository Structure

```
configs/          YAML configs and DeepSpeed ZeRO-2 config
data/             Data pipeline scripts (fetch, build, CoT synthesis)
eval/             Evaluation: FAS/PRS scoring, embedding index builder
train/            Training code: SFT, RL (GRPO/RLOO), reward functions, prompt builders
dashboard/        Flask training dashboard with per-experiment live tabs
scripts/          End-to-end orchestration shell scripts (00–07)
scripts/ablations/  8 ablation experiments with hparam search (exp01–exp08)
runs/             Generated at runtime — datasets, checkpoints, logs (git-ignored)
```

---

## Quickstart

### 1. Environment

```bash
conda create -n loongflow_ml python=3.11
conda activate loongflow_ml
pip install trl==1.0.0 transformers peft torch deepspeed flash-attn \
            sentence-transformers flask pyyaml httpx
```

### 2. Download base model

```bash
python scripts/00_download_model.py
```

### 3. Build dataset

```bash
bash scripts/01_fetch_refs.sh        # fetch reference metadata via Semantic Scholar
bash scripts/02_build_dataset.sh     # build train/val/test JSONL splits
bash scripts/03_synthesize_cot.sh    # synthesize CoT proposals with Claude
bash scripts/04_build_index.sh       # build val-set embedding index for FAS eval
```

Data splits use arXiv papers from cs.LG / cs.AI / cs.CL / cs.CV / cs.IR / cs.NE / stat.ML:

| Split | Months |
|-------|--------|
| Train | 2025-04 → 2025-10 |
| Val   | 2025-11 → 2025-12 |
| Test  | 2026-01 → 2026-03 |

### 4. SFT

```bash
bash scripts/05_sft.sh
# checkpoint → runs/sft/final/
```

### 5. RL fine-tuning

```bash
bash scripts/06_grpo.sh
# checkpoint → runs/rl/final/
```

### 6. Evaluate

```bash
bash scripts/07_evaluate.sh
# results → runs/eval/
```

---

## Reward Signals

| Name | Key | Description |
|------|-----|-------------|
| **PRS** (Paper Recovery Score) | `reward_type: prs` | Cosine similarity between generated proposal and source abstract using a sentence encoder. Measures whether the model correctly identifies the paper's core contribution. |
| **FAS** (Future Alignment Score) | `reward_type: fas` | Similarity to a held-out future corpus index (val set). Measures whether the proposal anticipates directions that actually materialized in the literature. |
| **Format** | always on | Binary reward for correct `<proposal>…</proposal>` tag structure. |
| **Anti-leakage** | FAS mode | Penalizes proposals that directly quote the abstract (prevents shortcut). |

---

## Prompt Builder Strategies

Controlled by `prompt_builder.strategy` in `configs/base.yaml`:

| Strategy | Description |
|----------|-------------|
| `full_refs` | All references included verbatim (up to `max_refs`, truncated to 400 chars each) |
| `top_k_refs` | LLM selects the K most relevant references, feeds those as a numbered list |
| `related_work` | LLM synthesises a related-work paragraph from all references |
| `top_k_related_work` | Two-stage: LLM selects top-K refs, then synthesises a related-work paragraph from those |
| `with_research_question` | `top_k_refs` + LLM-generated 1–3 sentence research question appended |

LLM-based strategies cache their outputs under `runs/dataset/prompt_cache/` to avoid redundant API calls across training runs.

---

## Ablation Experiments

Eight ablation experiments isolate the effect of each design choice. Each script runs a 2×2 hparam search (LR ∈ {2e-6, 5e-6} × KL ∈ {0.02, 0.05}) before full training.

| Script | Prompt strategy | Finetune | Reward | Ablates |
|--------|----------------|----------|--------|---------|
| `exp01_baseline.sh` | `top_k_refs` | full | PRS | — baseline |
| `exp02_full_refs.sh` | `full_refs` | full | PRS | top-K selection vs. full list |
| `exp03_related_work.sh` | `related_work` | full | PRS | narrative vs. list conditioning |
| `exp04_topk_related_work.sh` | `top_k_related_work` | full | PRS | two-stage LLM conditioning |
| `exp05_topk_research_question.sh` | `with_research_question` | full | PRS | explicit RQ vs. none |
| `exp06_lora.sh` | `top_k_refs` | LoRA | PRS | full-FT vs. LoRA |
| `exp07_fas.sh` | `top_k_refs` | full | FAS | PRS vs. FAS reward |
| `exp08_llm_prs.sh` | `top_k_refs` | full | LLM-judge | embedding vs. LLM-judge reward |

Run an experiment:

```bash
# Single experiment (foreground)
bash scripts/ablations/exp01_baseline.sh 2>&1 | tee /tmp/exp01.log

# Override GPU count or dashboard port
NGPU=4 DASHBOARD_PORT=8081 bash scripts/ablations/exp01_baseline.sh
```

See [`scripts/README.md`](scripts/README.md) for the full monitoring and multi-machine deployment guide.

---

## Training Dashboard

```bash
conda run -n loongflow_ml python dashboard/server.py --port 8080
# Open http://localhost:8080
```

Each ablation script self-registers as a tab when it starts. The dashboard shows live training curves, hparam search results, checkpoint history, rollout examples, and failure alerts.

---

## Configuration

All hyperparameters live in `configs/base.yaml`. Key sections:

```yaml
prompt_builder:
  strategy: top_k_refs   # prompt strategy
  top_k: 5               # refs to select (LLM-based strategies)

rl:
  algo: grpo             # grpo | rloo
  finetune_mode: lora    # lora | full
  reward_type: prs       # prs | fas
  num_generations: 8     # GRPO rollout group size
  kl_coeff: 0.05
  learning_rate: 5.0e-6
```

Override specific values without editing the file:

```bash
python scripts/make_config.py \
  --base configs/base.yaml \
  --out  /tmp/my_config.yaml \
  --set  rl.learning_rate=2e-6 rl.kl_coeff=0.02
```

---

## Citation

```bibtex
@misc{proposalrl2026,
  title  = {Learning to Propose: RL-Trained Research Proposal Generation from Reference Lists},
  author = {XinghanLi66},
  year   = {2026},
}
```
