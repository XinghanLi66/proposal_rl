#!/usr/bin/env python3
"""
Training dashboard for proposal_rl.

Supports two views:
  • Pipeline view  — the classic SFT → GRPO pipeline (tab "pipeline")
  • Experiment tabs — each ablation run registers itself and gets its own tab

Usage (from repo root):
    conda run -n loongflow_ml python dashboard/server.py
    # open http://localhost:8080
"""

from __future__ import annotations

import ast
import json
import os
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static")

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR  = REPO_ROOT / "runs"

# ── Parsing helpers ────────────────────────────────────────────────────────────

_ANSI = re.compile(r'\x1b\[[0-9;]*[mABCDEFGHJKSTfsu]|\r')
_DICT = re.compile(r'\{[^\{\}]+\}')
_STEP = re.compile(r'\|\s*(\d+)/(\d+)')


def _strip(s: str) -> str:
    return _ANSI.sub('', s)


def _flt(v: Any) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def parse_metrics(log_path: Path) -> list[dict]:
    out: list[dict] = []
    try:
        raw_bytes = log_path.read_bytes()
        for raw in raw_bytes.decode(errors='replace').split('\n'):
            line = _strip(raw)
            m = _DICT.search(line)
            if not m:
                continue
            try:
                d = ast.literal_eval(m.group())
            except Exception:
                continue
            if not isinstance(d, dict) or ('loss' not in d and 'reward' not in d):
                continue
            rec = {k: _flt(v) for k, v in d.items()}
            pre = line[:m.start()]
            candidates = list(_STEP.finditer(pre))
            sm = candidates[-1] if candidates else _STEP.search(line)
            if sm:
                rec['_step']  = int(sm.group(1))
                rec['_total'] = int(sm.group(2))
            out.append(rec)
    except Exception:
        pass
    return out


def latest_log(log_prefix: str, runs_dir: Path | None = None) -> Path | None:
    d = (runs_dir or RUNS_DIR) / "logs"
    logs = sorted(
        d.glob(f"{log_prefix}_*.log"),
        key=lambda p: p.stat().st_mtime, reverse=True
    )
    return logs[0] if logs else None


def is_alive(log: Path | None, stale: int = 200) -> bool:
    return bool(log and log.exists() and (time.time() - log.stat().st_mtime) < stale)


def read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def read_jsonl(p: Path, limit: int = 0) -> list[dict]:
    out: list[dict] = []
    try:
        with open(p) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
                if limit and len(out) >= limit:
                    break
    except Exception:
        pass
    return out


# ── Experiment registry ────────────────────────────────────────────────────────

def _registry_path() -> Path:
    return RUNS_DIR / "logs" / "experiments.json"


def _load_registry() -> list[dict]:
    p = _registry_path()
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


def _save_registry(exps: list[dict]) -> None:
    p = _registry_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(exps, indent=2))


# ── Stage summary helper (shared by pipeline and experiment views) ─────────────

def _stage_summary(stage_label: str, stage_dir: Path, log_prefix: str,
                   runs_dir: Path | None = None) -> dict:
    log  = latest_log(log_prefix, runs_dir)
    mets = parse_metrics(log) if log else []
    last = mets[-1] if mets else {}
    eval_res = read_json(stage_dir / "final" / "eval_results" / "summary.json") \
        if stage_dir.exists() else None

    eta = None
    if last.get('_step') and last.get('_total'):
        step, total = last['_step'], last['_total']
        step_time = last.get('step_time')
        if step_time and step < total:
            eta_s = int((total - step) * step_time)
            h, m = divmod(eta_s, 3600)
            m //= 60
            eta = f"{h}h{m:02d}m"

    return {
        "status":      "running" if is_alive(log) else (
                       "done" if (stage_dir / "final").exists() else "pending"),
        "log":         str(log) if log else None,
        "step":        last.get('_step'),
        "total_steps": last.get('_total'),
        "epoch":       last.get('epoch'),
        "eta":         eta,
        "last_loss":   last.get('loss'),
        "last_reward": last.get('reward'),
        "last_kl":     last.get('kl'),
        "last_prs":    last.get('rewards/reward_prs/mean'),
        "last_fas":    last.get('rewards/reward_fas/mean'),
        "last_fmt":    last.get('rewards/reward_format/mean'),
        "last_update": datetime.fromtimestamp(log.stat().st_mtime).strftime("%H:%M:%S")
                       if log and log.exists() else None,
        "eval":        eval_res,
    }


def _rollouts(log_prefix: str, stage_dir: Path,
              runs_dir: Path | None = None) -> dict:
    rdir = (runs_dir or RUNS_DIR) / "logs"
    # For experiment views, runs_dir is exp_dir which has logs/rollouts_*.jsonl
    # For pipeline views, log_prefix contains the stage name
    rf = sorted(
        list(rdir.glob(f"rollouts_{log_prefix}_*.jsonl")) or
        list(rdir.glob("rollouts_*.jsonl")),
        key=lambda p: p.stat().st_mtime, reverse=True
    )
    rollouts: list[dict] = []
    if rf:
        rollouts = read_jsonl(rf[0])
    rollouts.sort(key=lambda r: float(r.get("score") or 0), reverse=True)
    return {
        "total":    len(rollouts),
        "positive": rollouts[:5],
        "negative": rollouts[-5:][::-1] if len(rollouts) >= 5 else [],
    }


def _failures(log_prefix: str, stage_dir: Path,
              runs_dir: Path | None = None) -> list[dict]:
    log  = latest_log(log_prefix, runs_dir)
    mets = parse_metrics(log) if log else []
    failures: list[dict] = []

    if mets:
        last   = mets[-1]
        recent = mets[-15:]

        clip = last.get("completions/clipped_ratio") or 0
        if clip > 0.3:
            failures.append({"type": "truncation", "sev": "high",
                             "msg": f"clipped_ratio={clip:.2f} at step {last.get('_step','?')}"})

        rewards = [m["reward"] for m in recent if m.get("reward") is not None]
        if len(rewards) >= 5 and max(rewards) - min(rewards) < 1e-5:
            failures.append({"type": "reward_collapse", "sev": "critical",
                             "msg": f"reward variance≈0 over {len(rewards)} steps, val={rewards[-1]:.4f}"})

        prs_vals = [m.get("rewards/reward_prs/mean") for m in recent
                    if m.get("rewards/reward_prs/mean") is not None]
        if len(prs_vals) >= 6 and prs_vals[-1] < prs_vals[0] - 0.04:
            failures.append({"type": "prs_declining", "sev": "warn",
                             "msg": f"PRS {prs_vals[0]:.3f}→{prs_vals[-1]:.3f} over last {len(prs_vals)} steps"})

        fas_vals = [m.get("rewards/reward_fas/mean") for m in recent
                    if m.get("rewards/reward_fas/mean") is not None]
        if len(fas_vals) >= 6 and fas_vals[-1] < fas_vals[0] - 0.04:
            failures.append({"type": "fas_declining", "sev": "warn",
                             "msg": f"FAS {fas_vals[0]:.3f}→{fas_vals[-1]:.3f} over last {len(fas_vals)} steps"})

        fmt = last.get("rewards/reward_format/mean") or 1
        if fmt < 0.5:
            failures.append({"type": "format_degraded", "sev": "warn",
                             "msg": f"format_reward={fmt:.3f} (expected >0.9)"})

    rdir = (runs_dir or RUNS_DIR) / "logs"
    rf = sorted(
        list(rdir.glob(f"rollouts_{log_prefix}_*.jsonl")) or
        list(rdir.glob("rollouts_*.jsonl")),
        key=lambda p: p.stat().st_mtime, reverse=True
    )
    if rf:
        for r in read_jsonl(rf[0])[-60:]:
            out  = r.get("output", "") or ""
            tags = []
            if not out.strip():                        tags.append("empty_output")
            elif len(out.split()) < 30:               tags.append("very_short")
            if "<proposal>" not in out:               tags.append("no_open_tag")
            if "</proposal>" not in out:              tags.append("no_close_tag")
            words = out.lower().split()
            if len(words) > 20:
                bg = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
                top = Counter(bg).most_common(1)
                if top and top[0][1] > 6:
                    tags.append(f"repetition×{top[0][1]}")
            if tags:
                failures.append({"type": "rollout", "tags": tags,
                                 "step": r.get("step"), "score": r.get("score"),
                                 "preview": out[:200]})
    return failures


# ── Pipeline API (classic SFT + GRPO) ─────────────────────────────────────────

@app.get("/api/summary")
def api_summary():
    res: dict = {}
    for stage in ("sft", "grpo"):
        sd = RUNS_DIR / stage
        res[stage] = _stage_summary(stage, sd, stage)

    ds: dict = {}
    for split in ("train", "train_cot", "val", "test"):
        p = RUNS_DIR / "dataset" / f"{split}.jsonl"
        if p.exists():
            ds[split] = sum(1 for _ in open(p))
    res["dataset"] = ds

    try:
        import yaml
        cfg = yaml.safe_load((REPO_ROOT / "configs" / "base.yaml").read_text())
        res["config"] = {
            "model":        Path(cfg.get("model_name_or_path", "?")).name or "?",
            "train_months": cfg.get("train_months", []),
            "val_months":   cfg.get("val_months",   []),
            "test_months":  cfg.get("test_months",  []),
            "rl": {
                "max_completion_length": cfg.get("rl", {}).get("max_completion_length"),
                "num_generations":       cfg.get("rl", {}).get("num_generations"),
                "kl_coeff":              cfg.get("rl", {}).get("kl_coeff"),
                "reward_type":           cfg.get("rl", {}).get("reward_type"),
            },
        }
    except Exception:
        pass

    return jsonify(res)


@app.get("/api/curves/<stage>")
def api_curves(stage):
    log = latest_log(stage)
    if not log:
        return jsonify({"error": "no log", "metrics": []})
    return jsonify({"stage": stage, "metrics": parse_metrics(log)})


@app.get("/api/checkpoints/<stage>")
def api_checkpoints(stage):
    sd = RUNS_DIR / stage
    cps = []
    if sd.exists():
        for cp in sorted(sd.glob("checkpoint-*"),
                         key=lambda p: int(p.name.split('-')[-1])
                         if p.name.split('-')[-1].isdigit() else 0):
            step = int(cp.name.split('-')[-1]) if cp.name.split('-')[-1].isdigit() else 0
            cps.append({
                "name":  cp.name,
                "step":  step,
                "mtime": datetime.fromtimestamp(cp.stat().st_mtime).strftime("%m-%d %H:%M"),
                "eval":  read_json(cp / "eval_results" / "summary.json"),
            })
        fin = sd / "final"
        if fin.exists():
            cps.append({
                "name":     "final ★",
                "step":     "final",
                "mtime":    datetime.fromtimestamp(fin.stat().st_mtime).strftime("%m-%d %H:%M"),
                "eval":     read_json(fin / "eval_results" / "summary.json"),
                "is_final": True,
            })
    return jsonify({"stage": stage, "checkpoints": cps})


@app.get("/api/rollouts/<stage>")
def api_rollouts(stage):
    sd = RUNS_DIR / stage
    data = _rollouts(stage, sd)
    if not data["total"] and stage == "sft":
        for ex in read_jsonl(RUNS_DIR / "sft" / "final" / "eval_results" / "per_example.jsonl"):
            fas = (ex.get("recall_at_k", 0) + ex.get("mean_sim", 0)) / 2
            data["positive" if fas >= 0.5 else "negative"].append({
                "step":   "eval",
                "prompt": ex.get("arxiv_id", ""),
                "output": ex.get("proposal", ""),
                "score":  fas,
                "source": "eval",
            })
        data["total"] = len(data["positive"]) + len(data["negative"])
    return jsonify({"stage": stage, **data})


@app.get("/api/failures/<stage>")
def api_failures(stage):
    sd = RUNS_DIR / stage
    failures = _failures(stage, sd)
    return jsonify({"stage": stage, "failures": failures})


@app.get("/api/datastats")
def api_datastats():
    stats: dict = {}
    for split in ("train", "train_cot", "val", "test"):
        p = RUNS_DIR / "dataset" / f"{split}.jsonl"
        if not p.exists():
            continue
        total = has_cot = leakage = fmt_ok = 0
        wlen: list[int] = []
        for rec in read_jsonl(p):
            total += 1
            prop = rec.get("cot_proposal") or rec.get("target_proposal") or ""
            if prop:
                has_cot += 1
                wlen.append(len(prop.split()))
                if "<proposal>" in prop and "</proposal>" in prop:
                    fmt_ok += 1
            if rec.get("leakage_flagged"):
                leakage += 1
        wlen.sort()
        n = len(wlen)
        stats[split] = {
            "total":         total,
            "has_cot":       has_cot,
            "leakage":       leakage,
            "format_valid":  fmt_ok,
            "wlen_p50":      wlen[n // 2] if n else None,
            "wlen_p95":      wlen[int(n * .95)] if n else None,
            "wlen_max":      wlen[-1] if n else None,
        }
    return jsonify(stats)


# ── Experiment registry API ────────────────────────────────────────────────────

@app.get("/api/experiments")
def api_list_experiments():
    return jsonify(_load_registry())


@app.post("/api/experiments")
def api_register_experiment():
    """Register a new ablation run.

    Body (JSON):
        {
          "exp_id":   "exp01_baseline_20260422_143022",
          "name":     "baseline (top-k / full-ft / embed-PRS)",
          "runs_dir": "/abs/path/to/runs/exps/exp01_baseline_20260422_143022",
          "config":   { ... optional config snapshot ... }
        }
    """
    data = request.get_json(force=True, silent=True) or {}
    required = ["exp_id", "name", "runs_dir"]
    missing = [k for k in required if not data.get(k)]
    if missing:
        return jsonify({"error": f"missing fields: {missing}"}), 400

    exps = _load_registry()
    ids = {e["exp_id"] for e in exps}
    if data["exp_id"] in ids:
        return jsonify({"status": "already_registered", "exp_id": data["exp_id"]}), 200

    entry = {
        "exp_id":     data["exp_id"],
        "name":       data["name"],
        "runs_dir":   data["runs_dir"],
        "config":     data.get("config", {}),
        "registered": datetime.utcnow().isoformat(),
    }
    exps.append(entry)
    _save_registry(exps)
    return jsonify({"status": "registered", "exp_id": data["exp_id"]}), 201


@app.get("/api/experiment/<exp_id>/summary")
def api_exp_summary(exp_id):
    exps = {e["exp_id"]: e for e in _load_registry()}
    if exp_id not in exps:
        return jsonify({"error": "unknown experiment"}), 404
    exp = exps[exp_id]
    exp_runs_dir = Path(exp["runs_dir"])
    rl_dir = exp_runs_dir / "rl"
    summary = _stage_summary("rl", rl_dir, f"rl_{exp_id}", exp_runs_dir)
    return jsonify({
        "exp_id": exp_id,
        "name":   exp["name"],
        "config": exp.get("config", {}),
        "rl":     summary,
    })


@app.get("/api/experiment/<exp_id>/curves")
def api_exp_curves(exp_id):
    exps = {e["exp_id"]: e for e in _load_registry()}
    if exp_id not in exps:
        return jsonify({"error": "unknown experiment"}), 404
    exp_runs_dir = Path(exps[exp_id]["runs_dir"])
    log = latest_log(f"rl_{exp_id}", exp_runs_dir)
    if not log:
        return jsonify({"metrics": []})
    return jsonify({"metrics": parse_metrics(log)})


@app.get("/api/experiment/<exp_id>/checkpoints")
def api_exp_checkpoints(exp_id):
    exps = {e["exp_id"]: e for e in _load_registry()}
    if exp_id not in exps:
        return jsonify({"error": "unknown experiment"}), 404
    exp_runs_dir = Path(exps[exp_id]["runs_dir"])
    rl_dir = exp_runs_dir / "rl"
    cps = []
    if rl_dir.exists():
        for cp in sorted(rl_dir.glob("checkpoint-*"),
                         key=lambda p: int(p.name.split('-')[-1])
                         if p.name.split('-')[-1].isdigit() else 0):
            step = int(cp.name.split('-')[-1]) if cp.name.split('-')[-1].isdigit() else 0
            cps.append({
                "name":  cp.name,
                "step":  step,
                "mtime": datetime.fromtimestamp(cp.stat().st_mtime).strftime("%m-%d %H:%M"),
                "eval":  read_json(cp / "eval_results" / "summary.json"),
            })
        fin = rl_dir / "final"
        if fin.exists():
            cps.append({
                "name":  "final ★",
                "step":  "final",
                "mtime": datetime.fromtimestamp(fin.stat().st_mtime).strftime("%m-%d %H:%M"),
                "eval":  read_json(fin / "eval_results" / "summary.json"),
                "is_final": True,
            })
    return jsonify({"checkpoints": cps})


@app.get("/api/experiment/<exp_id>/rollouts")
def api_exp_rollouts(exp_id):
    exps = {e["exp_id"]: e for e in _load_registry()}
    if exp_id not in exps:
        return jsonify({"error": "unknown experiment"}), 404
    exp_runs_dir = Path(exps[exp_id]["runs_dir"])
    rl_dir = exp_runs_dir / "rl"
    data = _rollouts(f"rl_{exp_id}", rl_dir, exp_runs_dir)
    return jsonify(data)


@app.get("/api/experiment/<exp_id>/failures")
def api_exp_failures(exp_id):
    exps = {e["exp_id"]: e for e in _load_registry()}
    if exp_id not in exps:
        return jsonify({"error": "unknown experiment"}), 404
    exp_runs_dir = Path(exps[exp_id]["runs_dir"])
    rl_dir = exp_runs_dir / "rl"
    failures = _failures(f"rl_{exp_id}", rl_dir, exp_runs_dir)
    return jsonify({"failures": failures})


# ── Hparam search result endpoint ─────────────────────────────────────────────

@app.get("/api/experiment/<exp_id>/hparam_search")
def api_exp_hparam(exp_id):
    """Return hparam search results for this experiment (mini-runs)."""
    exps = {e["exp_id"]: e for e in _load_registry()}
    if exp_id not in exps:
        return jsonify({"error": "unknown experiment"}), 404
    exp_runs_dir = Path(exps[exp_id]["runs_dir"])
    results_file = exp_runs_dir / "hparam_search.json"
    if not results_file.exists():
        return jsonify({"results": [], "best": None})
    try:
        data = json.loads(results_file.read_text())
        return jsonify(data)
    except Exception:
        return jsonify({"results": [], "best": None})


# ── Static files ───────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return send_from_directory("static", "index.html")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port",     type=int, default=8080)
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    args = ap.parse_args()
    RUNS_DIR = Path(args.runs_dir)
    print(f"Dashboard → http://localhost:{args.port}  (runs: {RUNS_DIR.resolve()})")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
