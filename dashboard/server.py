#!/usr/bin/env python3
"""
Minimal training dashboard — rule-based log parsing, no LLM calls.

Auto-detects run type (SFT / RL / SFT+RL) from filesystem:
  • runs/exps/EXP_ID/   — ablation experiment runs (one tab each)
  • runs/sft/           — classic SFT run
  • runs/grpo/          — classic GRPO run

Usage (from repo root):
    conda run -n loongflow_ml python dashboard/server.py [--port 8080]
"""
import ast, json, re, time
from datetime import datetime
from pathlib import Path
from flask import Flask, jsonify, send_from_directory

app = Flask(__name__, static_folder="static")
REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"

# ── Parsing ───────────────────────────────────────────────────────────────────

_ANSI = re.compile(r'\x1b\[[0-9;]*[mABCDEFGHJKSTfsu]|\r')
_DICT = re.compile(r'\{[^\{\}]+\}')
_STEP = re.compile(r'\|\s*(\d+)/(\d+)')


def _flt(v):
    try:   return float(v)
    except: return None


def parse_metrics(log_path):
    """Extract metric dicts from a TRL training log (regex + ast.literal_eval)."""
    out = []
    try:
        for raw in log_path.read_bytes().decode(errors='replace').split('\n'):
            line = _ANSI.sub('', raw)
            m = _DICT.search(line)
            if not m: continue
            try:   d = ast.literal_eval(m.group())
            except: continue
            if not isinstance(d, dict) or ('loss' not in d and 'reward' not in d):
                continue
            rec = {k: _flt(v) for k, v in d.items()}
            sm = list(_STEP.finditer(line[:m.start()]))
            if sm:
                rec['_step']  = int(sm[-1].group(1))
                rec['_total'] = int(sm[-1].group(2))
            out.append(rec)
    except Exception:
        pass
    return out


def is_alive(log_path, stale=200):
    return bool(log_path and log_path.exists()
                and (time.time() - log_path.stat().st_mtime) < stale)


def read_json(p):
    try:   return json.loads(Path(p).read_text())
    except: return None


def latest_log(log_dir, prefix):
    """Most recent log matching <prefix>*.log in log_dir, or None."""
    try:
        logs = sorted(Path(log_dir).glob(f"{prefix}*.log"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        return logs[0] if logs else None
    except Exception:
        return None


# ── Stage summary ─────────────────────────────────────────────────────────────

def stage_info(stage_dir, log_path):
    """
    Summarize one training stage from its directory and log file.
      stage_dir : e.g. runs/exps/EXP_ID/rl  or  runs/sft
      log_path  : path to the training .log file, or None
    """
    stage_dir = Path(stage_dir)
    mets = parse_metrics(log_path) if log_path else []
    last = mets[-1] if mets else {}
    final_dir = stage_dir / "final"

    if is_alive(log_path):
        status = "running"
    elif final_dir.exists():
        status = "done"
    elif mets:
        status = "interrupted"
    else:
        status = "pending"

    eta = None
    if last.get('_step') and last.get('_total') and last.get('step_time'):
        rem = (last['_total'] - last['_step']) * last['step_time']
        if rem > 0:
            eta = f"{int(rem)//3600}h{(int(rem)%3600)//60:02d}m"

    checkpoints = []
    if stage_dir.exists():
        for cp in sorted(
            stage_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split('-')[-1]) if p.name.split('-')[-1].isdigit() else 0,
        ):
            checkpoints.append({
                "name":  cp.name,
                "step":  int(cp.name.split('-')[-1]) if cp.name.split('-')[-1].isdigit() else 0,
                "mtime": datetime.fromtimestamp(cp.stat().st_mtime).strftime("%m-%d %H:%M"),
                "eval":  read_json(cp / "eval_results" / "summary.json"),
            })
        if final_dir.exists():
            checkpoints.append({
                "name":     "final ★",
                "step":     None,
                "mtime":    datetime.fromtimestamp(final_dir.stat().st_mtime).strftime("%m-%d %H:%M"),
                "eval":     read_json(final_dir / "eval_results" / "summary.json"),
                "is_final": True,
            })

    return {
        "status":      status,
        "step":        last.get('_step'),
        "total_steps": last.get('_total'),
        "eta":         eta,
        "last_loss":   last.get('loss'),
        "last_reward": last.get('reward'),
        "last_kl":     last.get('kl'),
        "last_prs":    last.get('rewards/reward_prs/mean'),
        "last_fas":    last.get('rewards/reward_fas/mean'),
        "last_fmt":    last.get('rewards/reward_format/mean'),
        "last_update": datetime.fromtimestamp(log_path.stat().st_mtime).strftime("%H:%M:%S")
                       if log_path and log_path.exists() else None,
        "eval":        read_json(final_dir / "eval_results" / "summary.json")
                       if final_dir.exists() else None,
        "checkpoints": checkpoints,
    }


# ── Run discovery ─────────────────────────────────────────────────────────────

def _exp_label(exp_id):
    """exp01_baseline_20260425_193356 → 'exp01_baseline 04/25'"""
    m = re.match(r'^(.+?)_(\d{4})(\d{2})(\d{2})_\d{6}$', exp_id)
    if m:
        return f"{m.group(1)} {m.group(3)}/{m.group(4)}"
    return exp_id


def _read_config(exp_dir):
    try:
        import yaml
        cfg = yaml.safe_load((Path(exp_dir) / "config_base.yaml").read_text())
        return {
            "strategy":      (cfg.get("prompt_builder") or {}).get("strategy"),
            "finetune_mode": (cfg.get("rl") or {}).get("finetune_mode"),
            "reward_type":   (cfg.get("rl") or {}).get("reward_type"),
        }
    except Exception:
        return {}


def discover_runs():
    """Scan filesystem and return run summaries (no curve data)."""
    runs = []

    # ── Ablation runs in runs/exps/EXP_ID/ ───────────────────────────────────
    exps_dir = RUNS / "exps"
    if exps_dir.exists():
        for exp_dir in sorted(exps_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            exp_id  = exp_dir.name
            log_dir = exp_dir / "logs"

            # Detect what stages exist by checking for log files
            has_rl  = latest_log(log_dir, f"rl_{exp_id}") is not None
            has_sft = latest_log(log_dir, f"sft_{exp_id}") is not None
            if not has_rl and not has_sft:
                continue

            run = {
                "id":            exp_id,
                "label":         _exp_label(exp_id),
                "type":          "sft_rl" if (has_sft and has_rl) else ("sft" if has_sft else "rl"),
                "config":        _read_config(exp_dir),
                "hparam_search": read_json(exp_dir / "hparam_search.json"),
            }
            if has_sft:
                run["sft"] = stage_info(exp_dir / "sft", latest_log(log_dir, f"sft_{exp_id}"))
            if has_rl:
                run["rl"]  = stage_info(exp_dir / "rl",  latest_log(log_dir, f"rl_{exp_id}"))
            runs.append(run)

    # ── Classic pipeline runs in runs/sft/ and runs/grpo/ ────────────────────
    log_dir = RUNS / "logs"
    for stage, run_type in [("sft", "sft"), ("grpo", "rl"), ("rl", "rl")]:
        d = RUNS / stage
        if not d.exists():
            continue
        if not list(d.glob("checkpoint-*")) and not (d / "final").exists():
            continue
        log = latest_log(log_dir, stage)
        run = {
            "id":            f"classic_{stage}",
            "label":         f"classic {stage.upper()}",
            "type":          run_type,
            "config":        {},
            "hparam_search": None,
        }
        if run_type == "sft":
            run["sft"] = stage_info(d, log)
        else:
            run["rl"]  = stage_info(d, log)
        runs.append(run)

    return runs


# ── API ───────────────────────────────────────────────────────────────────────

@app.get("/api/runs")
def api_runs():
    """All runs with summaries (no curve data)."""
    return jsonify(discover_runs())


@app.get("/api/runs/<run_id>/curves")
def api_curves(run_id):
    """Training curves for one run, split by stage."""
    # Ablation experiment
    exp_dir = RUNS / "exps" / run_id
    if exp_dir.exists():
        log_dir = exp_dir / "logs"
        rl_log  = latest_log(log_dir, f"rl_{run_id}")
        sft_log = latest_log(log_dir, f"sft_{run_id}")
        return jsonify({
            "rl":  parse_metrics(rl_log)  if rl_log  else [],
            "sft": parse_metrics(sft_log) if sft_log else [],
        })
    # Classic run
    if run_id.startswith("classic_"):
        stage = run_id[len("classic_"):]
        log   = latest_log(RUNS / "logs", stage)
        mets  = parse_metrics(log) if log else []
        return jsonify({"sft": mets, "rl": []} if stage == "sft"
                       else {"rl": mets, "sft": []})
    return jsonify({"error": "not found"}), 404


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    print(f"Dashboard → http://localhost:{args.port}  (runs: {RUNS})")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
