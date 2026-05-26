#!/usr/bin/env python3
"""
Read a verl (or TRL) SFT training log and print the mean loss over the last N steps.
Lower is better (used for SFT hparam selection).

verl SFT logs metrics as JSON lines:
  {"train/loss": 1.23, "train/global_step": 10, ...}

Usage: python extract_sft_metric.py <log_file> [last_n=20]
"""
import ast
import json
import re
import sys

log_file = sys.argv[1]
last_n   = int(sys.argv[2]) if len(sys.argv) > 2 else 20

_ANSI = re.compile(r'\x1b\[[0-9;]*[mABCDEFGHJKSTfsu]|\r')

losses = []
try:
    with open(log_file, 'rb') as f:
        for raw in f.read().decode(errors='replace').split('\n'):
            line = _ANSI.sub('', raw).strip()
            if not line:
                continue

            # Try JSON (verl metric lines)
            if line.startswith('{'):
                try:
                    d = json.loads(line)
                    for key in ("train/loss", "loss", "train_loss"):
                        if key in d:
                            losses.append(float(d[key]))
                            break
                    continue
                except Exception:
                    pass

            # Fallback: dict literal
            m = re.search(r'\{[^\{\}]+\}', line)
            if m:
                try:
                    d = ast.literal_eval(m.group())
                    if isinstance(d, dict) and 'reward' not in d:
                        val = d.get('train/loss') or d.get('loss') or d.get('train_loss')
                        if val is not None:
                            losses.append(float(val))
                except Exception:
                    pass
except Exception:
    pass

if losses:
    tail = losses[-last_n:]
    print(f"{sum(tail)/len(tail):.6f}")
else:
    print("9999.0")   # sentinel: no data = worst possible loss
