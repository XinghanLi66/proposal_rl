#!/usr/bin/env python3
"""
Read a verl (or TRL) training log and print the mean reward over the last N steps.

Patterns searched (in order):
  {"critic/rewards/mean": 0.42, ...}           — verl JSON metric lines
  {"reward": 0.42, ...}                         — TRL-style dicts (fallback)
  step:N - ... - critic/rewards/mean:0.42 -     — verl console text format

Usage: python extract_reward.py <log_file> [last_n=10]
"""
import ast
import json
import re
import sys

log_file = sys.argv[1]
last_n   = int(sys.argv[2]) if len(sys.argv) > 2 else 10

_ANSI = re.compile(r'\x1b\[[0-9;]*[mABCDEFGHJKSTfsu]|\r')

rewards = []
try:
    with open(log_file, 'rb') as f:
        for raw in f.read().decode(errors='replace').split('\n'):
            line = _ANSI.sub('', raw).strip()
            if not line:
                continue

            # Try JSON parse first (verl metric lines)
            if line.startswith('{'):
                try:
                    d = json.loads(line)
                    # verl reward keys
                    for key in ("critic/rewards/mean", "reward/reward_score/mean",
                                "actor/reward_score/mean", "reward"):
                        if key in d:
                            rewards.append(float(d[key]))
                            break
                    continue
                except Exception:
                    pass

            # verl console text format: "step:N - ... - critic/rewards/mean:0.42 - ..."
            # These lines contain "step:" and key:value pairs separated by " - "
            if 'step:' in line and 'critic/rewards/mean:' in line:
                m = re.search(r'critic/rewards/mean:([-\d.eE+]+)', line)
                if m:
                    try:
                        rewards.append(float(m.group(1)))
                    except Exception:
                        pass
                continue

            # Fallback: dict literal anywhere on line
            m = re.search(r'\{[^\{\}]+\}', line)
            if m:
                try:
                    d = ast.literal_eval(m.group())
                    if isinstance(d, dict):
                        for key in ("critic/rewards/mean", "reward/reward_score/mean",
                                    "actor/reward_score/mean", "reward"):
                            if key in d:
                                rewards.append(float(d[key]))
                                break
                except Exception:
                    pass
except Exception:
    pass

if rewards:
    tail = rewards[-last_n:]
    print(f"{sum(tail)/len(tail):.6f}")
else:
    print("0.0")
