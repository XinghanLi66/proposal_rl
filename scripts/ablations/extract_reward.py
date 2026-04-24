#!/usr/bin/env python3
"""
Read a TRL training log and print the mean reward over the last N steps.
Usage: python extract_reward.py <log_file> [last_n=10]
"""
import ast
import re
import sys

log_file = sys.argv[1]
last_n   = int(sys.argv[2]) if len(sys.argv) > 2 else 10

_ANSI = re.compile(r'\x1b\[[0-9;]*[mABCDEFGHJKSTfsu]|\r')
_DICT = re.compile(r'\{[^\{\}]+\}')

rewards = []
try:
    with open(log_file, 'rb') as f:
        for raw in f.read().decode(errors='replace').split('\n'):
            line = _ANSI.sub('', raw)
            m = _DICT.search(line)
            if not m:
                continue
            try:
                d = ast.literal_eval(m.group())
            except Exception:
                continue
            if isinstance(d, dict) and 'reward' in d:
                try:
                    rewards.append(float(d['reward']))
                except Exception:
                    pass
except Exception:
    pass

if rewards:
    tail = rewards[-last_n:]
    print(f"{sum(tail)/len(tail):.6f}")
else:
    print("0.0")
