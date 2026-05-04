#!/usr/bin/env python3
"""
Read a TRL SFT training log and print the mean loss over the last N steps.
Lower is better (used for SFT hparam selection).
Usage: python extract_sft_metric.py <log_file> [last_n=20]
"""
import ast
import re
import sys

log_file = sys.argv[1]
last_n   = int(sys.argv[2]) if len(sys.argv) > 2 else 20

_ANSI = re.compile(r'\x1b\[[0-9;]*[mABCDEFGHJKSTfsu]|\r')
_DICT = re.compile(r'\{[^\{\}]+\}')

losses = []
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
            if isinstance(d, dict) and 'loss' in d and 'reward' not in d:
                try:
                    losses.append(float(d['loss']))
                except Exception:
                    pass
except Exception:
    pass

if losses:
    tail = losses[-last_n:]
    print(f"{sum(tail)/len(tail):.6f}")
else:
    print("9999.0")   # sentinel: no data = worst possible loss
