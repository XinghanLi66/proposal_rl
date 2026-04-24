#!/usr/bin/env python3
"""
Generate a per-experiment YAML config by layering key=value overrides on top
of a base YAML.

Usage:
    python scripts/make_config.py \
        --base configs/base.yaml \
        --out  runs/exps/exp01/config.yaml \
        --set  prompt_builder.strategy=top_k_refs \
               rl.finetune_mode=full \
               rl.reward_type=prs \
               rl.output_dir=runs/exps/exp01/rl \
               rl.learning_rate=5e-6 \
               rl.kl_coeff=0.05

Keys use dotted notation for nested dicts.  Values are parsed as YAML scalars
(so 5e-6 → float, true → bool, null → None, quoted strings → str, etc.).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml


def set_nested(d: dict, key: str, value) -> None:
    """Set d[a][b][c] = value given key 'a.b.c'."""
    parts = key.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


def parse_value(s: str):
    """Parse a YAML scalar from a string.

    yaml.safe_load handles most types, but some PyYAML versions return '2e-6'
    as a string instead of float.  Apply numeric coercion as a fallback.
    """
    val = yaml.safe_load(s)
    if isinstance(val, str):
        try:
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                return val
    return val


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="Base YAML config path")
    ap.add_argument("--out",  required=True, help="Output YAML path")
    ap.add_argument("--set",  nargs="*",     default=[], metavar="KEY=VALUE",
                    help="Dotted key=value overrides")
    args = ap.parse_args()

    with open(args.base) as f:
        cfg = yaml.safe_load(f)

    for kv in args.set:
        if "=" not in kv:
            print(f"[make_config] WARNING: skipping malformed override {kv!r} (missing '=')",
                  file=sys.stderr)
            continue
        key, _, val_str = kv.partition("=")
        val = parse_value(val_str)
        set_nested(cfg, key.strip(), val)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"[make_config] Wrote {out_path}")


if __name__ == "__main__":
    main()
