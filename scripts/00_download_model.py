#!/usr/bin/env python3
"""Download Qwen2.5-7B-Instruct from HuggingFace if not already available."""
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

CACHE_DIR = "/newcpfs/lxh/agentic-training/proposal_rl/runs/model_cache"
HF_ID = "Qwen/Qwen2.5-7B-Instruct"
PATH_FILE = "/newcpfs/lxh/agentic-training/proposal_rl/runs/model_path.txt"

# Check known local paths first
CANDIDATES = [
    "/newcpfs/user/sujianghao/model/Qwen/Qwen2.5-7B-Instruct",
    str(Path(CACHE_DIR) / "models--Qwen--Qwen2.5-7B-Instruct" / "snapshots"),
]
for c in CANDIDATES:
    p = Path(c)
    if p.is_dir() and (p / "config.json").exists():
        print(f"Found existing model: {c}")
        Path(PATH_FILE).write_text(c)
        sys.exit(0)
    # Check snapshot subdirs
    if p.is_dir():
        for snap in sorted(p.iterdir()):
            if (snap / "config.json").exists():
                print(f"Found existing snapshot: {snap}")
                Path(PATH_FILE).write_text(str(snap))
                sys.exit(0)

print(f"Downloading {HF_ID} to {CACHE_DIR} ...")
path = snapshot_download(HF_ID, cache_dir=CACHE_DIR)
print(f"Downloaded to: {path}")
Path(PATH_FILE).write_text(path)
