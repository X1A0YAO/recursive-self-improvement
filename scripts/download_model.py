"""下载 Qwen3-4B 原始权重到本地（只读基线）。

用法：
  python scripts/download_model.py --repo Qwen/Qwen3-4B --target models/Qwen3-4B
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-4B")
    ap.add_argument("--target", default="models/Qwen3-4B")
    ap.add_argument(
        "--allow-patterns",
        default="*.safetensors,*.json,*.txt",
        help="逗号分隔；只下 BF16 safetensors 与配置，排除量化/GGUF 变体",
    )
    args = ap.parse_args()

    patterns = [p.strip() for p in args.allow_patterns.split(",") if p.strip()]
    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    path = snapshot_download(
        repo_id=args.repo,
        local_dir=str(target),
        allow_patterns=patterns,
        max_workers=4,
    )
    print(f"[download] complete -> {path}")

    # 校验：列出文件与总大小
    total = 0
    for f in sorted(target.rglob("*")):
        if f.is_file():
            total += f.stat().st_size
            print(f"  {f.relative_to(target)}  {f.stat().st_size/1e6:.1f} MB")
    print(f"[download] total {total/1e9:.2f} GB")


if __name__ == "__main__":
    main()
