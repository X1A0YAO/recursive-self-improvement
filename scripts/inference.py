"""命令行推理入口：快速验证模型加载与生成是否正常。

用法：
  python scripts/inference.py --model-dir models/Qwen3-4B --prompt "1+1=?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from model_utils import generate, load_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/Qwen3-4B")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--thinking", action="store_true", help="启用 Qwen3 思考模式")
    args = ap.parse_args()

    prompt = args.prompt or (
        "Solve: A bakery sells 24 cookies in the morning and twice as many in the "
        "afternoon. If each cookie costs $2, how much money did they make in total? "
        "End your answer with 'Answer: <number>'."
    )

    print(f"[infer] loading {args.model_dir}", flush=True)
    model, tokenizer = load_model(args.model_dir)
    if torch.cuda.is_available():
        print(
            f"[infer] GPU mem allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB"
        )

    out = generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=args.thinking if args.thinking else False,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
