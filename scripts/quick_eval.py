"""快速评测：只跑指定小子集，用于缺陷注入前后的快速对比。

Phase 1 流程中用于：
  1. V0-defect 生成后快速确认行为退化（vs V0 基线）
  2. 各 arm checkpoint 的快速筛查

用 optimization 的 math 子集（对中后层 MLP 扰动最敏感）+ logic，n 可调。

用法：
  python scripts/quick_eval.py --model-dir checkpoints/V0-defect --tag defect --n 16
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluate import load_split, score_one, summarize  # noqa: E402
from model_utils import generate_batch, load_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=16, help="math 题数量；另加前 4 条 logic")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    examples = load_split(Path("benchmarks/optimization/data.jsonl"))
    math_ex = [e for e in examples if e["group"] == "math"][: args.n]
    logic_ex = [e for e in examples if e["group"] == "logic"][:4]
    chosen = math_ex + logic_ex

    print(f"[quick_eval] {args.tag}: {len(chosen)} examples ({len(math_ex)} math + {len(logic_ex)} logic)", flush=True)
    model, tokenizer = load_model(args.model_dir)

    results = []

    def on_result(gen, idx):
        ex = chosen[idx]
        truncated = gen["num_new_tokens"] >= args.max_new_tokens
        s = score_one(ex, gen["response"], truncated)
        s["response"] = gen["response"][:500]
        s["num_new_tokens"] = gen["num_new_tokens"]
        results.append(s)
        n = len(results)
        if n % 4 == 0 or n == len(chosen):
            mean = sum(r["score"] for r in results) / n
            print(f"  [{args.tag}] {n}/{len(chosen)}  mean={mean:.3f}", flush=True)

    generate_batch(
        model, tokenizer, [e["prompt"] for e in chosen],
        max_new_tokens=args.max_new_tokens, enable_thinking=False,
        batch_size=args.batch_size, on_result=on_result,
    )

    summary = summarize(results)
    report = {
        "tag": args.tag,
        "model_dir": args.model_dir,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "results": results,
    }
    out = Path(args.out_dir) / f"quick_{args.tag}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[quick_eval] {args.tag} overall={summary['overall']['mean']:.3f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
