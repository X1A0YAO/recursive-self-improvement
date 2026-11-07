"""Phase 4.0：官方 IFEval 数据集探测评估器。

官方集 541 prompts（google-research/instruction_following_eval），
用本地已有的官方检查器（scripts/instructions*.py）严格评分。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate import _ifeval_check, mcnemar_exact  # noqa: E402
from model_utils import load_model, generate_batch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="checkpoints/V0")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--baseline", default="")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("google/IFEval", split="train")
    items = list(ds)
    if args.n:
        items = items[: args.n]
    print(f"[ifeval] {len(items)} prompts", flush=True)

    model, tokenizer = load_model(str(ROOT / args.model_dir))
    model.eval()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.parent / (out_path.stem + "_progress.jsonl")
    done: dict[str, dict] = {}
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").strip().splitlines():
            r = json.loads(line)
            done[r["uid"]] = r

    results = []
    todo = []
    for i, ex in enumerate(items):
        uid = f"ifeval-{i:04d}"
        if uid in done:
            results.append(done[uid])
            continue
        # 官方字段：prompt / instruction_id_list / kwargs
        todo.append((uid, i, ex))

    B = args.batch_size
    for b in range(0, len(todo), B):
        batch = todo[b : b + B]
        outs = list(generate_batch(model, tokenizer, [items[i]["prompt"] for _, i, _ in batch], max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, enable_thinking=False))
        for (uid, i, ex), gen in zip(batch, outs):
            # 构造与 evaluate.py 相同的 example 结构走官方检查器
            example = {
                "uid": uid,
                "prompt": ex["prompt"],
                "meta": {
                    "extract": "ifeval_strict",
                    "instruction_id_list": ex.get("instruction_id_list", []),
                    "kwargs": ex.get("kwargs", []),
                },
            }
            # 截断判定修正：batch padding 使 num_new_tokens 失真（恒=max）。
            # n_tok < max 必未截断；== max 时用句尾完整性启发式
            n_tok = gen.get("num_new_tokens", 0)
            if n_tok < args.max_new_tokens:
                truncated = False
            else:
                resp = gen["response"].rstrip()
                truncated = not (resp.endswith((".", "!", "?", '"', ")", "]", "*", ">", ":")) or resp[-1:].isalnum())
            if truncated:
                score = 0.0
            else:
                s = _ifeval_check(example, gen["response"])
                score = s[0] if isinstance(s, tuple) else float(s)
            r = {
                "uid": uid, "score": score,
                "instruction_ids": ex.get("instruction_id_list", []),
                "response": gen["response"][:400],
                "num_new_tokens": gen.get("num_new_tokens", 0),
                "truncated": truncated,
            }
            results.append(r)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        acc = sum(r["score"] for r in results) / len(results)
        print(f"  {len(results)}/{len(items)} acc={acc:.3f}", flush=True)

    acc = sum(r["score"] for r in results) / max(len(results), 1)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": args.model_dir,
        "n": len(results),
        "accuracy": round(acc, 4),
        "config": {"max_new_tokens": args.max_new_tokens, "greedy": True},
        "results": results,
    }
    if args.baseline and Path(args.baseline).exists():
        bl = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        bm = {r["uid"]: r["score"] for r in bl["results"]}
        cm = {r["uid"]: r["score"] for r in results}
        report["mcnemar_vs_baseline"] = mcnemar_exact(bm, cm)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ifeval] accuracy = {acc:.4f} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
