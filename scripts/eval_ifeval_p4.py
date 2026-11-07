"""Phase 4.1：IFEval 分片评估器（按 benchmarks/ifeval_p4/{split}/data.jsonl）。

与官方检查器同协议：贪心、max_new_tokens=512、batch 8、
截断启发式（修正过 padding bug 的版本）、逐题 jsonl 断点续跑、
--baseline 支持 McNemar。
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
    ap.add_argument("--split", required=True, choices=["train", "validation", "test"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--baseline", default="")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    data_path = ROOT / f"benchmarks/ifeval_p4/{args.split}/data.jsonl"
    items = [json.loads(x) for x in data_path.read_text(encoding="utf-8").strip().splitlines()]
    print(f"[p4eval] {args.split}: {len(items)} items | model={args.model_dir}", flush=True)

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

    results: list[dict] = []
    todo = [ex for ex in items if ex["uid"] not in done]
    for ex in items:
        if ex["uid"] in done:
            results.append(done[ex["uid"]])

    B = args.batch_size
    for b in range(0, len(todo), B):
        batch = todo[b : b + B]
        outs = list(generate_batch(model, tokenizer, [ex["prompt"] for ex in batch], max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, enable_thinking=False))
        for ex, gen in zip(batch, outs):
            example = {"uid": ex["uid"], "prompt": ex["prompt"], "meta": ex["meta"]}
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
                "uid": ex["uid"], "score": score,
                "instruction_ids": ex["meta"]["instruction_id_list"],
                "response": gen["response"][:400],
                "num_new_tokens": n_tok,
                "truncated": truncated,
            }
            results.append(r)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        acc = sum(r["score"] for r in results) / len(results)
        print(f"  [{args.split}] {len(results)}/{len(items)} acc={acc:.3f}", flush=True)

    acc = sum(r["score"] for r in results) / max(len(results), 1)
    # 按指令类型汇总
    by_type: dict[str, list[float]] = {}
    for r in results:
        for iid in r["instruction_ids"][:1]:
            by_type.setdefault(iid, []).append(r["score"])
    type_summary = {k: {"n": len(v), "acc": round(sum(v) / len(v), 3)} for k, v in sorted(by_type.items())}

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": args.model_dir,
        "split": args.split,
        "n": len(results),
        "accuracy": round(acc, 4),
        "by_instruction_type": type_summary,
        "config": {"max_new_tokens": args.max_new_tokens, "greedy": True, "batch_size": args.batch_size},
        "results": results,
    }
    if args.baseline and Path(args.baseline).exists():
        bl = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        bm = {r["uid"]: r["score"] for r in bl["results"]}
        cm = {r["uid"]: r["score"] for r in results}
        report["mcnemar_vs_baseline"] = mcnemar_exact(bm, cm)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[p4eval] {args.split} accuracy = {acc:.4f} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
