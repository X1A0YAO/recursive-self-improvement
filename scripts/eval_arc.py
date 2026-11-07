"""Phase 4.0：ARC-Challenge 评估器（复用 evaluate.py 的评分框架，新增 mc_letter 支持）。

格式化：问题 + 字母标号选项；提取：答案字母。
与主 evaluate.py 同协议：贪心、max_new_tokens、截断计 0、逐题落盘可复现。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate import mcnemar_exact  # noqa: E402
from model_utils import load_model, generate_batch  # noqa: E402

LETTERS = "ABCDE"


def format_arc(ex: dict) -> tuple[str, str, list[str]]:
    q = ex["question"].strip()
    texts = ex["choices"]["text"]
    labels = ex["choices"]["label"]
    # 统一转字母标号（ARC 有 A/B/C/D 与 1/2/3/4 两种标签）
    letter_by_label = {}
    for i, lab in enumerate(labels):
        letter_by_label[str(lab)] = LETTERS[i] if i < len(LETTERS) else str(lab)
    opts = []
    for i, t in enumerate(texts):
        opts.append(f"{LETTERS[i]}. {t}")
    prompt = f"Question: {q}\n" + "\n".join(opts) + "\n\nAnswer with just the letter."
    gold_letter = letter_by_label.get(str(ex["answerKey"]), str(ex["answerKey"]))
    return prompt, gold_letter, texts


def extract_letter(response: str) -> str | None:
    # 优先独立字母行，其次 "answer is X"，再退首个字母字符
    m = re.search(r"^([A-E])\s*$", response.strip(), re.M)
    if m:
        return m.group(1)
    m = re.search(r"[Aa]nswer\s*(?:is|:)?\s*([A-E])\b", response)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-E])\b", response)
    return m.group(1) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="checkpoints/V0")
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--n", type=int, default=0, help="子集大小（0=全部）")
    ap.add_argument("--output", required=True)
    ap.add_argument("--baseline", default="", help="对比基线 json（McNemar）")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=args.split)
    items = list(ds)
    if args.n:
        items = items[: args.n]

    print(f"[arc] {args.split}: {len(items)} examples", flush=True)
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
    prompts = []
    metas = []
    for i, ex in enumerate(items):
        uid = f"arc-{args.split}-{i:04d}"
        if uid in done:
            results.append(done[uid])
            continue
        prompt, gold, texts = format_arc(ex)
        prompts.append((uid, prompt, gold))

    # 批量生成未完成的
    if prompts:
        gens = []
        for uid, prompt, gold in prompts:
            pass  # 逐批组装
        B = args.batch_size
        for b in range(0, len(prompts), B):
            batch = prompts[b : b + B]
            outs = []
            for gen in generate_batch(model, tokenizer, [p[1] for p in batch], max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, enable_thinking=False):
                outs.append(gen)
            for (uid, prompt, gold), gen in zip(batch, outs):
                letter = extract_letter(gen["response"])
                correct = letter == gold
                r = {
                    "uid": uid, "gold": gold, "pred": letter,
                    "score": 1.0 if correct else 0.0,
                    "response": gen["response"][:600],
                    "num_new_tokens": gen.get("num_new_tokens", 0),
                    "truncated": gen.get("num_new_tokens", 0) >= args.max_new_tokens,
                }
                results.append(r)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            acc = sum(r["score"] for r in results) / len(results)
            print(f"  [{args.split}] {len(results)}/{len(items)} acc={acc:.3f}", flush=True)

    acc = sum(r["score"] for r in results) / max(len(results), 1)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": args.model_dir,
        "split": args.split,
        "n": len(results),
        "accuracy": round(acc, 4),
        "config": {"max_new_tokens": args.max_new_tokens, "greedy": True, "batch_size": args.batch_size},
        "results": results,
    }
    if args.baseline and Path(args.baseline).exists():
        bl = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        bm = {r["uid"]: r["score"] for r in bl["results"]}
        cm = {r["uid"]: r["score"] for r in results}
        report["mcnemar_vs_baseline"] = mcnemar_exact(bm, cm)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[arc] accuracy = {acc:.4f} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
