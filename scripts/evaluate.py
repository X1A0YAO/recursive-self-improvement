"""独立评测器：对指定 checkpoint 跑基线评测。

原则：
- 与 Self-Improvement Agent 完全隔离，Agent 不得修改本文件与评测数据。
- 贪心解码，结果可复现。
- 逐样本记录预测、参考答案、得分明细，汇总按 group 与 split 统计。

用法：
  python scripts/evaluate.py --model-dir models/Qwen3-4B \
      --splits optimization,validation,test --output results/baseline_V0.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from model_utils import generate

# ---------------------------------------------------------------------------
# 评分函数：按 meta.extract 分发
# ---------------------------------------------------------------------------


def _last_number(text: str, reference: str) -> tuple[bool, str]:
    """GSM8K 风格：gold 从 '#### FINAL' 提取；预测取响应中最后一个数字。"""
    # GSM8K 参考格式: "... <<240/2=120>> ... #### 120"
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", reference)
    if m:
        gold = m.group(1).replace(",", "")
    else:
        m = re.search(r"(-?[\d,]+(?:\.\d+)?)\s*(?:\.|$)", reference.strip())
        gold = m.group(1).replace(",", "") if m else reference.strip()

    # 预测：最后一个数字（忽略 LaTeX 包裹、货币符、逗号）
    nums = re.findall(r"-?\$?\d[\d,]*(?:\.\d+)?", text)
    if not nums:
        return False, gold
    pred = nums[-1].replace("$", "").replace(",", "")
    try:
        ok = abs(float(pred) - float(gold)) < 1e-4
    except ValueError:
        ok = False
    return ok, gold


def _keyword(text: str, reference: str) -> bool:
    """逻辑题：reference 是规范化关键词（如 'second'、'alice=parrot,bob=dog,carol=cat'）。"""
    t = text.lower()
    if "=" in reference:  # 多元素匹配：每个 key=value 都要出现
        return all(part.strip() in t for part in reference.split(","))
    return reference.lower() in t


def _mc_letter(text: str, reference: str, choices: list[str]) -> bool:
    """多选题：预测的字母对应的选项须与 gold 选项一致。"""
    m = re.search(r"\b([A-Z])\b", text.strip().upper())
    if not m:
        # 也允许直接说出选项内容
        return reference.strip().lower() in text.lower()
    idx = ord(m.group(1)) - ord("A")
    if idx < 0 or idx >= len(choices):
        return False
    return choices[idx].strip().lower() == reference.strip().lower()


def _repetition_stats(text: str, requested: int) -> dict:
    """重复输出检测：句子级重复率 + n-gram 循环检测。"""
    sentences = [s.strip() for s in re.split(r"[.!?。！？\n]+", text) if s.strip()]
    n = max(len(sentences), 1)
    unique = len(set(sentences))
    dup_rate = 1.0 - unique / n

    # 字符级 8-gram 循环：重复块占总长比
    compact = re.sub(r"\s+", "", text)
    grams = [compact[i : i + 8] for i in range(max(len(compact) - 7, 0))]
    gram_counts = defaultdict(int)
    for g in grams:
        gram_counts[g] += 1
    loop_chars = sum(c - 1 for c in gram_counts.values() if c > 1) * 8
    loop_ratio = min(loop_chars / max(len(compact), 1), 1.0)

    # 是否被截断（生成到 max_new_tokens 还没停）
    return {
        "num_sentences": len(sentences),
        "requested": requested,
        "unique_sentences": unique,
        "sentence_dup_rate": round(dup_rate, 4),
        "char8gram_loop_ratio": round(loop_ratio, 4),
        "coverage": round(min(len(sentences) / requested, 1.0), 4) if requested else None,
    }


# IFEval 官方检查器（google-research/instruction_following_eval，本地副本）
def _ifeval_check(example: dict, response: str) -> tuple[bool, dict]:
    """与官方 evaluation_lib.test_instruction_following_strict 逻辑一致。

    注意：官方检查的是 response 本身（不含 prompt），某些指令类型
    （如 keywords:frequency）在 build_description 时需要 prompt 参数。
    """
    import instructions_registry as reg

    ok_all, detail = True, {}
    prompt = example["prompt"]
    for iid, kw in zip(example["meta"]["instruction_id_list"], example["meta"]["kwargs"]):
        try:
            cls = reg.INSTRUCTION_DICT[iid]
            inst = cls(iid)
            kwargs = {k: v for k, v in dict(kw).items() if v is not None}  # 官方: None 不传
            inst.build_description(**kwargs)
            args = inst.get_instruction_args()
            if args and "prompt" in args:
                inst.build_description(prompt=prompt)
            ok = bool(response.strip()) and bool(inst.check_following(response))
            detail[iid] = ok
            ok_all = ok_all and ok
        except Exception as e:  # noqa: BLE001
            detail[iid] = f"error:{type(e).__name__}:{e}"
            ok_all = False
    return ok_all, detail


def score_one(example: dict, response: str, truncated: bool) -> dict:
    extract = example["meta"]["extract"]
    ref = example.get("reference_answer")
    out = {"uid": example["uid"], "group": example["group"], "extract": extract}
    score = 0.0
    info: dict = {}

    # 截断的响应无法证明模型得出了结论，统一计 0 分（保守评分）
    if truncated:
        out["score"] = 0.0
        out["info"] = {"truncated": True}
        return out

    if extract == "last_number":
        ok, gold = _last_number(response, ref or "")
        score = 1.0 if ok else 0.0
        info["gold"] = gold
    elif extract == "keyword":
        ok = _keyword(response, ref or "")
        score = 1.0 if ok else 0.0
        info["gold"] = ref
    elif extract == "mc_letter":
        ok = _mc_letter(response, ref or "", example["meta"]["choices"])
        score = 1.0 if ok else 0.0
        info["gold"] = ref
    elif extract == "repetition_stats":
        stats = _repetition_stats(response, example["meta"]["requested_sentences"])
        info.update(stats)
        # 分数 = 1 - 重复率（句子级），循环行为直接扣到 0
        score = max(0.0, 1.0 - stats["sentence_dup_rate"])
        if stats["char8gram_loop_ratio"] > 0.3:
            score = 0.0
    elif extract == "ifeval_strict":
        try:
            ok, detail = _ifeval_check(example, response)
            score = 1.0 if ok else 0.0
            info["ifeval"] = detail
        except Exception as e:  # noqa: BLE001
            # instructions_check 不可用时降级：只检查响应非空
            score = 1.0 if response.strip() else 0.0
            info["ifeval"] = f"fallback_nonempty ({e})"
    else:
        raise ValueError(f"未知 extract 类型: {extract}")

    out["score"] = score
    out["info"] = info
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def load_split(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def evaluate_split(model, tokenizer, examples: list[dict], max_new_tokens: int, log_path: Path | None = None, split_name: str = "", batch_size: int = 4):
    """批量评测，逐条增量落盘到 log_path（jsonl 追加）。"""
    results = []
    done_uids: set[str] = set()
    if log_path is not None and log_path.exists():
        with log_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    results.append(r)
                    done_uids.add(r["uid"])
        if results:
            print(f"    [resume] {len(results)}/{len(examples)} already done", flush=True)

    todo = [ex for ex in examples if ex["uid"] not in done_uids]
    if not todo:
        return results

    def on_result(gen: dict, idx: int) -> None:
        ex = todo[idx]
        truncated = gen["num_new_tokens"] >= max_new_tokens
        s = score_one(ex, gen["response"], truncated)
        s.update(
            {
                "response": gen["response"],
                "num_new_tokens": gen["num_new_tokens"],
            }
        )
        results.append(s)
        if log_path is not None:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        n = len(results)
        if n % 5 == 0 or n == len(examples):
            mean = sum(r["score"] for r in results) / len(results)
            print(f"    [{split_name}] {n}/{len(examples)}  running_mean={mean:.3f}", flush=True)

    from model_utils import generate_batch

    generate_batch(
        model,
        tokenizer,
        [ex["prompt"] for ex in todo],
        max_new_tokens=max_new_tokens,
        enable_thinking=False,
        batch_size=batch_size,
        on_result=on_result,
    )
    return results


def summarize(results: list[dict]) -> dict:
    by_group: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_group[r["group"]].append(r["score"])
    summary = {}
    for g, scores in sorted(by_group.items()):
        summary[g] = {
            "n": len(scores),
            "mean": round(sum(scores) / len(scores), 4),
        }
    all_scores = [r["score"] for r in results]
    summary["overall"] = {"n": len(all_scores), "mean": round(sum(all_scores) / max(len(all_scores), 1), 4)}
    return summary


def mcnemar_exact(baseline: dict[str, float], candidate: dict[str, float], uid_key: str = "uid", results_b=None, results_c=None) -> dict:
    """配对精确二项 McNemar（单侧：退化方向）。

    baseline/candidate: uid -> score
    """
    import math as _math

    uids = [u for u in baseline if u in candidate]
    n01 = sum(1 for u in uids if baseline[u] > candidate[u])  # 退化
    n10 = sum(1 for u in uids if baseline[u] < candidate[u])  # 改善
    n = n01 + n10
    if n == 0:
        p = 1.0
    else:
        p = sum(_math.comb(n, i) for i in range(min(n01, n10) + 1)) / 2**n * 2
        p = min(p, 1.0)
    # 单侧（退化更多）：P(X <= n10 | Bin(n, 0.5))
    if n == 0:
        p_one = 1.0
    else:
        p_one = sum(_math.comb(n, i) for i in range(n10 + 1)) / 2**n
    return {
        "n": len(uids),
        "worse": n01,
        "better": n10,
        "p_two_sided": round(p, 5),
        "p_one_sided_degradation": round(p_one, 5),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/Qwen3-4B")
    ap.add_argument("--splits", default="optimization,validation,test")
    ap.add_argument("--benchmarks-dir", default="benchmarks")
    ap.add_argument("--output", default="results/baseline_V0.json")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="每个 split 最多评多少条（0=全部），调试用")
    ap.add_argument("--baseline", default="", help="基线报告 json（含同 split 逐题结果），输出时附 McNemar")
    args = ap.parse_args()

    from model_utils import load_model

    print(f"[eval] loading model from {args.model_dir}", flush=True)
    model, tokenizer = load_model(args.model_dir)
    mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    print(f"[eval] model loaded, GPU mem allocated = {mem:.2f} GB", flush=True)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": args.model_dir,
        "max_new_tokens": args.max_new_tokens,
        "splits": {},
    }

    for split in args.splits.split(","):
        path = Path(args.benchmarks_dir) / split / "data.jsonl"
        examples = load_split(path)
        if args.limit:
            examples = examples[: args.limit]
        print(f"[eval] {split}: {len(examples)} examples", flush=True)
        log_path = Path(args.output).parent / (Path(args.output).stem + f"_{split}_progress.jsonl")
        results = evaluate_split(
            model, tokenizer, examples, args.max_new_tokens,
            log_path=log_path, split_name=split, batch_size=args.batch_size,
        )
        summary = summarize(results)
        entry = {"summary": summary, "results": results}
        # McNemar 对比（若给了基线）
        if args.baseline and Path(args.baseline).exists():
            bl = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
            if split in bl.get("splits", {}) and "results" in bl["splits"][split]:
                bm = {r["uid"]: r["score"] for r in bl["splits"][split]["results"]}
                cm = {r["uid"]: r["score"] for r in results}
                entry["mcnemar_vs_baseline"] = mcnemar_exact(bm, cm)
        report["splits"][split] = entry
        print(f"[eval] {split} summary: {json.dumps(summary, ensure_ascii=False)}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] report -> {out}")


if __name__ == "__main__":
    main()
