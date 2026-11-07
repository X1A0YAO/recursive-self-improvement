"""Phase 4.1：IFEval 分层划分（seed=20261107，锁定）。

官方 541 题 → Train 181 / Val 180 / Test 180
分层键：instruction_id_list 首 id（25 类）
输出：experiments/phase4/DATASET.md + benchmarks/ifeval_p4/{train,validation,test}/data.jsonl
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
SEED = 20261107


def main() -> None:
    from datasets import load_dataset

    ds = load_dataset("google/IFEval", split="train")
    items = list(ds)
    assert len(items) == 541, len(items)

    # 分层桶
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, ex in enumerate(items):
        key = ex["instruction_id_list"][0] if ex["instruction_id_list"] else "none"
        buckets[key].append(i)

    rng = random.Random(SEED)
    splits: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    # 每桶内部按 181/180/180 近似比例切（大桶按比例，小桶轮转分配）
    for key in sorted(buckets):
        idxs = list(buckets[key])
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = round(n * 181 / 541)
        n_val = round(n * 180 / 541)
        splits["train"].extend(idxs[:n_train])
        splits["validation"].extend(idxs[n_train : n_train + n_val])
        splits["test"].extend(idxs[n_train + n_val :])
    # 校正总数（rounding 误差）
    for name, target in (("train", 181), ("validation", 180), ("test", 180)):
        while len(splits[name]) > target:
            mv = splits[name].pop()
            # 移到最欠的集
            other = min(("train", "validation", "test"), key=lambda k: len(splits[k]) if k != name else 999)
            splits[other].append(mv)
        while len(splits[name]) < target:
            donor = max(("train", "validation", "test"), key=lambda k: len(splits[k]) if k != name else -1)
            splits[name].append(splits[donor].pop())
    assert len(splits["train"]) == 181 and len(splits["validation"]) == 180 and len(splits["test"]) == 180

    # 类型分布统计（分层质量）
    def dist(idxs: list[int]) -> dict[str, int]:
        c: dict[str, int] = defaultdict(int)
        for i in idxs:
            k = items[i]["instruction_id_list"][0] if items[i]["instruction_id_list"] else "none"
            c[k] += 1
        return dict(c)

    d_train, d_val, d_test = dist(splits["train"]), dist(splits["validation"]), dist(splits["test"])

    # 落盘 data.jsonl
    out_root = ROOT / "benchmarks/ifeval_p4"
    for name, idxs in splits.items():
        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "data.jsonl", "w", encoding="utf-8") as f:
            for i in idxs:
                ex = items[i]
                rec = {
                    "uid": f"p4-{name}-{i:04d}",
                    "prompt": ex["prompt"],
                    "meta": {
                        "extract": "ifeval_strict",
                        "instruction_id_list": ex["instruction_id_list"],
                        "kwargs": ex["kwargs"],
                    },
                    "checksum": hashlib.sha256(ex["prompt"].encode()).hexdigest()[:16],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[split] {name}: {len(idxs)} items")

    # DATASET.md
    md = [
        "# Phase 4 数据划分（IFEval 官方集，锁定）",
        "",
        f"- 生成：{SEED=}，分层键=instruction_id_list 首 id",
        f"- Train {len(splits['train'])} / Validation {len(splits['validation'])} / Test {len(splits['test'])}",
        "",
        "| instruction type | train | val | test |",
        "|---|---|---|---|",
    ]
    for k in sorted(d_train):
        md.append(f"| {k} | {d_train.get(k,0)} | {d_val.get(k,0)} | {d_test.get(k,0)} |")
    (ROOT / "experiments/phase4/DATASET.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("[split] DATASET.md written")


if __name__ == "__main__":
    main()
