"""构建基线评测数据集：optimization / validation / test 三组严格隔离。

数据源（走 Hugging Face datasets）：
  - math        : openai/gsm8k (main, test split)
  - instruction : google/IFEval (train split)
  - logic       : 本地人工校验题库（确定性答案）
  - repetition  : 本地模板（要求写 N 句不重复句子，统计重复率）
  - general     : truthfulqa/generic（多选题，通用语言能力）

用法：
  .venv\Scripts\python.exe scripts\build_benchmarks.py --output-dir benchmarks
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

SEED = 20261104

# 每组每个类别的样本数
PER_SPLIT = {
    "optimization": {"math": 25, "instruction": 10, "logic": 8, "repetition": 8, "general": 12},
    "validation": {"math": 25, "instruction": 10, "logic": 6, "repetition": 6, "general": 12},
    "test": {"math": 40, "instruction": 12, "logic": 6, "repetition": 6, "general": 15},
}

LOGIC_TEMPLATE = (
    "Solve this logic puzzle. Think step by step, then give the final answer "
    "as 'Answer: X' on the last line.\n\n{body}"
)

LOGIC_ITEMS = [
    ("Alice, Bob and Carol each have a different pet: cat, dog, parrot. Alice is allergic to fur. Bob's pet can't fly and isn't a cat. What pet does each person have?", "alice=parrot,bob=dog,carol=cat"),
    ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost (in dollars)?", "0.05"),
    ("If all Blorps are Zarps and some Zarps are Quibs, is it necessarily true that some Blorps are Quibs? Answer Yes or No and explain.", "no"),
    ("Five machines make five widgets in five minutes. How many minutes do 100 machines take to make 100 widgets?", "5"),
    ("In a race, Anne finished ahead of Beth, and Carol finished ahead of Anne. Who won the race?", "carol"),
    ("A farmer has 17 sheep. All but 9 die. How many sheep are left?", "9"),
    ("Every student who studies passes. Some students don't study. Can we conclude that some students don't pass? Answer Yes or No and explain.", "no"),
    ("Mary's father has five daughters: Nana, Nene, Nini, Nono, and what is the fifth daughter named?", "mary"),
    ("If you are running a race and pass the person in second place, what place are you in?", "second"),
    ("Two coins add up to 30 cents and one of them is not a quarter. What are the two coins? Give both names.", "quarter_and_nickel"),
    ("John is taller than Mike. Mike is taller than Sue. Who is tallest: John, Mike or Sue?", "john"),
    ("A lily pad doubles in size every day. It covers the lake on day 48. On what day was the lake half covered?", "47"),
    ("John's father has a sister named Lily. What is Lily to John?", "aunt"),
    ("If it takes 8 people 10 hours to build a wall, how many hours do 4 people take at the same speed?", "20"),
    ("A man looks at a photo and says: 'Brothers and sisters I have none, but that man's father is my father's son.' Who is in the photo?", "his_son"),
    ("Which is larger: half of a quarter, or a quarter of a half?", "equal"),
    ("Sam eats 2 apples on day 1 and doubles his intake every day. How many apples does he eat on day 5?", "32"),
    ("A shirt costs $100. The price is discounted 20%, then rises 20%. What is the final price in dollars?", "96"),
    ("Two fathers and two sons went fishing. Each caught one fish but they brought home only 3 fish in total. Explain in one short phrase how this is possible.", "three_generations"),
    ("If A>B, B>C, and C>D, is A necessarily greater than D? Answer Yes or No.", "yes"),
    ("A plane crashes exactly on the border of two countries. Where do they bury the survivors? Answer in one word.", "nowhere"),
    ("Some months have 31 days, some have 30. How many months have 28 days?", "12"),
    ("You have one match and enter a room with a candle, an oil lamp and a fireplace. What do you light first?", "match"),
    ("If yesterday's tomorrow was Friday, what day is today?", "friday"),
]

REPETITION_TEMPLATE = (
    "Write {n} distinct sentences about the topic below. "
    "Every sentence must add new information and no two sentences may be near-duplicates.\n"
    "Topic: {topic}"
)

REPETITION_TOPICS = [
    "the water cycle", "how coffee is made", "the history of paper",
    "why the sky is blue", "how vaccines work", "the life of a star",
    "photosynthesis in plants", "the invention of the telephone",
    "how bridges stay up", "why leaves change color", "ocean tides",
    "the printing press", "magnets", "volcanoes", "ocean water",
    "how bicycles work", "rainbows", "electricity in homes",
    "the pyramids of Egypt", "why ships float", "bees and flowers",
    "the seasons of the year", "deserts", "penguins", "the moon",
    "glaciers", "the game of chess", "maple syrup", "lighthouses",
    "wind", "salt", "clocks", "honey", "camels", "trains",
    "submarines", "kites", "elevators", "echoes", "shadows",
]

REPETITION_N = [3, 4, 5, 6, 8]


def build_math(rng: random.Random) -> dict[str, list[dict]]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    need = sum(PER_SPLIT[s]["math"] for s in PER_SPLIT)
    assert len(ds) >= need, f"GSM8K test 只有 {len(ds)} 条，不够 {need}"
    chosen = idx[:need]
    cursor = 0
    out: dict[str, list[dict]] = {s: [] for s in PER_SPLIT}
    for split in PER_SPLIT:
        n = PER_SPLIT[split]["math"]
        for r in ds.select(chosen[cursor : cursor + n]):
            out[split].append(
                {
                    "group": "math",
                    "source": "openai/gsm8k:main:test",
                    "prompt": r["question"].strip(),
                    "reference_answer": r["answer"].strip(),
                    "meta": {"extract": "last_number"},
                }
            )
        cursor += n
    return out


def build_instruction(rng: random.Random) -> dict[str, list[dict]]:
    from datasets import load_dataset

    ds = load_dataset("google/IFEval", split="train")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    need = sum(PER_SPLIT[s]["instruction"] for s in PER_SPLIT)
    assert len(ds) >= need
    chosen = idx[:need]
    cursor = 0
    out: dict[str, list[dict]] = {s: [] for s in PER_SPLIT}
    for split in PER_SPLIT:
        n = PER_SPLIT[split]["instruction"]
        for r in ds.select(chosen[cursor : cursor + n]):
            out[split].append(
                {
                    "group": "instruction",
                    "source": "google/IFEval:train",
                    "prompt": r["prompt"].strip(),
                    "reference_answer": None,
                    "meta": {
                        "extract": "ifeval_strict",
                        "instruction_id_list": list(r["instruction_id_list"]),
                        "kwargs": [dict(k) for k in r["kwargs"]],
                    },
                }
            )
        cursor += n
    return out


def build_logic(rng: random.Random) -> dict[str, list[dict]]:
    items = LOGIC_ITEMS.copy()
    rng.shuffle(items)
    need = sum(PER_SPLIT[s]["logic"] for s in PER_SPLIT)
    assert len(items) >= need, f"逻辑题库只有 {len(items)} 题，不够 {need}"
    chosen = items[:need]
    cursor = 0
    out: dict[str, list[dict]] = {s: [] for s in PER_SPLIT}
    for split in PER_SPLIT:
        n = PER_SPLIT[split]["logic"]
        for body, ans in chosen[cursor : cursor + n]:
            out[split].append(
                {
                    "group": "logic",
                    "source": "local:logic_v1",
                    "prompt": LOGIC_TEMPLATE.format(body=body),
                    "reference_answer": ans,
                    "meta": {"extract": "keyword"},
                }
            )
        cursor += n
    return out


def build_repetition(rng: random.Random) -> dict[str, list[dict]]:
    topics = REPETITION_TOPICS.copy()
    rng.shuffle(topics)
    need = sum(PER_SPLIT[s]["repetition"] for s in PER_SPLIT)
    assert len(topics) >= need
    chosen = topics[:need]
    cursor = 0
    out: dict[str, list[dict]] = {s: [] for s in PER_SPLIT}
    for split in PER_SPLIT:
        n = PER_SPLIT[split]["repetition"]
        for _ in range(n):
            topic = chosen[cursor]
            n_sent = rng.choice(REPETITION_N)
            cursor += 1
            out[split].append(
                {
                    "group": "repetition",
                    "source": "local:repetition_v1",
                    "prompt": REPETITION_TEMPLATE.format(n=n_sent, topic=topic),
                    "reference_answer": None,
                    "meta": {"extract": "repetition_stats", "requested_sentences": n_sent},
                }
            )
    return out


def build_general(rng: random.Random) -> dict[str, list[dict]]:
    from datasets import load_dataset

    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    need = sum(PER_SPLIT[s]["general"] for s in PER_SPLIT)
    assert len(ds) >= need
    chosen = idx[:need]
    cursor = 0
    out: dict[str, list[dict]] = {s: [] for s in PER_SPLIT}
    for split in PER_SPLIT:
        n = PER_SPLIT[split]["general"]
        for r in ds.select(chosen[cursor : cursor + n]):
            choices = list(r["mc1_targets"]["choices"])
            labels = list(r["mc1_targets"]["labels"])
            correct = choices[labels.index(1)] if 1 in labels else None
            out[split].append(
                {
                    "group": "general",
                    "source": "truthfulqa/truthful_qa:multiple_choice:validation",
                    "prompt": (
                        "Answer this multiple-choice question with the letter only.\n\n"
                        f"Question: {r['question'].strip()}\n"
                        + "\n".join(f"{'ABCD'[i]}. {c}" for i, c in enumerate(choices))
                        + "\n\nAnswer with a single letter (A, B, C, ...)."
                    ),
                    "reference_answer": correct,
                    "meta": {"extract": "mc_letter", "choices": choices},
                }
            )
        cursor += n
    return out


def stable_hash(example: dict) -> str:
    payload = json.dumps(example, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="benchmarks")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    builders = {
        "math": build_math,
        "instruction": build_instruction,
        "logic": build_logic,
        "repetition": build_repetition,
        "general": build_general,
    }

    splits: dict[str, list[dict]] = {s: [] for s in PER_SPLIT}

    for name, fn in builders.items():
        print(f"[build] {name} ...", flush=True)
        part = fn(random.Random(args.seed + hash(name) % 10000))
        for split in PER_SPLIT:
            assert len(part[split]) == PER_SPLIT[split][name], (
                f"{name}/{split}: 期望 {PER_SPLIT[split][name]}，得到 {len(part[split])}"
            )
            splits[split].extend(part[split])

    # 编号 + 哈希
    for split in PER_SPLIT:
        for i, ex in enumerate(splits[split]):
            ex["uid"] = f"{split}-{i:04d}"
            ex["checksum"] = stable_hash(ex)

    # 校验三组隔离：prompt 层面不允许任何重叠
    prompts = {s: {e["prompt"] for e in splits[s]} for s in splits}
    for a in splits:
        for b in splits:
            if a < b:
                overlap = prompts[a] & prompts[b]
                assert not overlap, f"{a} 与 {b} 存在 {len(overlap)} 条重叠样本！"

    out_dir = Path(args.output_dir)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "per_split": PER_SPLIT,
        "splits": {},
    }
    for split, rows in splits.items():
        p = out_dir / split / "data.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for ex in rows:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        manifest["splits"][split] = {
            "file": str(p),
            "n": len(rows),
            "groups": {g: sum(1 for e in rows if e["group"] == g) for g in sorted({e["group"] for e in rows})},
        }
        print(f"[write] {p}  n={len(rows)}")

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[done] manifest ->", out_dir / "manifest.json")


if __name__ == "__main__":
    main()
