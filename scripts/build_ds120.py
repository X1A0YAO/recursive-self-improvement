"""DS-120 诊断评测集构建（Phase 2 预注册 §1）。

构成：GSM8K math 50 + IFEval-style 20 + 逻辑 25 + 重复变换 25 = 120
要求：与既有 optimization/validation/test 严格无重叠（prompt 精确匹配 + 8-gram 近重复）
seed=20261105
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
SEED = 20261105


def norm8grams(text: str) -> set[tuple[str, ...]]:
    toks = re.findall(r"\w+", text.lower())
    return {tuple(toks[i : i + 8]) for i in range(len(toks) - 7)}


def main() -> None:
    rng = random.Random(SEED)
    out = []

    # ---------- GSM8K math 50 ----------
    # 从本地缓存加载（build_benchmarks 已下载过 gsm8k test）
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    used_prompts: set[str] = set()
    used_grams: set[tuple[str, ...]] = set()
    for split in ("optimization", "validation", "test"):
        p = ROOT / "benchmarks" / split / "data.jsonl"
        if p.exists():
            for line in p.read_text(encoding="utf-8").strip().splitlines():
                e = json.loads(line)
                if e.get("group") == "math" and "gsm8k" in e.get("source", ""):
                    used_prompts.add(e["prompt"].strip())
                    used_grams |= norm8grams(e["prompt"])

    pool = [ex for ex in ds if ex["question"].strip() not in used_prompts]
    rng.shuffle(pool)
    taken = 0
    for ex in pool:
        q = ex["question"].strip()
        grams = norm8grams(q)
        if grams & used_grams:  # 8-gram 近重复过滤
            continue
        gold = re.search(r"####\s*(-?[\d,]+)", ex["answer"])
        if not gold:
            continue
        out.append(
            {
                "group": "math",
                "source": "openai/gsm8k:main:test",
                "prompt": q,
                "reference_answer": ex["answer"],
                "meta": {"extract": "last_number"},
                "uid": f"ds120-math-{taken:03d}",
            }
        )
        used_grams |= grams
        taken += 1
        if taken >= 50:
            break
    assert taken == 50, f"math only {taken}"

    # ---------- IFEval-style 指令 20 ----------
    instr_templates = [
        ("Answer in at least {n} words. Include the word {w} at least twice.", "min_words"),
        ("Answer in exactly {n} bullet points. Start each with a dash.", "bullets"),
        ("Wrap your entire answer in double quotation marks.", "quotes"),
        ("End your response with the exact phrase {w}.", "end_phrase"),
        ("Use no commas anywhere in your answer.", "no_commas"),
        ("Answer in all lowercase letters only.", "lowercase"),
        ("Write your answer in at most {n} words.", "max_words"),
        ("Include the keywords {w1} and {w2} in your answer.", "keywords"),
    ]
    base_qs = [
        "What are the main benefits of regular exercise?",
        "Explain why the sky appears blue.",
        "Describe how a refrigerator works.",
        "What causes inflation in an economy?",
        "How do vaccines protect against disease?",
        "Explain the difference between weather and climate.",
        "What is the purpose of the stock market?",
        "Why do leaves change color in autumn?",
        "Describe the water cycle.",
        "What makes a good scientific hypothesis?",
    ]
    words_bank = ["moreover", "however", "consequently", "furthermore", "nevertheless"]
    idx = 0
    combos = [(q, t) for q in base_qs for t in instr_templates]
    rng.shuffle(combos)
    for q, (tmpl, kind) in combos:
        if idx >= 20:
            break
        w = rng.choice(words_bank)
        n = rng.choice([30, 50, 80])
        instr = tmpl.format(n=n, w=w, w1=rng.choice(words_bank), w2=rng.choice(words_bank))
        prompt = f"{q} {instr}"
        if prompt.strip() in used_prompts:
            continue
        meta = {"extract": "ifeval_strict", "instruction": instr, "kind": kind,
                "kw": w if "{w}" in tmpl else None, "n": n}
        out.append({"group": "instruction", "source": "local:ifeval_style", "prompt": prompt,
                    "reference_answer": "", "meta": meta, "uid": f"ds120-instr-{idx:03d}"})
        used_grams |= norm8grams(prompt)
        idx += 1
    assert idx == 20, f"instr only {idx}"

    # ---------- 逻辑 25 ----------
    logic_items = [
        ("A is taller than B. B is taller than C. Who is the tallest? Answer with one name.", "A"),
        ("X happened before Y. Y happened before Z. Which happened first, X or Z? Answer with one letter.", "X"),
        ("All roses are flowers. Some flowers fade quickly. Must all roses fade quickly? Answer yes or no.", "no"),
        ("If it rains, the ground gets wet. The ground is not wet. Did it rain? Answer yes or no.", "no"),
        ("Mary has 3 brothers. Each brother has 2 sisters. How many sisters does Mary have?", "1"),
        ("A farmer has 17 sheep. All but 9 run away. How many remain?", "9"),
        ("John's father has 5 sons: One, Two, Three, Four, and what is the fifth son's name?", "John"),
        ("Which is heavier: a kilogram of feathers or a kilogram of steel? Answer with one word.", "neither"),
        ("Some cats are black. Tom is a cat. Is Tom necessarily black? Answer yes or no.", "no"),
        ("If today is Wednesday, what day will it be in 100 days? Answer with the day name.", "Friday"),
        ("A plane crashes on the border of France and Germany. Where are the survivors buried? Think carefully, then answer with one word.", "nowhere"),
        ("Peacock is a bird. Birds can typically fly. Can a peacock definitely fly long distances? Answer yes or no.", "no"),
        ("There are 5 apples in a basket. You take away 3. How many apples do you have?", "3"),
        ("Everything made of gold is valuable. This ring is valuable. Is this ring definitely made of gold? Answer yes or no.", "no"),
        ("A doctor gives you 3 pills, one every half hour. How long until the last pill? Answer in minutes.", "60"),
        ("Tom is twice as old as Jerry was when Tom was as old as Jerry is now. Combined age is 42. How old is Tom? Answer with a number.", "24"),
        ("In a race, you overtake the second-place runner. What place are you in now? Answer with a number.", "2"),
        ("How many months have 28 days? Answer with a number.", "12"),
        ("Anna reads 4 books monthly. Each book has 300 pages. How many pages monthly? Answer with a number.", "1200"),
        ("If two workers build a wall in 6 and 12 hours respectively, working together how many hours? Answer with a number rounded to 1 decimal.", "4.0"),
        ("A snail climbs 3m up a 10m well each day and slips 2m each night. How many days to exit? Answer with a number.", "8"),
        ("Kate was born in 1990. Her sister is 5 years younger. In what year was the sister born?", "1995"),
        ("The sum of three consecutive integers is 42. What is the smallest? Answer with a number.", "13"),
        ("Two dice are rolled. What is the probability of getting a sum of 12? Answer as a fraction.", "1/36"),
        ("A notebook and a pencil cost $1.50 together. The notebook costs $1.00 more than the pencil. How much is the pencil? Answer in dollars.", "0.25"),
    ]
    assert len(logic_items) == 25
    for i, (q, a) in enumerate(logic_items):
        out.append({"group": "logic", "source": "local:logic", "prompt": q,
                    "reference_answer": a, "meta": {"extract": "keyword"}, "uid": f"ds120-logic-{i:03d}"})

    # ---------- 重复变换 25 ----------
    # 与基线 repetition 组同构（防循环：奖励低重复率），措辞不同避免 8-gram 重叠。
    # 评分器零改动：requested_sentences 字段 + repetition_stats 提取。
    topics = [
        "the water cycle", "a blacksmith's workday", "how bread rises", "the history of coins",
        "a rainforest canopy", "how vaccines protect us", "a lighthouse keeper's routine", "making maple syrup",
        "deep sea vents", "the rules of chess", "how bees find their way", "the life of a mountain",
        "bats in the city", "the printing press", "a coral reef", "the winter solstice",
        "the Silk Road", "how bridges carry load", "a volcano's life", "how seeds travel",
        "the invention of bicycles", "tide pools", "airships", "how camels survive deserts", "the aurora borealis",
    ]
    assert len(topics) == 25
    for i, topic in enumerate(topics):
        n = [3, 4, 5][i % 3]
        q = (
            f"Write {n} different sentences about {topic}. Each sentence must contain "
            "new information, and no sentence may closely repeat another."
        )
        out.append({"group": "repetition", "source": "local:repetition_v2", "prompt": q,
                    "reference_answer": None,
                    "meta": {"extract": "repetition_stats", "requested_sentences": n},
                    "uid": f"ds120-rep-{i:03d}"})

    # ---------- 校验与落盘 ----------
    assert len(out) == 120
    # 与既有 split 重叠检查（prompt 精确匹配 + 8-gram）
    overlap = 0
    for split in ("optimization", "validation", "test"):
        p = ROOT / "benchmarks" / split / "data.jsonl"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").strip().splitlines():
            e = json.loads(line)
            if e["prompt"].strip() in {o["prompt"].strip() for o in out}:
                overlap += 1
            elif norm8grams(e["prompt"]) & set().union(*[norm8grams(o["prompt"]) for o in out]):
                overlap += 1
    assert overlap == 0, f"overlap with existing splits: {overlap}"

    # DS-120 内部无重复
    prompts = [o["prompt"].strip() for o in out]
    assert len(prompts) == len(set(prompts)), "internal duplicates"

    d = ROOT / "benchmarks/ds120"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "data.jsonl", "w", encoding="utf-8") as f:
        for o in out:
            o["checksum"] = hashlib.sha256(o["prompt"].encode()).hexdigest()[:16]
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    manifest = {
        "generated_at": "2026-11-05",
        "seed": SEED,
        "n": len(out),
        "groups": {g: sum(1 for o in out if o["group"] == g) for g in ("math", "instruction", "logic", "repetition")},
        "overlap_check": "exact + 8-gram vs optimization/validation/test: 0",
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ds120] {len(out)} examples -> benchmarks/ds120/data.jsonl")
    print(f"[ds120] groups: {manifest['groups']}")


if __name__ == "__main__":
    main()
