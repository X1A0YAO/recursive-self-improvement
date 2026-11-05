"""实验一阶段 A：18 格扰动矩阵粗筛（预注册 §2、§3.1）。

每格：注入（内存中 patch，不落盘）→ teacher-forced logprob Δ（25 math gold）
     + quick-20 行为粗筛（16 math + 4 logic，批量生成）
只做描述统计排序，不做推断（推断留给阶段 B 的 DS-120 + McNemar + Holm）。

用法：
  python scripts/matrix_screen.py            # 全 18 格
  python scripts/matrix_screen.py --cells activity_zero_512 # 单格调试
产出：results/matrix_screen.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from model_utils import load_model, generate_batch  # noqa: E402
from evaluate import score_one  # noqa: E402
from neuron_activity import capture_layer_activity, load_reference_prompts  # noqa: E402

LAYER = 26
PROJS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SELECTS = ("norm", "activity")
OPS = ("zero", "scale", "lowrank")
KS = (128, 512, 2048)
SEED = 178798052


def cell_id(select: str, op: str, k: int) -> str:
    return f"{select}_{op}_{k}"


def apply_cell(W: torch.Tensor, rows: torch.Tensor, op: str, seed: int) -> None:
    if op == "zero":
        W[rows] = 0
    elif op == "scale":
        W[rows] = W[rows] * 0.5
    elif op == "lowrank":
        sub = W[rows]
        g = torch.Generator().manual_seed(seed)
        U = torch.randn(sub.shape[0], 4, generator=g).to(sub.device, sub.dtype)
        V = torch.randn(4, sub.shape[1], generator=g).to(sub.device, sub.dtype)
        delta = (U @ V) / 2.0
        sigma = 0.3 * sub.float().norm().item() / delta.float().norm().item()
        W[rows] = sub + (delta * sigma).to(sub.dtype)


@torch.no_grad()
def logprob_of(model, tokenizer, examples) -> float:
    """teacher-forced 全序列负对数似然均值（越小越好）。"""
    tot, n = 0.0, 0
    for ex in examples:
        text = ex["prompt"] + "\n" + ex["reference_answer"]
        ids = tokenizer(text, return_tensors="pt").to(model.device)
        out = model(**ids, labels=ids["input_ids"])
        tot += (-out.loss.item()) * ids["input_ids"].shape[1]
        n += 1
    return tot / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="", help="逗号分隔 cell id；空=全 18 格")
    ap.add_argument("--output", default="results/matrix_screen.json")
    args = ap.parse_args()

    # 数据
    opt_items = [json.loads(x) for x in (ROOT / "benchmarks/optimization/data.jsonl").read_text(encoding="utf-8").strip().splitlines()]
    math_items = [e for e in opt_items if e["group"] == "math"]
    assert len(math_items) == 25
    quick = [e for e in opt_items if e["group"] == "math"][:16] + [e for e in opt_items if e["group"] == "logic"][:4]

    model, tokenizer = load_model(str(ROOT / "checkpoints/V0"))
    model.eval()

    # 活跃度
    act_file = ROOT / "results/activity_L26.pt"
    if not act_file.exists():
        prompts = load_reference_prompts()
        torch.save(capture_layer_activity(model, tokenizer, prompts, LAYER), act_file)
    activity = torch.load(act_file, map_location="cpu")

    w_name = f"model.layers.{LAYER}.mlp.up_proj.weight"
    W = dict(model.named_parameters())[w_name]
    W0 = W.data.clone()

    # 基线 logprob（健康）
    base_lp = logprob_of(model, tokenizer, math_items)
    print(f"[screen] baseline logprob(25 math gold) = {base_lp:.2f}", flush=True)

    # 基线 quick-20
    base_quick = []
    for gen in generate_batch(model, tokenizer, [e["prompt"] for e in quick], max_new_tokens=768, batch_size=4, enable_thinking=False):
        base_quick.append(gen)
    base_scores = []
    for ex, g in zip(quick, base_quick):
        s = score_one(ex, g["response"], truncated=g["num_new_tokens"] >= 768)["score"]
        base_scores.append(s)
    print(f"[screen] baseline quick-20 mean = {sum(base_scores)/20:.3f}", flush=True)

    cells = args.cells.split(",") if args.cells else [cell_id(s, o, k) for s in SELECTS for o in OPS for k in KS]
    results = {}
    for cid in cells:
        select, op, k = cid.rsplit("_", 2)[0].split("_")[0], None, None
        # 解析 cell id: {select}_{op}_{k}
        parts = cid.split("_")
        select, op, k = parts[0], parts[1], int(parts[2])
        assert select in SELECTS and op in OPS and k in KS, cid

        rows = torch.argsort(W0.float().cpu().norm(dim=1), descending=True)[:k] if select == "norm" else torch.argsort(activity, descending=True)[:k]
        rows = rows.to(W.device)

        W.data.copy_(W0)
        apply_cell(W.data, rows, op, SEED + LAYER)
        lp = logprob_of(model, tokenizer, math_items)

        gens = []
        for gen in generate_batch(model, tokenizer, [e["prompt"] for e in quick], max_new_tokens=768, batch_size=4, enable_thinking=False):
            gens.append(gen)
        scores = [score_one(ex, g["response"], truncated=g["num_new_tokens"] >= 768)["score"] for ex, g in zip(quick, gens)]
        n_worse = sum(1 for b, s in zip(base_scores, scores) if s < b)
        n_better = sum(1 for b, s in zip(base_scores, scores) if s > b)
        mean_q = sum(scores) / len(scores)
        results[cid] = {
            "logprob_delta": round(lp - base_lp, 4),
            "quick20_mean": round(mean_q, 3),
            "quick20_worse": n_worse,
            "quick20_better": n_better,
        }
        print(f"[screen] {cid}: logprobΔ={lp-base_lp:+.3f} quick={mean_q:.3f} (worse {n_worse}/better {n_better})", flush=True)

        W.data.copy_(W0)  # 恢复

    out = {
        "baseline": {"logprob": base_lp, "quick20_mean": sum(base_scores) / 20},
        "layer": LAYER,
        "seed": SEED,
        "cells": results,
    }
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[screen] -> {args.output}")


if __name__ == "__main__":
    main()
