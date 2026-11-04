"""病灶方案 A/B/C 行为级对比（一次性生成测试，不落盘 checkpoint）。

A. layer26 up_proj top-256 行置零（重要神经元病灶）
B. layer26 down_proj top-256 列置零（关键输出通路病灶）
C. layer26 input_layernorm 权重 top-25% 置零（归一化通路病灶）

各跑 8 道数学题（贪心），看正确率 + 输出健康度。
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import load_split, score_one  # noqa: E402
from model_utils import generate, load_model  # noqa: E402


def run_arm(model, tokenizer, examples, arm_name):
    results = []
    for ex in examples:
        gen = generate(model, tokenizer, ex["prompt"], max_new_tokens=512, enable_thinking=False)
        truncated = gen["num_new_tokens"] >= 512
        s = score_one(ex, gen["response"], truncated)
        results.append(s)
    mean = sum(r["score"] for r in results) / len(results)
    print(f"[arm {arm_name}] mean={mean:.3f}", flush=True)
    return mean


def main():
    examples = [e for e in load_split(Path("benchmarks/optimization/data.jsonl")) if e["group"] == "math"][:8]

    print("[ab] loading V0", flush=True)
    model, tok = load_model("checkpoints/V0")

    params = dict(model.named_parameters())

    # baseline
    base = run_arm(model, tok, examples, "baseline")

    # A: up_proj top-256 rows
    W = params["model.layers.26.mlp.up_proj.weight"].data
    W0 = W.clone()
    top = torch.argsort(W.float().norm(dim=1), descending=True)[:256]
    W[top] = 0
    a = run_arm(model, tok, examples, "A_up_rows")
    W.copy_(W0)

    # B: down_proj top-256 cols
    W = params["model.layers.26.mlp.down_proj.weight"].data
    W0 = W.clone()
    top = torch.argsort(W.float().norm(dim=0), descending=True)[:256]
    W[:, top] = 0
    b = run_arm(model, tok, examples, "B_down_cols")
    W.copy_(W0)

    # C: input_layernorm top 25% 置零
    W = params["model.layers.26.input_layernorm.weight"].data
    W0 = W.clone()
    k = int(W.numel() * 0.25)
    top = torch.argsort(W.float(), descending=True)[:k]
    W[top] = 0
    c = run_arm(model, tok, examples, "C_layernorm")
    W.copy_(W0)

    print(f"\n[ab] summary: base={base:.3f}  A={a:.3f}  B={b:.3f}  C={c:.3f}", flush=True)


if __name__ == "__main__":
    main()
