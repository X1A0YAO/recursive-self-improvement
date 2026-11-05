"""Algorithmic Baseline 审计 Part 2：范数 vs 功能重要性（需加载模型）。

机理假设检验：
  H1 范数≠重要性：病灶选的是 top-512 行范数神经元，但高范数≠高功能负载
  H2 冗余：SwiGLU 9728 维中间层大量冗余，5.3% 神经元死亡可被补偿
  H3 活跃度稀疏：病灶神经元在数学任务上可能根本不活跃（贡献≈0）

实验：
  1. 在 16 条 optimization math prompts 上捕获 L26 mlp 的逐神经元贡献
     c_i = |silu(gate(x)_i) * up(x)_i|（SwiGLU 真实输出贡献）
  2. 计算重合度：top-512范数神经元 ∩ top-512活跃度神经元
  3. 反事实 logprob 测试（teacher-forced，快）：
     (a) top-norm-512 行置零  (b) top-activity-512 行置零  (c) random-512
     对比 8 条数学题 gold 完整解的 logprob 变化
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from model_utils import load_model  # noqa: E402

LAYER = 26


@torch.no_grad()
def capture_contributions(model, tokenizer, prompts: list[str]) -> torch.Tensor:
    """返回 [n_neurons] 平均 |silu(gate)*up| 贡献。"""
    layer = model.model.layers[LAYER].mlp
    acc = None
    n = 0
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt").to(model.device)
        # hook 捕获 up 与 gate 输出
        ups, gates = [], []

        def up_hook(m, i, o):
            ups.append(o.detach())

        def gate_hook(m, i, o):
            gates.append(o.detach())

        h1 = layer.up_proj.register_forward_hook(up_hook)
        h2 = layer.gate_proj.register_forward_hook(gate_hook)
        model(**ids)
        h1.remove()
        h2.remove()
        c = (torch.nn.functional.silu(gates[0]) * ups[0]).abs().mean(dim=(0, 1))  # [9728]
        acc = c if acc is None else acc + c
        n += 1
    return acc / n


@torch.no_grad()
def logprob_delta(model, tokenizer, examples: list[dict], lesion_rows: torch.Tensor | None, base_lp: list[float]) -> float:
    """对 8 条数学题的 gold 解做 teacher-forced logprob，返回均值变化。"""
    deltas = []
    for ex, bl in zip(examples, base_lp):
        row = model.model.layers[LAYER].mlp.up_proj.weight.data
        saved = None
        if lesion_rows is not None:
            saved = row[lesion_rows].clone()
            row[lesion_rows] = 0
        prompt = ex["prompt"]
        gold = ex.get("gold_answer", "")
        text = prompt + "\n" + str(gold)
        ids = tokenizer(text, return_tensors="pt").to(model.device)
        out = model(**ids, labels=ids["input_ids"])
        lp = -out.loss.item() * ids["input_ids"].shape[1]
        if saved is not None:
            row[lesion_rows] = saved
        deltas.append(lp - bl)
    return sum(deltas) / len(deltas)


def main() -> None:
    model, tokenizer = load_model(str(ROOT / "checkpoints/V0"))
    model.eval()

    bench = json.loads((ROOT / "benchmarks/manifest.json").read_text(encoding="utf-8"))
    # 16 条 math optimization prompts（激活测量用）
    import json as _json

    opt_lines = (ROOT / "benchmarks/optimization/data.jsonl").read_text(encoding="utf-8").strip().splitlines()
    opt = [_json.loads(x) for x in opt_lines]
    math_items = [e for e in opt if e.get("group") == "math"][:16]
    prompts = [e["prompt"] for e in math_items]

    print(f"[audit2] 捕获 L{LAYER} mlp 逐神经元贡献（{len(prompts)} prompts）...", flush=True)
    contrib = capture_contributions(model, tokenizer, prompts).float().cpu()

    from safetensors import safe_open

    with safe_open(str(ROOT / "checkpoints/V0/model-00002-of-00003.safetensors"), framework="pt", device="cpu") as f:
        pass  # 仅占位；实际用 named_parameters

    w = dict(model.named_parameters())[f"model.layers.{LAYER}.mlp.up_proj.weight"].data.float().cpu()
    row_norms = w.norm(dim=1)
    top_norm = torch.argsort(row_norms, descending=True)[:512]
    top_act = torch.argsort(contrib, descending=True)[:512]

    inter = len(set(top_norm.tolist()) & set(top_act.tolist()))
    print(f"\n[audit2] top-512范数 ∩ top-512活跃度 = {inter}/512 ({inter/512*100:.1f}%)")

    # 病灶神经元的活跃度分布
    cn = contrib[top_norm]
    print(f"[audit2] 病灶神经元(top-norm)贡献: mean={cn.mean():.4f} median={cn.median():.4f}")
    ca = contrib[top_act]
    print(f"[audit2] top-activity神经元贡献:   mean={ca.mean():.4f} median={ca.median():.4f}")
    ratio = cn.mean() / ca.mean()
    print(f"[audit2] 病灶神经元平均贡献仅为最高活跃神经元的 {ratio*100:.1f}%")

    # 活跃度稀疏：贡献近零的神经元比例
    dead_thresh = contrib.quantile(0.5)
    print(f"[audit2] 全层贡献中位数={dead_thresh:.5f}；病灶神经元中低于中位数的有 {(cn < dead_thresh).float().mean()*100:.0f}%")

    # 反事实 logprob：8 条数学题 gold 解
    ex_for_lp = []
    for e in math_items[:8]:
        ex_for_lp.append({"prompt": e["prompt"], "gold_answer": e.get("answer", e.get("gold", ""))})

    base = []
    with torch.no_grad():
        for ex in ex_for_lp:
            text = ex["prompt"] + "\n" + str(ex["gold_answer"])
            ids = tokenizer(text, return_tensors="pt").to(model.device)
            out = model(**ids, labels=ids["input_ids"])
            base.append(-out.loss.item() * ids["input_ids"].shape[1])

    gen = torch.Generator().manual_seed(0)
    rand_rows = torch.randperm(9728, generator=gen)[:512]

    arms = {
        "top_norm_512": top_norm.to(model.device),
        "top_activity_512": top_act.to(model.device),
        "random_512": rand_rows.to(model.device),
    }
    results = {}
    for name, rows in arms.items():
        d = logprob_delta(model, tokenizer, ex_for_lp, rows, base)
        results[name] = d
        print(f"[audit2] logprob Δ [{name}]: {d:+.4f}")

    out = {
        "layer": LAYER,
        "overlap_topnorm_topact": inter,
        "lesion_neuron_contrib_mean": float(cn.mean().item()),
        "topact_neuron_contrib_mean": float(ca.mean().item()),
        "ratio_pct": float(ratio * 100),
        "counterfactual_logprob_delta": {k: float(v) for k, v in results.items()},
    }
    Path(ROOT / "results/audit_part2.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n[audit2] -> results/audit_part2.json")


if __name__ == "__main__":
    main()
