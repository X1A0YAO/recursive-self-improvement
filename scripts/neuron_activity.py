"""神经元任务集活跃度公共模块（Phase 2 预注册 §0 的 a_i 定义实现）。

a_i = mean over prompts x, mean over tokens t, of |silu(gate_i(x_t)) * up_i(x_t)|

参照集 P = optimization split 全部 25 条 math prompt（仅 prompt 前向）。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent


@torch.no_grad()
def capture_layer_activity(model, tokenizer, prompts: list[str], layer: int, proj: str = "mlp.up_proj") -> torch.Tensor:
    """返回 [n_neurons] 的 a_i（CPU float32）。

    对 up_proj/gate_proj 的行（神经元）计算 SwiGLU 贡献 h_i = silu(g_i)*u_i。
    对其它 proj（如 down_proj 的列）不支持（预注册只定义了 up_proj 行）。
    """
    assert proj == "mlp.up_proj", "预注册 §0 仅定义 up_proj 行的 a_i"
    mlp = model.model.layers[layer].mlp
    acc = None
    n = 0
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt").to(model.device)
        ups, gates = [], []

        def up_hook(m, i, o):
            ups.append(o.detach())

        def gate_hook(m, i, o):
            gates.append(o.detach())

        h1 = mlp.up_proj.register_forward_hook(up_hook)
        h2 = mlp.gate_proj.register_forward_hook(gate_hook)
        model(**ids)
        h1.remove()
        h2.remove()
        c = (torch.nn.functional.silu(gates[0]) * ups[0]).abs().float().mean(dim=(0, 1)).cpu()
        acc = c if acc is None else acc + c
        n += 1
    return acc / n


def load_reference_prompts() -> list[str]:
    lines = (ROOT / "benchmarks/optimization/data.jsonl").read_text(encoding="utf-8").strip().splitlines()
    items = [json.loads(x) for x in lines]
    return [e["prompt"] for e in items if e.get("group") == "math"]


def top_rows(w: torch.Tensor, activity: torch.Tensor, k: int, select: str) -> torch.Tensor:
    """按选择策略返回行索引（降序前 k）。w: [n,d] 权重；activity: [n]。"""
    if select == "norm":
        return torch.argsort(w.float().norm(dim=1), descending=True)[:k]
    if select == "activity":
        return torch.argsort(activity, descending=True)[:k]
    raise ValueError(f"unknown select: {select}")
