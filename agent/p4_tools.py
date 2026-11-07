"""Phase 4.2：Agent RSI 工具组（真实任务驱动的自主权重修改）。

设计约束（预注册 §3 + Phase 1-3 教训 C1-C5）：
  - 观察类只给信息不替 Agent 决定（C1/C2：结构化统计视图，非裸数字）
  - 修改类执行 Agent 的结构化决定，带预算校验
  - 流程类对抗提案悬崖（C4：每轮必须输出结构化决定）
  - 反馈一律自由生成评测（C3）

工具：
  task_report      Train/Val 逐题对错 + 按指令类型失败统计（无官方检查细节）
  weight_stats     层/投影统计概览（复用 b2_tools 聚合缓存）
  activation_probe Agent 指定层/投影的激活统计（跑一小批失败题）
  apply_edit       结构化修改（预算校验后应用）
  decide           每轮结束的结构化决定（edit/stop/explore）
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# 观察类
# ---------------------------------------------------------------------------

def task_report(train_results: list[dict], val_results: list[dict]) -> str:
    """对错汇总 + 指令类型失败统计。Agent 看不到官方检查细节。"""
    def summarize(results: list[dict], name: str) -> list[str]:
        n = len(results)
        acc = sum(r["score"] for r in results) / max(n, 1)
        by_type: dict[str, list[float]] = {}
        for r in results:
            iid = r["instruction_ids"][0] if r["instruction_ids"] else "none"
            by_type.setdefault(iid, []).append(r["score"])
        lines = [
            f"=== {name}: {n} items, accuracy = {acc:.3f} ===",
            "",
            "Per-instruction-type accuracy (sorted by failure count):",
        ]
        fails = sorted(by_type.items(), key=lambda kv: sum(1 for s in kv[1] if s < 1), reverse=True)
        for iid, scores in fails[:12]:
            f = sum(1 for s in scores if s < 1)
            lines.append(f"  {iid}: {sum(scores):.0f}/{len(scores)} pass, {f} FAIL")
        return lines

    lines = []
    lines += summarize(train_results, "TRAIN (you may study these freely)")
    lines += [""] + summarize(val_results, "VALIDATION (your feedback signal)")
    # 失败样本摘录（train only——val 逐题细节不给，防止 Agent 直接拟合 val）
    lines += ["", "Sample FAILING train prompts (for your analysis):"]
    shown = 0
    for r in train_results:
        if r["score"] < 1 and shown < 5:
            lines.append(f"  [{r['instruction_ids'][0] if r['instruction_ids'] else 'none'}] {r.get('response', '')[:150]!r}")
            shown += 1
    return "\n".join(lines)


def weight_stats(model_dir_hint: str = "") -> str:
    """张量统计概览（252 单元聚合摘要，复用 b2_tools）。"""
    from agent.b2_tools import _compute_all_units_subprocess, _render_summary

    model_dir = model_dir_hint or "checkpoints/V0"
    units = _compute_all_units_subprocess(model_dir, 36)
    text, _ = _render_summary(units, 36)
    return text


@torch.no_grad()
def activation_probe(model, tokenizer, prompts: list[str], layer: int, proj: str) -> str:
    """Agent 指定层/投影的激活统计（用一小批题跑前向）。"""
    fam = "mlp" if proj in ("gate_proj", "up_proj", "down_proj") else "self_attn"
    mod = getattr(getattr(model.model.layers[layer], fam), proj)
    outs = []
    for p in prompts[:6]:
        ids = tokenizer(p, return_tensors="pt").to(model.device)
        captured: list[torch.Tensor] = []

        def hook(m, i, o):
            captured.append(o.detach().float())

        h = mod.register_forward_hook(hook)
        model(**ids)
        h.remove()
        o = captured[0]
        outs.append(o.flatten())
    t = torch.cat(outs)
    qs = torch.quantile(t, torch.tensor([0.1, 0.5, 0.9], device=t.device))
    return (
        f"activation stats for layer {layer} {proj} on {len(prompts[:6])} prompts:\n"
        f"  mean={t.mean():.4f} std={t.std():.4f}\n"
        f"  q10={qs[0]:.4f} q50={qs[1]:.4f} q90={qs[2]:.4f}\n"
        f"  frac_zero={(t == 0).float().mean():.3f} frac_neg={(t < 0).float().mean():.3f}"
    )


# ---------------------------------------------------------------------------
# 修改类（预算校验 + manifest 落盘）
# ---------------------------------------------------------------------------

EDIT_BUDGET = {"max_tensors_per_round": 2, "max_rounds": 6, "global_rel_delta_l2": 0.05}


def apply_edit(
    working_ckpt: Path,
    edits: list[dict],
    round_no: int,
    used_budget: dict,
) -> tuple[str, dict]:
    """应用结构化修改。edits: [{tensor, op, rows?, factor?}]。

    op: scale (整张量乘 factor) | row_scale (指定行乘 factor) |
        zero_rows | add_low_rank (u,v,alpha)
    """
    from safetensors.torch import load_file, save_file

    if len(edits) > EDIT_BUDGET["max_tensors_per_round"]:
        return f"REJECTED: >{EDIT_BUDGET['max_tensors_per_round']} tensors per round", used_budget

    manifest = {"round": round_no, "edits": [], "pre_values": {}}
    tensors = load_file(str(working_ckpt / "model.safetensors"), device="cpu")
    total_delta_l2 = 0.0
    total_norm = 0.0

    for ed in edits:
        name = ed["tensor"]
        if name not in tensors:
            return f"REJECTED: unknown tensor {name}", used_budget
        W = tensors[name].clone()
        W0 = W.clone()
        op = ed["op"]
        if op == "scale":
            W = W * float(ed["factor"])
        elif op == "row_scale":
            rows = [int(r) for r in ed["rows"]]
            W[rows] = W[rows] * float(ed["factor"])
        elif op == "zero_rows":
            rows = [int(r) for r in ed["rows"]]
            W[rows] = 0
        elif op == "add_low_rank":
            u = torch.tensor(ed["u"], dtype=W.dtype).reshape(-1, 1)
            v = torch.tensor(ed["v"], dtype=W.dtype).reshape(1, -1)
            W = W + float(ed["alpha"]) * (u @ v)
        else:
            return f"REJECTED: unknown op {op}", used_budget
        d = (W - W0).float().norm().item()
        n0 = W0.float().norm().item()
        total_delta_l2 += d
        total_norm += n0
        manifest["edits"].append({**ed, "delta_l2": round(d, 4), "rel": round(d / max(n0, 1e-9), 6)})
        manifest["pre_values"][name] = W0  # 存原值支持回滚

    rel_total = total_delta_l2 / max(total_norm, 1e-9)
    new_used = dict(used_budget)
    new_used["rel_delta_l2"] = used_budget.get("rel_delta_l2", 0.0) + rel_total
    new_used["rounds"] = round_no
    if new_used["rel_delta_l2"] > EDIT_BUDGET["global_rel_delta_l2"]:
        return (
            f"REJECTED: global budget exceeded "
            f"({new_used['rel_delta_l2']:.4f} > {EDIT_BUDGET['global_rel_delta_l2']})",
            used_budget,
        )

    # 写回（in-place 更新工作副本）
    for ed in edits:
        name = ed["tensor"]
        # 已在上面的循环里算好新值——重算一次写入（简单起见）
    # 直接用 tensors dict（已在循环中更新？没有——clone 上算的。重做写入）
    for ed in edits:
        name = ed["tensor"]
        W = tensors[name]
        op = ed["op"]
        if op == "scale":
            tensors[name] = W * float(ed["factor"])
        elif op == "row_scale":
            rows = [int(r) for r in ed["rows"]]
            W2 = tensors[name].clone()
            W2[rows] = W2[rows] * float(ed["factor"])
            tensors[name] = W2
        elif op == "zero_rows":
            rows = [int(r) for r in ed["rows"]]
            W2 = tensors[name].clone()
            W2[rows] = 0
            tensors[name] = W2
        elif op == "add_low_rank":
            u = torch.tensor(ed["u"], dtype=tensors[name].dtype).reshape(-1, 1)
            v = torch.tensor(ed["v"], dtype=tensors[name].dtype).reshape(1, -1)
            tensors[name] = tensors[name] + float(ed["alpha"]) * (u @ v)
    save_file(tensors, str(working_ckpt / "model.safetensors"), metadata={"format": "pt"})

    mf_path = working_ckpt.parent / f"edit_manifest_r{round_no}.json"
    # pre_values 是大张量——单独存 .pt
    torch.save(manifest["pre_values"], working_ckpt.parent / f"pre_values_r{round_no}.pt")
    mf = {k: v for k, v in manifest.items() if k != "pre_values"}
    mf["rel_total"] = round(rel_total, 6)
    mf_path.write_text(json.dumps(mf, indent=2, ensure_ascii=False), encoding="utf-8")
    return (
        f"APPLIED {len(edits)} edit(s): rel_delta={rel_total:.5f} "
        f"(cumulative {new_used['rel_delta_l2']:.5f}/{EDIT_BUDGET['global_rel_delta_l2']})",
        new_used,
    )


def rollback(working_ckpt: Path, round_no: int) -> str:
    """回滚某轮修改（用 pre_values.pt）。"""
    from safetensors.torch import load_file, save_file

    pre = torch.load(working_ckpt.parent / f"pre_values_r{round_no}.pt", map_location="cpu")
    tensors = load_file(str(working_ckpt / "model.safetensors"), device="cpu")
    for name, W0 in pre.items():
        tensors[name] = W0
    save_file(tensors, str(working_ckpt / "model.safetensors"), metadata={"format": "pt"})
    return f"ROLLED BACK round {round_no} ({len(pre)} tensors restored)"
