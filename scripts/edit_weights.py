"""Weight Editor：执行经过验证的结构化修改提案（Phase 1+ 使用，Phase 0 仅就位）。

安全规则：
- 源 checkpoint 目录只读，绝不就地修改。
- 每次修改产出新的 checkpoint 目录（含完整权重副本）。
- 应用前经 edit_protocol.validate_edit() 校验。
- 应用后做 NaN/Inf 检查、L1/L2 统计、参数占比统计，写 edit manifest。
- 任何数值异常 -> 自动丢弃输出目录并报错（回滚）。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.edit_protocol import load_param_index, validate_edit  # noqa: E402


def _copy_static_files(src: Path, dst: Path) -> None:
    """复制 json/txt 等配置文件（权重文件单独处理）。"""
    for f in src.iterdir():
        if f.is_file() and f.suffix in {".json", ".txt", ".py", ".md"}:
            shutil.copy2(f, dst / f.name)


def _load_all_tensors(src: Path) -> dict[str, torch.Tensor]:
    single = src / "model.safetensors"
    index_file = src / "model.safetensors.index.json"
    tensors: dict[str, torch.Tensor] = {}
    # 单文件存在且比 index 新（或无 index）→ 单文件优先（注入器产物）
    if single.exists() and (not index_file.exists() or single.stat().st_mtime >= index_file.stat().st_mtime - 60):
        return load_file(str(single), device="cpu")
    if index_file.exists():
        index = json.loads(index_file.read_text(encoding="utf-8"))
        shards = sorted(set(index["weight_map"].values()))
    else:
        shards = sorted(p.name for p in src.glob("*.safetensors"))
    for shard in shards:
        tensors.update(load_file(str(src / shard), device="cpu"))
    return tensors


def apply_low_rank_add(W: torch.Tensor, u: torch.Tensor, v: torch.Tensor, alpha: float) -> torch.Tensor:
    if W.dim() != 2:
        raise ValueError(f"low_rank_add 需要 2D 权重，得到 {tuple(W.shape)}")
    u = u.reshape(-1, 1).to(W.dtype)
    v = v.reshape(1, -1).to(W.dtype)
    return W + alpha * (u @ v)


def apply_edit(W: torch.Tensor, op: str, params: dict) -> torch.Tensor:
    if op == "scale":
        return W * params["factor"]
    if op == "add":
        delta = params["delta"]  # 同形状张量（序列化路径下用 spec 解析，此处内存态）
        return W + delta
    if op == "low_rank_add":
        return apply_low_rank_add(W, params["u"], params["v"], params["alpha"])
    if op == "index_assign":
        idx = params["indices"]
        values = params["values"]
        W = W.clone()
        W[tuple(idx)] = values
        return W
    raise ValueError(f"未知 op: {op}")


def tensor_stats(t: torch.Tensor) -> dict:
    tf = t.float()
    return {
        "mean": tf.mean().item(),
        "std": tf.std().item() if tf.numel() > 1 else 0.0,
        "norm_l1": tf.abs().sum().item(),
        "norm_l2": tf.norm().item(),
        "max_abs": tf.abs().max().item(),
        "has_nan": bool(torch.isnan(t).any()),
        "has_inf": bool(torch.isinf(t).any()),
    }


def run_edit(proposal_path: Path, param_index_path: Path, src_dir: Path, dst_dir: Path) -> dict:
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    param_index = load_param_index(param_index_path)

    v = validate_edit(proposal, param_index)
    if not v.ok:
        raise ValueError(f"提案未通过验证: {v.errors}")

    if dst_dir.exists():
        raise FileExistsError(f"输出目录已存在（禁止覆盖）: {dst_dir}")

    print(f"[edit] loading tensors from {src_dir}")
    tensors = _load_all_tensors(src_dir)

    edit_records = []
    global_delta_l2 = 0.0

    for e in proposal["edits"]:
        name = e["tensor"]
        W = tensors[name]
        W_new = apply_edit(W, e["op"], e.get("params", {}))

        before = tensor_stats(W.float())
        after = tensor_stats(W_new.float())
        delta = (W_new.float() - W.float())
        d_stats = {
            "l1": delta.abs().sum().item(),
            "l2": delta.norm().item(),
            "max_abs": delta.abs().max().item(),
            "num_changed": int((delta != 0).sum()),
            "numel": W.numel(),
            "changed_ratio": float((delta != 0).sum()) / W.numel(),
        }
        if d_stats["l2"] > (e.get("max_delta_l2") or float("inf")):
            raise ValueError(f"{name}: delta L2 {d_stats['l2']:.4f} 超过限制")
        global_delta_l2 += d_stats["l2"]

        # NaN/Inf 检查：异常即中止（不写出 dst）
        if after["has_nan"] or after["has_inf"]:
            raise ValueError(f"{name}: 修改后出现 NaN/Inf，回滚")

        tensors[name] = W_new
        edit_records.append(
            {
                "tensor": name,
                "op": e["op"],
                "before": before,
                "after": after,
                "delta": d_stats,
            }
        )

    budget = proposal.get("budget", {})
    if global_delta_l2 > budget.get("max_delta_l2_global", 2.0):
        raise ValueError(f"全局 delta L2 {global_delta_l2:.4f} 超预算")

    # 写出新 checkpoint
    dst_dir.mkdir(parents=True)
    _copy_static_files(src_dir, dst_dir)
    save_file(tensors, str(dst_dir / "model.safetensors"), metadata={"format": "pt"})

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(src_dir),
        "proposal_id": proposal["proposal_id"],
        "rationale": proposal.get("rationale"),
        "hypothesis": proposal.get("hypothesis"),
        "global_delta_l2": global_delta_l2,
        "num_tensors_edited": len(edit_records),
        "edits": edit_records,
    }
    (dst_dir / "edit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[edit] new checkpoint -> {dst_dir}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal", required=True, help="修改提案 JSON 路径")
    ap.add_argument("--param-index", default="results/param_index_V0.json")
    ap.add_argument("--src", default="models/Qwen3-4B")
    ap.add_argument("--dst", required=True, help="输出 checkpoint 目录（须不存在）")
    args = ap.parse_args()

    run_edit(
        proposal_path=Path(args.proposal),
        param_index_path=Path(args.param_index),
        src_dir=Path(args.src),
        dst_dir=Path(args.dst),
    )


if __name__ == "__main__":
    main()
