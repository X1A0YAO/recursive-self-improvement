"""Phase 1：受控缺陷注入器。

从 V0 生成 V0-defect：向选定的权重张量注入可逆的低秩扰动。

设计原则：
- 扰动记录（位置、幅度、随机种子）写入 sealed_ground_truth.json —— Agent 不可见
  （放在 experiments/iteration_1/_sealed/，该目录加入 .gitignore？不——实验必须可复现，
   所以要提交，但 Agent 的提示词工程中绝不引用它）。
- 扰动是确定性的（固定 seed），可精确回滚（V0 本身就是回滚点）。
- 注入强度分级（sigma 相对张量 std），目标：产生可测量但不毁灭性的行为退化。

用法：
  python scripts/inject_defect.py --src checkpoints/V0 --dst checkpoints/V0-defect \
      --layer-idx 18 --proj mlp.down_proj --sigma-frac 0.30 --seed 314 \
      --record experiments/iteration_1/_sealed/defect_record.json
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).parent))
from edit_weights import _copy_static_files, _load_all_tensors, tensor_stats  # noqa: E402


def inject_low_rank_noise(
    W: torch.Tensor,
    rng: torch.Generator,
    sigma_frac: float,
    rank: int = 4,
) -> tuple[torch.Tensor, dict]:
    """W' = W + sigma * (U V^T)，U/V 随机正交列。

    sigma = sigma_frac * W.std()，即相对幅度可控。
    rank 控制扰动的自由度（低秩 → 便于 oracle 对照修复）。
    """
    assert W.dim() == 2, f"expect 2D, got {tuple(W.shape)}"
    m, n = W.shape
    r = min(rank, m, n)
    U = torch.randn(m, r, generator=rng, dtype=torch.float32)
    V = torch.randn(n, r, generator=rng, dtype=torch.float32)
    # QR 正交化，让扰动方向均匀
    U, _ = torch.linalg.qr(U)
    V, _ = torch.linalg.qr(V)
    sigma = sigma_frac * W.float().std().item()
    delta = sigma * (U @ V.T) / math.sqrt(r)
    W_new = (W.float() + delta).to(W.dtype)
    info = {
        "sigma_abs": sigma,
        "sigma_frac": sigma_frac,
        "rank": r,
        "delta_l2": delta.norm().item(),
        "W_l2": W.float().norm().item(),
        "rel_l2": (delta.norm() / W.float().norm()).item(),
    }
    return W_new, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="checkpoints/V0")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--layer-idx", type=int, required=True, help="transformer 层号")
    ap.add_argument(
        "--proj",
        required=True,
        choices=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    ap.add_argument("--sigma-frac", type=float, default=0.30, help="噪声 sigma 相对张量 std 的比例")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--seed", type=int, default=314)
    ap.add_argument("--record", required=True, help="密封的 ground truth 记录路径")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        raise FileExistsError(f"目标已存在（禁止覆盖）: {dst}")

    tensor_name = f"model.layers.{args.layer_idx}.mlp.{args.proj}.weight" if args.proj in (
        "gate_proj", "up_proj", "down_proj") else f"model.layers.{args.layer_idx}.self_attn.{args.proj}.weight"

    print(f"[inject] loading {src}")
    tensors = _load_all_tensors(src)
    if tensor_name not in tensors:
        raise KeyError(f"张量不存在: {tensor_name}")

    W = tensors[tensor_name]
    rng = torch.Generator().manual_seed(args.seed)
    W_new, info = inject_low_rank_noise(W, rng, args.sigma_frac, args.rank)

    after = tensor_stats(W_new)
    if after["has_nan"] or after["has_inf"]:
        raise ValueError("注入后出现 NaN/Inf")

    tensors[tensor_name] = W_new

    dst.mkdir(parents=True)
    _copy_static_files(src, dst)
    save_file(tensors, str(dst / "model.safetensors"), metadata={"format": "pt"})

    record = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(src),
        "defect_checkpoint": str(dst),
        "tensor": tensor_name,
        "layer_idx": args.layer_idx,
        "proj": args.proj,
        "sigma_frac": args.sigma_frac,
        "rank": args.rank,
        "seed": args.seed,
        "stats": info,
        "before": tensor_stats(W),
        "after": after,
    }
    record_path = Path(args.record)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[inject] {tensor_name}")
    print(f"  sigma={info['sigma_abs']:.6f} (frac={args.sigma_frac}), rank={args.rank}")
    print(f"  delta_l2={info['delta_l2']:.4f}  rel_l2={info['rel_l2']:.6f}")
    print(f"[inject] defect checkpoint -> {dst}")
    print(f"[inject] sealed record -> {record_path}")
    print("[inject] 注意：密封记录不得以任何形式进入 Agent 提示词")


if __name__ == "__main__":
    main()
