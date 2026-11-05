"""Phase 2 通用缺陷注入器（预注册 §2：2 选择 × 3 操作 × 3 强度）。

用法：
  python scripts/inject_defect_p2.py --src checkpoints/V0 --dst checkpoints/p2/<cell> \
      --select activity --op zero --k 512 --layers 26 [--seed S]

操作定义（作用于所选行集合 R，|R|=k）：
  zero:   W[R] = 0
  scale:  W[R] = W[R] * 0.5
  lowrank+: 在 R 上叠加 delta[R] = (U V^T)/sqrt(r) * sigma，
            sigma = 0.3 * ||W[R]||_F（能量相对所选子块 30%），r=4，seed 驱动
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent))
from edit_weights import _copy_static_files, _load_all_tensors, tensor_stats  # noqa: E402

ROOT = Path(__file__).parent.parent


def apply_op(W: torch.Tensor, rows: torch.Tensor, op: str, seed: int) -> torch.Tensor:
    if op == "zero":
        W[rows] = 0
    elif op == "scale":
        W[rows] = W[rows] * 0.5
    elif op == "lowrank":
        sub = W[rows]
        g = torch.Generator().manual_seed(seed)
        U = torch.randn(sub.shape[0], 4, generator=g).to(sub.dtype)
        V = torch.randn(4, sub.shape[1], generator=g).to(sub.dtype)
        delta = (U @ V) / 2.0  # /sqrt(r)=2
        sigma = 0.3 * sub.float().norm().item() / delta.float().norm().item()
        W[rows] = sub + (delta * sigma).to(sub.dtype)
    else:
        raise ValueError(op)
    return W


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="checkpoints/V0")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--select", choices=["norm", "activity"], required=True)
    ap.add_argument("--op", choices=["zero", "scale", "lowrank"], required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--layers", required=True, help="逗号分隔层号")
    ap.add_argument("--seed", type=int, default=178798052)
    ap.add_argument("--activity-file", default="results/activity_L26.pt", help="活跃度向量缓存（按层）")
    ap.add_argument("--record", default="")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        raise FileExistsError(dst)
    layers = [int(x) for x in args.layers.split(",")]

    tensors = _load_all_tensors(src)

    # 活跃度（如需）：从缓存加载（每层一个文件 results/activity_L{L}.pt）
    acts = {}
    if args.select == "activity":
        for L in layers:
            f = ROOT / f"results/activity_L{L}.pt"
            if not f.exists():
                raise FileNotFoundError(f"先运行 capture_activity.py 生成 {f}")
            acts[L] = torch.load(f, map_location="cpu")

    touched = {}
    for L in layers:
        name = f"model.layers.{L}.mlp.up_proj.weight"
        W = tensors[name]
        if args.select == "norm":
            rows = torch.argsort(W.float().norm(dim=1), descending=True)[: args.k]
        else:
            rows = torch.argsort(acts[L], descending=True)[: args.k]
        W = apply_op(W, rows, args.op, args.seed + L)
        tensors[name] = W
        st = tensor_stats(W)
        assert not st["has_nan"] and not st["has_inf"]
        touched[name] = rows.tolist()

    dst.mkdir(parents=True)
    _copy_static_files(src, dst)
    save_file(tensors, str(dst / "model.safetensors"), metadata={"format": "pt"})

    record = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "type": "p2_matrix_cell",
        "source": str(src),
        "dst": str(dst),
        "select": args.select,
        "op": args.op,
        "k": args.k,
        "layers": layers,
        "proj": "mlp.up_proj",
        "seed": args.seed,
        "rows": touched,
    }
    if args.record:
        rp = Path(args.record)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(record), encoding="utf-8")
        print(f"[inject_p2] sealed record -> {rp}")
    print(f"[inject_p2] {args.select}/{args.op}/k={args.k}/L{layers} -> {dst}")


if __name__ == "__main__":
    main()
