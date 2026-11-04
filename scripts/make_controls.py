"""Phase 1 对照组生成器。

arm E (oracle)：已知扰动位置（defect_record），直接构造逆扰动。
  - 精确逆：W' = W_defect - delta（需要保存注入时的 delta —— inject_defect 保存了 seed，
    可精确重放）。
arm C (random)：随机选层、随机张量、相近幅度扰动，作为"盲目修改"对照。
  - 注意：random arm 检验的是"诊断的价值"——如果随机修复也有效，说明任务太简单。

用法：
  # oracle 修复（读取密封记录，重放 delta 并取逆）
  python scripts/make_controls.py oracle --record experiments/iteration_1/_sealed/defect_record.json \
      --dst checkpoints/V1-oracle

  # random 修复（与提案相同的修改预算，但位置/方向随机）
  python scripts/make_controls.py random --record experiments/iteration_1/_sealed/defect_record.json \
      --dst checkpoints/V1-random --seed 777
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).parent))
from edit_weights import _copy_static_files, _load_all_tensors, tensor_stats  # noqa: E402
from inject_defect import inject_low_rank_noise  # noqa: E402


def make_oracle(record_path: Path, dst: Path) -> dict:
    """精确逆扰动：重放注入的随机 delta，然后减去。"""
    rec = json.loads(record_path.read_text(encoding="utf-8"))
    src = Path(rec["source"])
    tensor_name = rec["tensor"]
    seed = rec["seed"]
    sigma_frac = rec["sigma_frac"]
    rank = rec["rank"]

    tensors = _load_all_tensors(src)
    W = tensors[tensor_name]

    # 精确重放
    rng = torch.Generator().manual_seed(seed)
    W_corrupted, info = inject_low_rank_noise(W, rng, sigma_frac, rank)

    # 校验：dst 的输入应该是 W_corrupted（即 V0-defect）
    defect_dir = Path(rec["defect_checkpoint"])
    defect_tensors = _load_all_tensors(defect_dir)
    if not torch.equal(defect_tensors[tensor_name], W_corrupted):
        raise RuntimeError("重放校验失败：V0-defect 与重放结果不一致")

    # 逆 = 回到 V0（oracle 上界：完美修复）
    W_restored = W  # source 就是 V0
    tensors[tensor_name] = W_restored

    if dst.exists():
        raise FileExistsError(f"目标已存在: {dst}")
    dst.mkdir(parents=True)
    _copy_static_files(src, dst)
    save_file(tensors, str(dst / "model.safetensors"), metadata={"format": "pt"})

    delta_l2 = (W_corrupted.float() - W.float()).norm().item()
    return {
        "arm": "oracle",
        "method": "exact_inverse_replay",
        "tensor": tensor_name,
        "delta_l2_removed": delta_l2,
        "dst": str(dst),
        "note": "oracle = 完美还原 V0；作为可修复性上界参照",
    }


def make_random(record_path: Path, dst: Path, seed: int) -> dict:
    """随机修改对照：相同类型（低秩）、相近幅度，但位置与方向随机。

    随机修复不可能指向真实缺陷位置 —— 期望：无法恢复性能（甚至更糟）。
    """
    rec = json.loads(record_path.read_text(encoding="utf-8"))
    src = Path(rec["defect_checkpoint"])  # 从缺陷模型出发做"随机修复"
    sigma_frac = rec["sigma_frac"]
    rank = rec["rank"]

    tensors = _load_all_tensors(src)

    # 随机选一个不同的张量（排除真实缺陷所在层，避免碰巧命中）
    true_name = rec["tensor"]
    true_layer = rec["layer_idx"]
    proj_names = []
    for L in range(36):
        for proj in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                     "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"):
            n = f"model.layers.{L}.{proj}.weight"
            if n != true_name and L != true_layer:
                proj_names.append(n)

    g = torch.Generator().manual_seed(seed)
    pick = proj_names[int(torch.randint(0, len(proj_names), (1,), generator=g))]
    W = tensors[pick]
    rng = torch.Generator().manual_seed(seed + 1)
    W_new, info = inject_low_rank_noise(W, rng, sigma_frac, rank)
    tensors[pick] = W_new

    if dst.exists():
        raise FileExistsError(f"目标已存在: {dst}")
    dst.mkdir(parents=True)
    _copy_static_files(src, dst)
    save_file(tensors, str(dst / "model.safetensors"), metadata={"format": "pt"})

    return {
        "arm": "random",
        "method": "low_rank_noise_random_location",
        "tensor": pick,
        "sigma_frac": sigma_frac,
        "rank": rank,
        "seed": seed,
        "rel_l2": info["rel_l2"],
        "dst": str(dst),
        "note": "随机位置+随机方向的等幅修改；检验诊断价值",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["oracle", "random"])
    ap.add_argument("--record", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    record_path = Path(args.record)
    dst = Path(args.dst)
    if args.arm == "oracle":
        meta = make_oracle(record_path, dst)
    else:
        meta = make_random(record_path, dst, args.seed)

    meta["generated_at"] = datetime.now(timezone.utc).isoformat()
    out = dst / "control_manifest.json"
    out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
