"""Phase 3 张量预载脚本（Gate-1：GPU 加载前运行，独立进程）。

把实验所需张量从 safetensors 直读并保存为 RAM 友好的 .pt 缓存：
  results/_p3_tensor_cache/V0.pt          # 健康参照（全部 up_proj + 抽样张量）
  results/_p3_tensor_cache/<defect>.pt    # 各缺陷实例（同键集合）

为什么不是全模型：4B 全量 ~15GB RAM 压力大且无必要。预载集合 =
实验二诊断所需的张量子集（全部 36 层 7 投影太多——只预载 mlp 三投影
+ q/o 抽样层，足够缺陷池覆盖）。

用法：.venv\\Scripts\\python.exe -X utf8 scripts\\preload_tensors.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).parent.parent
CACHE_DIR = ROOT / "results" / "_p3_tensor_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFECTS = {
    "cascade3_activity_zero_512": "checkpoints/p2/cascade3_activity_zero_512",
    "activity_scale_512_L30": "checkpoints/p2/activity_scale_512_L30",
    "norm_lowrank_128_L19": "checkpoints/p2/norm_lowrank_128_L19",  # D3 用（注入见 sealed）
    "norm_zero_128_L19": "checkpoints/p2/norm_zero_128_L19",
}


def wanted_tensors() -> list[str]:
    """预载集合：36 层 mlp 三投影 + 8 个抽样层的 q/o。"""
    names = []
    for L in range(36):
        for p in ("up_proj", "gate_proj", "down_proj"):
            names.append(f"model.layers.{L}.mlp.{p}.weight")
    for L in (0, 5, 12, 19, 22, 26, 30, 35):
        names.append(f"model.layers.{L}.self_attn.q_proj.weight")
        names.append(f"model.layers.{L}.self_attn.o_proj.weight")
    return names


def load_from_dir(model_dir: Path, names: list[str]) -> dict[str, torch.Tensor]:
    single = model_dir / "model.safetensors"
    idx_file = model_dir / "model.safetensors.index.json"
    tensors: dict[str, torch.Tensor] = {}
    if single.exists() and (not idx_file.exists() or single.stat().st_mtime >= idx_file.stat().st_mtime - 60):
        with safe_open(str(single), framework="pt", device="cpu") as f:
            for n in names:
                tensors[n] = f.get_tensor(n)
    else:
        index = json.loads(idx_file.read_text(encoding="utf-8"))
        shards = {}
        for n in names:
            shards.setdefault(index["weight_map"][n], None)
        for shard in shards:
            with safe_open(str(model_dir / shard), framework="pt", device="cpu") as f:
                for n in names:
                    if index["weight_map"][n] == shard:
                        tensors[n] = f.get_tensor(n)
    return tensors


def main() -> None:
    names = wanted_tensors()
    print(f"[preload] {len(names)} tensors per checkpoint", flush=True)

    # 健康参照
    out = CACHE_DIR / "V0.pt"
    if not out.exists():
        print("[preload] V0 (healthy ref)...", flush=True)
        ts = load_from_dir(ROOT / "checkpoints/V0", names)
        torch.save(ts, out)
        print(f"[preload] V0 -> {out} ({sum(t.numel() for t in ts.values())/1e6:.0f}M params)", flush=True)
    else:
        print("[preload] V0 cache exists", flush=True)

    for defid, dpath in DEFECTS.items():
        out = CACHE_DIR / f"{defid}.pt"
        if out.exists():
            print(f"[preload] {defid} cache exists", flush=True)
            continue
        p = ROOT / dpath
        if not p.exists():
            print(f"[preload] SKIP {defid}: {p} not found", flush=True)
            continue
        print(f"[preload] {defid}...", flush=True)
        ts = load_from_dir(p, names)
        torch.save(ts, out)
        print(f"[preload] {defid} -> {out}", flush=True)


if __name__ == "__main__":
    main()
