"""算法定位基线：仅用 Agent 可见的统计量，能否定位病灶？

模拟 compare_tensor_stats 的输出（mean/std/l1/l2/zero_rows），
用简单的异常检测（z-score across layers）在 7 种投影上找异常层。
这是"如果模型会做数据分析，它能达到的水平"——定位任务难度的参照。
"""

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_tensor(model_dir: str, layer: int, proj: str):
    d = Path(model_dir)
    name = (
        f"model.layers.{layer}.mlp.{proj}.weight"
        if proj in ("gate_proj", "up_proj", "down_proj")
        else f"model.layers.{layer}.self_attn.{proj}.weight"
    )
    single = d / "model.safetensors"
    idx_file = d / "model.safetensors.index.json"
    if single.exists() and (not idx_file.exists() or idx_file.stat().st_mtime < single.stat().st_mtime):
        with safe_open(str(single), framework="pt", device="cpu") as f:
            return f.get_tensor(name)
    idx = json.loads(idx_file.read_text(encoding="utf-8"))
    shard = idx["weight_map"][name]
    with safe_open(str(d / shard), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def main():
    defect_dir = "checkpoints/V0-defect"
    n_layers = 36
    projs = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # 真实病灶（密封记录，用于评分）
    rec = json.loads(Path("experiments/iteration_1/_sealed/defect_record.json").read_text(encoding="utf-8"))
    truth = set(rec["tensor_names"])

    print("=== 模拟 Agent 视角：跨层统计 + z-score 异常检测 ===\n")
    found = {}
    for proj in projs:
        rows = []
        for L in range(n_layers):
            t = load_tensor(defect_dir, L, proj).float()
            rows.append(
                {
                    "layer": L,
                    "mean": t.mean().item(),
                    "std": t.std().item(),
                    "l2": t.norm().item(),
                    "zero_rows": int((t.abs().sum(dim=1) == 0).sum()) if t.dim() == 2 else 0,
                }
            )
        # z-score on each metric
        import statistics as st

        flagged = []
        for metric in ("mean", "std", "l2"):
            vals = [r[metric] for r in rows]
            mu, sd = st.mean(vals), st.stdev(vals)
            for r in rows:
                z = (r[metric] - mu) / sd if sd > 0 else 0
                if abs(z) > 2.5:
                    flagged.append((r["layer"], metric, round(z, 2)))
        zr = [(r["layer"], r["zero_rows"]) for r in rows if r["zero_rows"] > 0]
        if zr:
            flagged.extend([(L, "zero_rows", n) for L, n in zr])
        found[proj] = flagged
        if flagged:
            print(f"{proj}: 异常 -> {flagged}")
        else:
            print(f"{proj}: 无异常 (所有 |z|<2.5)")

    print("\n=== 判定 ===")
    all_flagged = {f"model.layers.{L}.{ 'mlp' if p in ('gate_proj','up_proj','down_proj') else 'self_attn'}.{p}.weight"
                   for p, flags in found.items() for (L, _, _) in flags}
    hits = all_flagged & truth
    print(f"真实病灶张量: {sorted(truth)}")
    print(f"算法标记张量: {sorted(all_flagged)}")
    print(f"命中: {len(hits)}/{len(truth)}  精确率: {len(hits)}/{len(all_flagged) if all_flagged else 0}")


if __name__ == "__main__":
    main()
