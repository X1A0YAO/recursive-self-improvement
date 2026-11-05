"""Algorithmic Baseline 审计 Part 1（无需加载模型，纯权重分析）。

审计对象：Phase 1 报告中的"算法定位基线 8/8 全召回"结论。
逐项检验：
  A. zero_rows 指纹是否健康模型本来就有（假阳性风险）
  B. 只用 z-score（不看 zero_rows）时的真实召回率 —— z-score 是否在"独立发现"病灶
  C. 病灶在 L2/mean/std 上的可见性量化（512/9728 行置零改变了什么）
  D. top-512 行范数选择是否"有幸存者偏差"（选的是范数最大的行——对 L2 影响最大？）
  E. V0 与 V0-defect 的逐题配对分析（McNemar 式检验：退化是否显著）
  F. 生成完整性分析（截断率对比）
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DEFECT_LAYERS = [22, 25, 26, 28, 29, 31, 32, 35]
K = 512


def load_tensor(model_dir: Path, name: str) -> torch.Tensor:
    single = model_dir / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt", device="cpu") as f:
            return f.get_tensor(name)
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shard = idx["weight_map"][name]
    with safe_open(str(model_dir / shard), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def main() -> None:
    out: dict[str, object] = {}
    v0 = ROOT / "checkpoints/V0"
    defect = ROOT / "checkpoints/V0-defect"

    # ---------- A. zero_rows 在健康 V0 的分布 ----------
    print("=" * 60)
    print("A. zero_rows 指纹检查（健康 V0 全部 36 层 x 7 投影）")
    healthy_zero = 0
    total_tensors = 0
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        for L in range(36):
            name = (
                f"model.layers.{L}.mlp.{proj}.weight"
                if proj in ("gate_proj", "up_proj", "down_proj")
                else f"model.layers.{L}.self_attn.{proj}.weight"
            )
            t = load_tensor(v0, name)
            zr = int((t.abs().sum(dim=1) == 0).sum()) if t.dim() == 2 else -1
            total_tensors += 1
            if zr > 0:
                healthy_zero += 1
                print(f"  !! 健康模型 {name}: zero_rows={zr}")
    print(f"  健康 V0: {healthy_zero}/{total_tensors} 个张量有零行")
    out["A_healthy_zero_rows"] = {"tensors_with_zero_rows": healthy_zero, "total": total_tensors}

    # ---------- B. z-score-only 召回率（从 Agent 可见统计量里剔除 zero_rows） ----------
    print("=" * 60)
    print("B. z-score-only（mean/std/l2，无 zero_rows）：能否发现病灶？")
    truth = {f"model.layers.{L}.mlp.up_proj.weight" for L in DEFECT_LAYERS}
    flagged = set()
    detail = []
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        rows = []
        for L in range(36):
            name = (
                f"model.layers.{L}.mlp.{proj}.weight"
                if proj in ("gate_proj", "up_proj", "down_proj")
                else f"model.layers.{L}.self_attn.{proj}.weight"
            )
            t = load_tensor(defect, name).float()
            rows.append({"layer": L, "name": name, "mean": t.mean().item(), "std": t.std().item(), "l2": t.norm().item()})
        import statistics as st

        for metric in ("mean", "std", "l2"):
            vals = [r[metric] for r in rows]
            mu, sd = st.mean(vals), st.stdev(vals)
            for r in rows:
                z = (r[metric] - mu) / sd if sd > 0 else 0.0
                if abs(z) > 2.5:
                    flagged.add(r["name"])
                    detail.append(f"{proj} L{r['layer']} {metric} z={z:+.2f}")
    hit = flagged & truth
    print(f"  z-score-only 标记 {len(flagged)} 张量，命中病灶 {len(hit)}/8")
    print(f"  明细: {sorted(detail)}")
    out["B_zscore_only"] = {"flagged": len(flagged), "hits": len(hit), "detail": detail}

    # ---------- C. 病灶张量的统计可见性量化 ----------
    print("=" * 60)
    print("C. 病灶层 up_proj 统计变化（V0 -> defect）")
    vis = []
    for L in DEFECT_LAYERS:
        name = f"model.layers.{L}.mlp.up_proj.weight"
        t0 = load_tensor(v0, name).float()
        t1 = load_tensor(defect, name).float()
        l2_0, l2_1 = t0.norm().item(), t1.norm().item()
        std_0, std_1 = t0.std().item(), t1.std().item()
        # 理论：置零 top-512 行范数应降
        row_norms = t0.norm(dim=1)
        top = torch.argsort(row_norms, descending=True)[:K]
        expected_l2 = (t0.norm() ** 2 - (row_norms[top] ** 2).sum()).sqrt().item()
        vis.append(
            {
                "layer": L,
                "l2_drop_pct": (l2_0 - l2_1) / l2_0 * 100,
                "std_drop_pct": (std_0 - std_1) / std_0 * 100,
                "expected_l2_drop_pct": (l2_0 - expected_l2) / l2_0 * 100,
            }
        )
        print(f"  L{L}: L2 -{vis[-1]['l2_drop_pct']:.2f}% (理论 -{vis[-1]['expected_l2_drop_pct']:.2f}%), std -{vis[-1]['std_drop_pct']:.2f}%")
    out["C_visibility"] = vis

    # ---------- D. top-norm 行的能量占比（选范数最大的行意味着什么） ----------
    print("=" * 60)
    print("D. top-512 行范数占该层总 Frobenius 能量的比例")
    for L in DEFECT_LAYERS[:3]:
        name = f"model.layers.{L}.mlp.up_proj.weight"
        t0 = load_tensor(v0, name).float()
        rn = t0.norm(dim=1) ** 2
        top = torch.sort(rn, descending=True).values[:K]
        print(f"  L{L}: top-512 行能量占比 {top.sum() / rn.sum() * 100:.2f}%（均匀基线 {K / t0.shape[0] * 100:.2f}%）")
        out.setdefault("D_top_energy", []).append({"layer": L, "pct": round((top.sum() / rn.sum() * 100).item(), 2)})

    # ---------- E. V0 vs defect 逐题配对（McNemar） ----------
    print("=" * 60)
    print("E. V0 vs defect 逐题配对分析")
    b = json.loads((ROOT / "results/baseline_V0.json").read_text(encoding="utf-8"))
    d = json.loads((ROOT / "results/eval_V0-defect.json").read_text(encoding="utf-8"))
    for split in ("optimization", "validation", "test"):
        bm = {r["uid"]: r["score"] for r in b["splits"][split]["results"]}
        dm = {r["uid"]: r["score"] for r in d["splits"][split]["results"]}
        uids = [u for u in bm if u in dm]
        n01 = sum(1 for u in uids if bm[u] == 1 and dm[u] == 0)  # 退化
        n10 = sum(1 for u in uids if bm[u] == 0 and dm[u] == 1)  # 改善
        # 精确二项检验（McNemar 精确版）
        n = n01 + n10
        if n > 0:
            p = sum(math.comb(n, i) for i in range(min(n01, n10) + 1)) / 2**n * 2
            p = min(p, 1.0)
        else:
            p = 1.0
        print(f"  {split}: n={len(uids)}  退化 {n01} / 改善 {n10}  (binomial p={p:.3f})")
        out.setdefault("E_mcnemar", {})[split] = {"n": len(uids), "worse": n01, "better": n10, "p": round(p, 4)}

    # ---------- F. 截断率对比（生成完整性） ----------
    print("=" * 60)
    print("F. 截断率（response 达到 max_new_tokens=768 的比例）")
    for split in ("optimization", "validation", "test"):
        for tag, data in (("V0", b), ("defect", d)):
            rs = data["splits"][split]["results"]
            trunc = sum(1 for r in rs if r.get("num_new_tokens", 0) >= 768)
            print(f"  {split} {tag}: {trunc}/{len(rs)} 截断")
            out.setdefault("F_truncation", {}).setdefault(split, {})[tag] = f"{trunc}/{len(rs)}"

    Path(ROOT / "results/audit_part1.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n[audit] -> results/audit_part1.json")


if __name__ == "__main__":
    main()
