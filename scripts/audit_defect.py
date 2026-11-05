"""实验二：可见性审计 + 跨实例泛化（预注册 §4）。

检测器（冻结规则，运行前锁定）：
  D1 z-score：每投影类型内跨层 |z|>2.5 于 mean/std/l2（任一指标触发即标记该张量）
  D2 行范数 KS：病灶层 vs 同投影全部层的两样本 KS（p < 0.05/36 Bonferroni）
  D3 零指纹：zero_rows > 0（仅记录，不计入 AUC）

AUC 协议：每实例 36 层 x 7 投影 = 252 单元；标签 = 是否真实病灶；
分数 = 各检测器最大异常分（D1: max|z|; D2: 1-KS_p; D3: binary）。

用法：
  python scripts/audit_defect.py --defect checkpoints/p2/<cell> --record _sealed/<cell>.json \
      [--label "标准缺陷"] --output results/defect_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from scipy import stats as sps
from safetensors import safe_open

ROOT = Path(__file__).parent.parent
PROJS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load_tensor(model_dir: Path, name: str) -> torch.Tensor:
    single = model_dir / "model.safetensors"
    idx_file = model_dir / "model.safetensors.index.json"
    if single.exists():
        with safe_open(str(single), framework="pt", device="cpu") as f:
            return f.get_tensor(name)
    idx = json.loads(idx_file.read_text(encoding="utf-8"))
    with safe_open(str(model_dir / idx["weight_map"][name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def tensor_name(L: int, proj: str) -> str:
    fam = "mlp" if proj in ("gate_proj", "up_proj", "down_proj") else "self_attn"
    return f"model.layers.{L}.{fam}.{proj}.weight"


def audit_instance(defect_dir: Path, truth_names: set[str]) -> dict:
    """返回 252 单元的 D1/D2/D3 分数与标签 + AUC。"""
    units = []
    for proj in PROJS:
        rows = []
        for L in range(36):
            t = load_tensor(defect_dir, tensor_name(L, proj)).float()
            rn = t.norm(dim=1) if t.dim() == 2 else None
            rows.append(
                {
                    "name": tensor_name(L, proj),
                    "layer": L,
                    "mean": t.mean().item(),
                    "std": t.std().item(),
                    "l2": t.norm().item(),
                    "zero_rows": int((rn == 0).sum()) if rn is not None else 0,
                    "row_norms": rn,
                }
            )
        # D1 z-score
        import statistics as st

        for metric in ("mean", "std", "l2"):
            vals = [r[metric] for r in rows]
            mu, sd = st.mean(vals), st.stdev(vals)
            for r in rows:
                z = abs((r[metric] - mu) / sd) if sd > 0 else 0
                r[f"z_{metric}"] = z
        # D2 KS vs 同投影其余层（Bonferroni alpha=0.05/36）
        pooled = torch.cat([r["row_norms"] for r in rows])
        for r in rows:
            others = torch.cat([x["row_norms"] for x in rows if x is not r])
            ks = sps.ks_2samp(r["row_norms"].numpy(), others.numpy())
            r["ks_p"] = float(ks.pvalue)
        units.extend(rows)

    # 标签与分数
    labels = [1 if u["name"] in truth_names else 0 for u in units]
    d1_score = [max(u["z_mean"], u["z_std"], u["z_l2"]) for u in units]
    d2_score = [1.0 - u["ks_p"] for u in units]
    d3_flag = [1 if u["zero_rows"] > 0 else 0 for u in units]

    def auc(scores: list[float], labs: list[int]) -> float:
        pos = [s for s, l in zip(scores, labs) if l == 1]
        neg = [s for s, l in zip(scores, labs) if l == 0]
        if not pos or not neg:
            return float("nan")
        wins = ties = 0
        for p in pos:
            for n in neg:
                if p > n:
                    wins += 1
                elif p == n:
                    ties += 1
        return (wins + 0.5 * ties) / (len(pos) * len(neg))

    return {
        "auc_d1": round(auc(d1_score, labels), 4),
        "auc_d2": round(auc(d2_score, labels), 4),
        "d3_detected": sum(1 for f, l in zip(d3_flag, labels) if f == 1 and l == 1),
        "d3_truth": sum(labels),
        "d3_false_positive": sum(1 for f, l in zip(d3_flag, labels) if f == 1 and l == 0),
        "d1_flagged": sum(1 for s in d1_score if s > 2.5),
        "d2_flagged": sum(1 for s in d2_score if s > 1 - 0.05 / 36),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--defect", required=True)
    ap.add_argument("--record", required=True, help="密封记录 json（含 tensor_names/rows）")
    ap.add_argument("--label", default="")
    ap.add_argument("--output", default="results/defect_audit.json")
    ap.add_argument("--healthy-check", action="store_true", help="附带跑健康 V0 假阳性基线")
    args = ap.parse_args()

    rec = json.loads(Path(args.record).read_text(encoding="utf-8"))
    truth = set(rec["tensor_names"]) if "tensor_names" in rec else set(rec["rows"].keys())
    print(f"[audit2] truth tensors: {sorted(truth)[:3]}... ({len(truth)})")

    result = audit_instance(Path(args.defect), truth)
    result["label"] = args.label or Path(args.defect).name
    print(json.dumps(result, indent=2, ensure_ascii=False))

    out_path = Path(args.output)
    existing = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {"instances": []}
    existing["instances"].append(result)
    out_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[audit2] -> {out_path}")


if __name__ == "__main__":
    main()
