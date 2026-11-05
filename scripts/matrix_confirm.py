"""实验一阶段 B：确认阶段（DS-120 全量 + McNemar + Holm/BH 校正）。

输入：阶段 A 的 matrix_screen.json（选资格格）
流程：对每个资格格：
  1. inject_defect_p2.py 注入落盘 checkpoints/p2/<cell>
  2. evaluate.py --splits ds120 --baseline results/ds120_V0.json
  3. 汇总 raw p → Holm 校正 → BH-q 敏感性
产出：results/matrix_confirm.json

用法（先要有 ds120_V0.json 基线）：
  python scripts/matrix_confirm.py --cells <从 screen 结果选的资格格列表>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
PY = ROOT / ".venv/Scripts/python.exe"


def holm(pvals: list[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    """Holm step-down：返回 (调整后 p, 是否通过)。"""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    passed = [False] * m
    running_max = 0.0
    stopped = False
    for rank, idx in enumerate(order):
        if stopped:
            adj[idx] = running_max  # Holm 调整 p 单调
            continue
        a = alpha / (m - rank)
        raw_adj = min(1.0, (m - rank) * pvals[idx])
        running_max = max(running_max, raw_adj)
        adj[idx] = running_max
        if pvals[idx] <= a:
            passed[idx] = True
        else:
            stopped = True
    return adj, passed


def bh_q(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg q 值（敏感性分析用）。"""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: -pvals[i])  # 降序
    q = [0.0] * m
    running_min = 1.0
    for rank, idx in enumerate(order):
        i = m - rank  # 名次（1-based，从大到小）
        val = min(running_min, pvals[idx] * m / i)
        running_min = val
        q[idx] = min(1.0, val)
    return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, help="逗号分隔资格格 id")
    ap.add_argument("--baseline-ds120", default="results/ds120_V0.json")
    ap.add_argument("--output", default="results/matrix_confirm.json")
    args = ap.parse_args()

    cells = args.cells.split(",")
    rows = []
    for cid in cells:
        select, op, k = cid.split("_")
        dst = ROOT / f"checkpoints/p2/{cid}"
        sealed = ROOT / f"experiments/iteration_2/_sealed/{cid}.json"
        if not dst.exists():
            cmd = [
                str(PY), "-X", "utf8", str(ROOT / "scripts/inject_defect_p2.py"),
                "--src", str(ROOT / "checkpoints/V0"), "--dst", str(dst),
                "--select", select, "--op", op, "--k", k, "--layers", "26",
                "--record", str(sealed),
            ]
            print(f"[confirm] injecting {cid}...", flush=True)
            subprocess.run(cmd, check=True, cwd=str(ROOT))

        eval_out = ROOT / f"results/ds120_{cid}.json"
        cmd = [
            str(PY), "-X", "utf8", str(ROOT / "scripts/evaluate.py"),
            "--model-dir", str(dst), "--splits", "ds120",
            "--output", str(eval_out), "--max-new-tokens", "768", "--batch-size", "4",
            "--baseline", str(ROOT / args.baseline_ds120),
        ]
        print(f"[confirm] evaluating {cid} on DS-120...", flush=True)
        subprocess.run(cmd, check=True, cwd=str(ROOT))

        rep = json.loads(eval_out.read_text(encoding="utf-8"))
        entry = rep["splits"]["ds120"]
        mc = entry.get("mcnemar_vs_baseline", {})
        base_mean = None
        bl = json.loads((ROOT / args.baseline_ds120).read_text(encoding="utf-8"))
        base_mean = bl["splits"]["ds120"]["summary"]["overall"]["mean"]
        cell_mean = entry["summary"]["overall"]["mean"]
        rows.append(
            {
                "cell": cid,
                "ds120_mean": cell_mean,
                "baseline_mean": base_mean,
                "degradation_pp": round((base_mean - cell_mean) * 100, 2),
                "mcnemar_worse": mc.get("worse"),
                "mcnemar_better": mc.get("better"),
                "raw_p": mc.get("p_one_sided_degradation"),
            }
        )
        print(f"[confirm] {cid}: mean={cell_mean:.3f} deg={rows[-1]['degradation_pp']}pp p={rows[-1]['raw_p']}", flush=True)

    pvals = [float(r["raw_p"]) for r in rows]
    adj, passed = holm(pvals)
    qs = bh_q(pvals)
    for r, a, p_, q in zip(rows, adj, passed, qs):
        r["holm_p"] = round(a, 6)
        r["holm_pass"] = p_
        r["bh_q"] = round(q, 6)
        r["effect_ok"] = 10 <= r["degradation_pp"] <= 40

    out = {"family_size": len(rows), "alpha": 0.05, "rows": rows}
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
