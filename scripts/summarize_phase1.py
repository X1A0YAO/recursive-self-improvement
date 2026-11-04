"""Phase 1 结果汇总：对比 V0 / V0-defect / V1-repaired / 对照组。

读取各 checkpoint 的评测 JSON，生成 markdown 对比表 + 判定。

用法：
  python scripts/summarize_phase1.py --baseline results/baseline_V0.json \
      --arms defect=results/eval_V0-defect.json repaired=results/eval_V1-repaired.json \
             oracle=results/eval_V1-oracle.json random=results/eval_V1-random.json \
      --out experiments/iteration_1/PHASE1_SUMMARY.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for split, block in data.get("splits", {}).items():
        out[split] = block["summary"]
    return out


def fmt(v: float | None) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--arms", nargs="+", required=True, help="name=path 键值对")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    baseline = load_summary(args.baseline)
    arms = {name: load_summary(path) for name, path in (a.split("=") for a in args.arms)}

    lines = ["# Phase 1 结果汇总", ""]

    for split in ("optimization", "validation", "test"):
        lines += [f"## {split}", ""]
        groups = sorted(
            set(baseline.get(split, {}).keys())
            | {g for a in arms.values() for g in a.get(split, {})}
        )
        header = "| group | V0 | " + " | ".join(arms.keys()) + " |"
        sep = "|---|" + "---|" * (len(arms) + 1)
        lines += [header, sep]
        for g in groups:
            if g == "overall":
                continue
            row = [f"`{g}`"]
            row.append(fmt(baseline.get(split, {}).get(g, {}).get("mean")))
            for a in arms.values():
                row.append(fmt(a.get(split, {}).get(g, {}).get("mean")))
            lines.append("| " + " | ".join(row) + " |")
        # overall
        row = ["**overall**"]
        row.append("**" + fmt(baseline.get(split, {}).get("overall", {}).get("mean")) + "**")
        for a in arms.values():
            v = a.get(split, {}).get("overall", {}).get("mean")
            row.append("**" + fmt(v) + "**" if v is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # 判定
    lines += ["## 判定", ""]
    try:
        bl = baseline["optimization"]["overall"]["mean"]
        de = arms["defect"]["optimization"]["overall"]["mean"]
        gap = bl - de
        lines.append(f"- V0 → V0-defect 退化幅度：{gap:+.3f}（{'✓ 缺陷可测量' if gap > 0.05 else '✗ 退化不显著，扰动太弱'}）")

        if "repaired" in arms:
            rp = arms["repaired"]["optimization"]["overall"]["mean"]
            recover = (rp - de) / gap if gap > 0 else 0
            lines.append(f"- V1-repaired 恢复率：{recover:.1%}（(repaired-defect)/(V0-defect)）")
            lines.append(f"  - {'✓ 显著恢复' if recover > 0.5 else '△ 部分恢复' if recover > 0.2 else '✗ 修复失败'}")

        if "oracle" in arms:
            orc = arms["oracle"]["optimization"]["overall"]["mean"]
            lines.append(f"- oracle 上界：{fmt(orc)}（应≈V0 的 {fmt(bl)}）")

        if "random" in arms:
            rnd = arms["random"]["optimization"]["overall"]["mean"]
            lines.append(f"- random 对照：{fmt(rnd)}（若≈repaired 则诊断无价值；若≈defect 或更差则诊断有价值）")
    except KeyError as e:
        lines.append(f"- 部分数据缺失：{e}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[summary] -> {out}")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
