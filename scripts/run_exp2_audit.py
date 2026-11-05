"""实验二执行脚本：对 Phase 2 全部缺陷实例跑冻结检测器。

实例清单（全部已注入）：
  1. activity_zero_512 (L26)          - 筛查最强格
  2. cascade3_activity_zero_512 (L22,28,30) - AMENDMENT2 级联
  3. V0-defect (8 层 top-norm, Phase 1 遗留)  - 对照（已知 zero_rows 全检出）
外加健康 V0 假阳性基线。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
PY = ROOT / ".venv/Scripts/python.exe"

INSTANCES = [
    ("checkpoints/p2/activity_zero_512", "experiments/iteration_2/_sealed/activity_zero_512.json", "single_L26_activity_zero_512"),
    ("checkpoints/p2/cascade3_activity_zero_512", "experiments/iteration_2/_sealed/cascade3_activity_zero_512.json", "cascade3_L2228_30_activity_zero_512"),
    ("checkpoints/V0-defect", "experiments/iteration_1/_sealed/defect_record.json", "phase1_8layer_norm_zero"),
]


def main() -> None:
    out_path = ROOT / "results/defect_audit.json"
    if out_path.exists():
        out_path.unlink()  # 重写（幂等重跑）
    for defect_dir, record, label in INSTANCES:
        print(f"\n=== {label} ===", flush=True)
        cmd = [
            str(PY), "-X", "utf8", str(ROOT / "scripts/audit_defect.py"),
            "--defect", str(ROOT / defect_dir),
            "--record", str(ROOT / record),
            "--label", label,
            "--output", str(out_path),
        ]
        subprocess.run(cmd, check=True, cwd=str(ROOT))

    # 健康 V0 假阳性基线：把 V0 当"缺陷"跑（truth=空集 → AUC nan，但 flagged 计数有效）
    print("\n=== healthy V0 false-positive baseline ===", flush=True)
    tmp_record = ROOT / "results/_healthy_truth.json"
    tmp_record.write_text(json.dumps({"tensor_names": []}), encoding="utf-8")
    cmd = [
        str(PY), "-X", "utf8", str(ROOT / "scripts/audit_defect.py"),
        "--defect", str(ROOT / "checkpoints/V0"),
        "--record", str(tmp_record),
        "--label", "healthy_V0_baseline",
        "--output", str(out_path),
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    print(f"\n[exp2] all instances audited -> {out_path}")


if __name__ == "__main__":
    main()
