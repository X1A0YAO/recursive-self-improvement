"""Phase 4.3 对照组 C：Random Edit。

与 B 组（Agent）同预算的随机权重修改：
  - 同全局预算：rel_delta_l2 ≤ 0.05
  - 操作同分布：整张量 scale（Agent 实际用的 op）
  - 随机张量选择（seed 固定）与 Agent 提交过的张量类型同池（7 投影）
流程与 B 组完全一致：apply → Val 180 题重评 → keep/rollback。
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
PY = ROOT / ".venv/Scripts/python.exe"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    workdir = ROOT / "experiments/phase4/controls/C_random"
    ckpt = workdir / "working"

    if not (ckpt / "model.safetensors").exists():
        workdir.mkdir(parents=True, exist_ok=True)
        from agent.p4_tools import apply_edit
        from edit_weights import _copy_static_files, _load_all_tensors
        from safetensors.torch import save_file

        tensors = _load_all_tensors(ROOT / "checkpoints/V0")
        ckpt.mkdir(parents=True)
        _copy_static_files(ROOT / "checkpoints/V0", ckpt)
        save_file(tensors, str(ckpt / "model.safetensors"), metadata={"format": "pt"})
        del tensors

        # 随机编辑：1 张量 scale（与 Agent 的 L33 o_proj ×0.9 同类同规模：
        # rel_delta≈0.05 对应 factor≈0.9——Agent 的单编辑恰好贴满预算）
        projs = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        layer = rng.randrange(36)
        proj = rng.choice(projs)
        fam = "mlp" if proj in ("gate_proj", "up_proj", "down_proj") else "self_attn"
        tensor = f"model.layers.{layer}.{fam}.{proj}.weight"
        factor = rng.choice([0.96, 1.04])  # rel_delta≈0.04，留边界余量
        edits = [{"tensor": tensor, "op": "scale", "factor": factor}]
        print(f"[C] random edit: {tensor} x{factor}", flush=True)
        from agent.p4_tools import apply_edit as ae

        msg, budget = ae(ckpt, edits, 0, {"rel_delta_l2": 0.0})
        print(f"[C] apply: {msg}", flush=True)
        json.dump({"edits": edits, "apply": msg}, open(workdir / "manifest.json", "w"), indent=2)

    # Val 评估
    out = workdir / "val_C.json"
    r = subprocess.run([str(PY), "-X", "utf8", str(ROOT / "scripts/eval_ifeval_p4.py"),
                        "--model-dir", str(ckpt), "--split", "validation", "--output", str(out)],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=14400,
                       env={**__import__("os").environ, "P4_EVAL_MAX_GPU_MB": "5800"})
    val = json.loads(out.read_text(encoding="utf-8"))
    print(f"[C] val accuracy: {val['accuracy']}", flush=True)


if __name__ == "__main__":
    main()
