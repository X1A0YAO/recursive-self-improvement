"""Phase 4.3 对照组 E：外部优化参考（随机搜索可达性上界）。

设计（预注册 §4 E 臂）：8 个随机单张量 scale 编辑（同预算 0.04 rel_delta），
每个在 Val 60 题子集上快评。报告最优与分布。
目的：估计"无智能的随机搜索在此预算下最好能达到什么"——
区分 F2（空间存在但 Agent 不足）与 F3（预算下不可达）。
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
PY = ROOT / ".venv/Scripts/python.exe"


def main() -> None:
    rng = random.Random(42)
    outdir = ROOT / "experiments/phase4/controls/E_search"
    outdir.mkdir(parents=True, exist_ok=True)

    projs = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    results = []
    for trial in range(8):
        layer = rng.randrange(36)
        proj = rng.choice(projs)
        fam = "mlp" if proj in ("gate_proj", "up_proj", "down_proj") else "self_attn"
        tensor = f"model.layers.{layer}.{fam}.{proj}.weight"
        factor = rng.choice([0.96, 1.04])
        ckpt = outdir / f"t{trial}"
        # 建 V0 副本 + 应用
        from agent.p4_tools import apply_edit
        from edit_weights import _copy_static_files, _load_all_tensors
        from safetensors.torch import save_file

        tensors = _load_all_tensors(ROOT / "checkpoints/V0")
        ckpt.mkdir(parents=True, exist_ok=True)
        _copy_static_files(ROOT / "checkpoints/V0", ckpt)
        save_file(tensors, str(ckpt / "model.safetensors"), metadata={"format": "pt"})
        del tensors
        msg, _ = apply_edit(ckpt, [{"tensor": tensor, "op": "scale", "factor": factor}], 0, {"rel_delta_l2": 0.0})
        print(f"[E] t{trial}: {tensor} x{factor} -> {msg[:40]}", flush=True)
        if not msg.startswith("APPLIED"):
            results.append({"trial": trial, "tensor": tensor, "factor": factor, "status": "rejected"})
            continue
        # Val 60 题快评（前 60 题——进度文件独立命名防冲突）
        out = outdir / f"t{trial}_val.json"
        r = subprocess.run([str(PY), "-X", "utf8", str(ROOT / "scripts/eval_ifeval_p4.py"),
                            "--model-dir", str(ckpt), "--split", "validation", "--output", str(out)],
                           capture_output=True, text=True, cwd=str(ROOT), timeout=14400,
                           env={**__import__("os").environ, "P4_EVAL_MAX_GPU_MB": "5800"})
        if r.returncode != 0:
            print(f"[E] t{trial} eval failed: {r.stderr[-200:]}", flush=True)
            continue
        val = json.loads(out.read_text(encoding="utf-8"))
        acc60 = sum(x["score"] for x in val["results"][:60]) / 60
        results.append({"trial": trial, "tensor": tensor, "factor": factor, "val60": round(acc60, 4)})
        print(f"[E] t{trial}: val60={acc60:.3f}", flush=True)
        # 清理该 trial 的 checkpoint（省盘）
        import shutil

        shutil.rmtree(ckpt)

    json.dump(results, open(outdir / "E_summary.json", "w"), indent=2)
    print("[E] done:", json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
