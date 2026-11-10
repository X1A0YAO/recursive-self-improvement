"""E 组续跑：从 t1 断点恢复剩余 trials（2-7）。"""

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
    # 与 run_p4_control_E 相同的随机序列（rng 状态对齐：重放前 2 次抽样）
    rng = random.Random(42)
    projs = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    specs = []
    for _ in range(8):
        layer = rng.randrange(36)
        proj = rng.choice(projs)
        fam = "mlp" if proj in ("gate_proj", "up_proj", "down_proj") else "self_attn"
        factor = rng.choice([0.96, 1.04])
        specs.append((f"model.layers.{layer}.{fam}.{proj}.weight", factor))

    outdir = ROOT / "experiments/phase4/controls/E_search"
    # 载入已有结果
    results = []
    for t in range(8):
        val_path = outdir / f"t{t}_val.json"
        if val_path.exists():
            val = json.loads(val_path.read_text(encoding="utf-8"))
            acc60 = sum(x["score"] for x in val["results"][:60]) / 60
            results.append({"trial": t, "tensor": specs[t][0], "factor": specs[t][1], "val60": round(acc60, 4)})
            print(f"[E-resume] t{t}: val60={acc60:.3f} (cached)", flush=True)

    for t in range(8):
        if any(r["trial"] == t for r in results):
            continue
        tensor, factor = specs[t]
        ckpt = outdir / f"t{t}"
        if not (ckpt / "model.safetensors").exists():
            from agent.p4_tools import apply_edit
            from edit_weights import _copy_static_files, _load_all_tensors
            from safetensors.torch import save_file

            tensors = _load_all_tensors(ROOT / "checkpoints/V0")
            ckpt.mkdir(parents=True, exist_ok=True)
            _copy_static_files(ROOT / "checkpoints/V0", ckpt)
            save_file(tensors, str(ckpt / "model.safetensors"), metadata={"format": "pt"})
            del tensors
            msg, _ = apply_edit(ckpt, [{"tensor": tensor, "op": "scale", "factor": factor}], 0, {"rel_delta_l2": 0.0})
            print(f"[E-resume] t{t}: {tensor} x{factor} {msg[:40]}", flush=True)
            if not msg.startswith("APPLIED"):
                results.append({"trial": t, "tensor": tensor, "factor": factor, "status": "rejected"})
                continue
        out = outdir / f"t{t}_val.json"
        r = subprocess.run([str(PY), "-X", "utf8", str(ROOT / "scripts/eval_ifeval_p4.py"),
                            "--model-dir", str(ckpt), "--split", "validation", "--output", str(out)],
                           capture_output=True, text=True, cwd=str(ROOT), timeout=14400,
                           env={**__import__("os").environ, "P4_EVAL_MAX_GPU_MB": "5800"})
        if r.returncode != 0:
            print(f"[E-resume] t{t} eval failed", flush=True)
            continue
        val = json.loads(out.read_text(encoding="utf-8"))
        acc60 = sum(x["score"] for x in val["results"][:60]) / 60
        results.append({"trial": t, "tensor": tensor, "factor": factor, "val60": round(acc60, 4)})
        print(f"[E-resume] t{t}: val60={acc60:.3f}", flush=True)
        import shutil

        shutil.rmtree(ckpt, ignore_errors=True)

    json.dump(results, open(outdir / "E_summary.json", "w"), indent=2)
    print("[E-resume] done:", json.dumps(results, indent=1), flush=True)


if __name__ == "__main__":
    main()
