"""最终缺陷注入：多层级联 top-k 行病灶（layer 26/29/32 的 mlp.up_proj）。

用法：
  python scripts/inject_defect_v3.py --src checkpoints/V0 --dst checkpoints/V0-defect
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent))
from edit_weights import _copy_static_files, _load_all_tensors, tensor_stats  # noqa: E402

LAYERS = [22, 25, 28, 31, 26, 29, 32, 35]
PROJ = "mlp.up_proj"
K = 512
SEED = 178798052


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="checkpoints/V0")
    ap.add_argument("--dst", default="checkpoints/V0-defect")
    ap.add_argument("--record", default="experiments/iteration_1/_sealed/defect_record.json")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        raise FileExistsError(f"目标已存在: {dst}")

    tensors = _load_all_tensors(src)

    lesion = {}
    for L in LAYERS:
        name = f"model.layers.{L}.{PROJ}.weight"
        W = tensors[name]
        row_norms = W.float().norm(dim=1)
        top = torch.argsort(row_norms, descending=True)[:K]
        lesion[name] = {
            "rows": top.tolist(),
            "original_rows": W[top].clone(),  # 仅内存用于校验；落盘只存索引
        }
        W[top] = 0
        st = tensor_stats(W)
        assert not st["has_nan"] and not st["has_inf"]
        print(f"[lesion] {name}: {K} rows zeroed (of {W.shape[0]})", flush=True)

    dst.mkdir(parents=True)
    _copy_static_files(src, dst)
    save_file(tensors, str(dst / "model.safetensors"), metadata={"format": "pt"})

    record = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "type": "multi_layer_cascade_lesion",
        "source": str(src),
        "defect_checkpoint": str(dst),
        "layers": LAYERS,
        "proj": PROJ,
        "k_per_layer": K,
        "seed": SEED,
        "row_selection": "top_k_row_norm",
        "tensor_names": [f"model.layers.{L}.{PROJ}.weight" for L in LAYERS],
        "rows": {f"model.layers.{L}.{PROJ}.weight": lesion[f"model.layers.{L}.{PROJ}.weight"]["rows"] for L in LAYERS},
        "note": "oracle 修复 = 从 V0 重新加载这三层（或等价地恢复这些行）",
    }
    rp = Path(args.record)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    print(f"[lesion] defect -> {dst}")
    print(f"[lesion] sealed record -> {rp}")


if __name__ == "__main__":
    main()
