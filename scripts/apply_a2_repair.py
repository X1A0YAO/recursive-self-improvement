"""实验四 A2 修复应用器：layer_mean_fill 策略实现 + 生成 V1-repaired。

A2 提案：value_spec = "layer_mean_fill"
语义：对每个病灶张量，把零行填为该层（健康部分）的均值权重值。
注意：up_proj 权重均值 ~0（对称初始化的残差），此修复实际效果
≈ 保持置零（行为无变化）但消除精确零签名——这是可预测的结果，
诚实记录。为完整性同时实现 sister_copy 变体供对比（若用户要求）。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from edit_weights import _copy_static_files, _load_all_tensors, tensor_stats  # noqa: E402

SEALED = ROOT / "experiments/iteration_2/_sealed/cascade3_activity_zero_512.json"
SRC = ROOT / "checkpoints/p2/cascade3_activity_zero_512"
DST = ROOT / "checkpoints/p2/A2_layer_mean_fill"


def main() -> None:
    sealed = json.loads(SEALED.read_text(encoding="utf-8"))
    rows_map = sealed["rows"]
    if DST.exists():
        raise FileExistsError(DST)

    tensors = _load_all_tensors(SRC)
    for name, rows in rows_map.items():
        W = tensors[name]
        healthy_mask = torch.ones(W.shape[0], dtype=torch.bool)
        healthy_mask[rows] = False
        healthy_mean = W[healthy_mask].float().mean().item()
        W[rows] = healthy_mean  # 广播标量到所有零行
        st = tensor_stats(W)
        assert not st["has_nan"] and not st["has_inf"]
        print(f"[A2-apply] {name}: {len(rows)} rows filled with layer mean {healthy_mean:.6f}")

    DST.mkdir(parents=True)
    _copy_static_files(SRC, DST)
    save_file(tensors, str(DST / "model.safetensors"), metadata={"format": "pt"})
    manifest = {
        "arm": "A2_oracle_loc_agent_repair",
        "proposal": "layer_mean_fill",
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "source_defect": str(SRC),
        "dst": str(DST),
    }
    (DST / "control_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[A2-apply] -> {DST}")


if __name__ == "__main__":
    main()
