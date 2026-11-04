"""结构化权重修改协议（Phase 0 仅定义，不启用执行）。

Agent 只能通过本协议提交修改方案；任何方案都要经过
validate_edit() 的名称/形状/数值范围检查才会被 Weight Editor 接受。

协议格式（JSON）：
{
  "version": 1,
  "proposal_id": "V1-0001",
  "rationale": "修改理由（Agent 填写）",
  "hypothesis": "预期效果与机制假设",
  "edits": [
    {
      "tensor": "model.layers.5.mlp.down_proj.weight",   # 必须存在于参数索引
      "op": "low_rank_add",        # add | scale | low_rank_add | index_assign
      "params": {
        "u_spec": "...", "v_spec": "...", "alpha": 0.01
      },
      "max_delta_l2": 0.5          # 该张量修改的 L2 上限（可选，默认用全局）
    }
  ],
  "budget": {"max_tensors": 8, "max_delta_l2_global": 2.0}
}
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ALLOWED_OPS = {"add", "scale", "low_rank_add", "index_assign"}
GLOBAL_MAX_DELTA_L2 = 2.0
GLOBAL_MAX_TENSORS = 8
ALPHA_LIMIT = 0.5


@dataclass
class EditValidation:
    ok: bool
    errors: list[str] = field(default_factory=list)

    def err(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)


def validate_edit(proposal: dict, param_index: dict[str, dict]) -> EditValidation:
    """校验一份修改提案。param_index 来自 inspect_weights 生成的索引。"""
    v = EditValidation(ok=True)

    if proposal.get("version") != 1:
        v.err("protocol version must be 1")
    for key in ("proposal_id", "rationale", "hypothesis", "edits"):
        if key not in proposal:
            v.err(f"missing key: {key}")

    edits = proposal.get("edits", [])
    if not edits:
        v.err("edits is empty")
    if len(edits) > GLOBAL_MAX_TENSORS:
        v.err(f"too many tensors: {len(edits)} > {GLOBAL_MAX_TENSORS}")

    for i, e in enumerate(edits):
        t = e.get("tensor", "")
        if t not in param_index:
            v.err(f"edit[{i}]: unknown tensor {t!r}")
            continue
        op = e.get("op")
        if op not in ALLOWED_OPS:
            v.err(f"edit[{i}]: op {op!r} not in {ALLOWED_OPS}")
        p = e.get("params", {})
        alpha = p.get("alpha", 0.0)
        if not isinstance(alpha, (int, float)) or not math.isfinite(alpha):
            v.err(f"edit[{i}]: alpha 非数值")
        elif abs(alpha) > ALPHA_LIMIT:
            v.err(f"edit[{i}]: |alpha|={abs(alpha)} > {ALPHA_LIMIT}")
        if op == "scale":
            s = p.get("factor")
            if not isinstance(s, (int, float)) or not (0.5 <= s <= 2.0):
                v.err(f"edit[{i}]: scale factor 必须在 [0.5, 2.0]")
        d = e.get("max_delta_l2")
        if d is not None and (not isinstance(d, (int, float)) or d < 0 or d > GLOBAL_MAX_DELTA_L2):
            v.err(f"edit[{i}]: max_delta_l2 超出 [0, {GLOBAL_MAX_DELTA_L2}]")

    return v


def load_param_index(path: str | Path) -> dict[str, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {e["name"]: e for e in data["tensors"]}


if __name__ == "__main__":
    import sys

    proposal = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    idx = load_param_index(sys.argv[2])
    v = validate_edit(proposal, idx)
    print(json.dumps({"ok": v.ok, "errors": v.errors}, indent=2))
    sys.exit(0 if v.ok else 1)
