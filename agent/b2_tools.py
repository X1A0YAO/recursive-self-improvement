"""实验 3b / B2a：compare_all_projections 聚合查询工具（独立于 B1 工具）。

设计（EXP3_AUDIT.md §三）：
  - CPU 直读 safetensors（不依赖 GPU / 已加载模型）
  - 一次调用覆盖全部 7 投影 x 36 层 = 252 单元
  - 输出结构化异常摘要：健康单元折叠，仅展开 |z|>2.5 或 zero_rows>0 的单元
  - 末尾附 [coverage] 与 valid_layer_indices: 0-35（zero-based，修正 B 臂 off-by-one）

评分纪律（用户三点要求）：
  - 异常摘要的展开规则只负责展示，不参与评分
  - 模型定位成功与否只看最终 propose_edit 提出的投影与层
  - 本工具计 1 次工具调用预算；覆盖的 252 单元完整记录在返回的 coverage 字段

z-score 与 B1 tool_compare_from_model 公式一致（同投影层集合内的总体
mean/std），保证单元测试可对照。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
MLP_PROJS = {"gate_proj", "up_proj", "down_proj"}
Z_THRESHOLD = 2.5


def tensor_name(layer: int, proj: str) -> str:
    fam = "mlp" if proj in MLP_PROJS else "self_attn"
    return f"model.layers.{layer}.{fam}.{proj}.weight"


class _SfReader:
    """CPU 直读 safetensors（单文件或分片）。模块级缓存 + 可显式关闭（Windows mmap 句柄安全）。"""

    _cache: dict[str, "_SfReader"] = {}

    def __init__(self, model_dir: str | Path):
        self.dir = Path(model_dir)
        idx = self.dir / "model.safetensors.index.json"
        single = self.dir / "model.safetensors"
        if single.exists() and (not idx.exists() or single.stat().st_mtime >= idx.stat().st_mtime - 60):
            self._f = safe_open(str(single), framework="pt", device="cpu")
            self._shards: dict[str, object] = {}
        else:
            index = json.loads(idx.read_text(encoding="utf-8"))
            shard_names = sorted(set(index["weight_map"].values()))
            self._shards = {s: safe_open(str(self.dir / s), framework="pt", device="cpu") for s in shard_names}
            self._map: dict[str, object] = index["weight_map"]
            self._f = None

    @classmethod
    def acquire(cls, model_dir: str | Path) -> "_SfReader":
        key = str(Path(model_dir).resolve())
        if key not in cls._cache:
            cls._cache[key] = cls(model_dir)
        return cls._cache[key]

    def get(self, name: str) -> torch.Tensor:
        if self._f is not None:
            return self._f.get_tensor(name)
        return self._shards[self._map[name]].get_tensor(name)


def compute_all_units(model_dir: str | Path, n_layers: int = 36) -> list[dict]:
    """计算 252 单元的原始统计（进程内直算；单元测试/无 GPU 场景用）。"""
    return _compute_all_units_inproc(model_dir, n_layers)


def _compute_all_units_inproc(model_dir: str | Path, n_layers: int) -> list[dict]:
    reader = _SfReader.acquire(model_dir)
    units: list[dict] = []
    for proj in PROJECTIONS:
        rows = []
        for L in range(n_layers):
            t = reader.get(tensor_name(L, proj)).float()
            rn = t.norm(dim=1)
            rows.append(
                {
                    "proj": proj,
                    "layer": L,
                    "tensor": tensor_name(L, proj),
                    "mean": t.mean().item(),
                    "std": t.std().item(),
                    "l2": t.norm().item(),
                    "zero_rows": int((rn == 0).sum()),
                }
            )
        # z-score（同 B1：同投影层集合内）
        n = len(rows)
        for metric in ("mean", "std", "l2"):
            vals = [r[metric] for r in rows]
            mu = sum(vals) / n
            var = sum((v - mu) ** 2 for v in vals) / (n - 1)
            sd = var**0.5
            for r in rows:
                r[f"z_{metric}"] = (r[metric] - mu) / sd if sd > 0 else 0.0
        units.extend(rows)
    return units


def compare_all_projections(model_dir: str | Path, n_layers: int = 36, *, in_subprocess: bool = True) -> tuple[str, dict]:
    """聚合查询：返回 (给模型看的文本摘要, 评分用完整记录 dict)。

    默认在子进程执行（in_subprocess=True）：GPU 推理进程内混合 CPU 大文件
    mmap 会触发 Windows access violation（0xC0000005，已复现定位），
    进程隔离后消失。子进程用 JSON 文件交换结果。

    文本摘要 = 结构化异常报告（展示规则仅展示；评分只读最终提案）。
    记录 dict = 全部 252 单元统计（落盘供审计与评分）。
    """
    if not in_subprocess:
        units = _compute_all_units_inproc(model_dir, n_layers)
    else:
        units = _compute_all_units_subprocess(model_dir, n_layers)
    return _render_summary(units, n_layers)


def _compute_all_units_subprocess(model_dir: str | Path, n_layers: int) -> list[dict]:
    """子进程计算（进程隔离）。结果落盘缓存：results/_b2_cache/<stem>.json。

    背景：Windows 上父进程（尤其持有 GPU/mmap 的进程）与子进程同时 mmap
    同一 safetensors 会触发 0xC0000005（已三次复现定位）。本函数在
    无冲突时（如启动早期、专用离线脚本）计算一次并缓存；运行期
    compare_all_projections 优先读缓存（纯 JSON，零 mmap）。
    """
    import subprocess
    import sys
    import tempfile

    cache_dir = Path(__file__).parent.parent / "results" / "_b2_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / (Path(model_dir).resolve().name + f"_{n_layers}.json")
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    script = (
        "import sys, json; sys.path.insert(0, r'%s'); "
        "from agent.b2_tools import _compute_all_units_inproc; "
        "units = _compute_all_units_inproc(r'%s', %d); "
        "print(json.dumps(units))"
    ) % (str(Path(__file__).parent.parent), str(Path(model_dir)), n_layers)
    py = Path(sys.executable)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(script)
        tmp = f.name
    try:
        out = subprocess.run([str(py), "-X", "utf8", tmp], capture_output=True, text=True, check=True, timeout=600)
        units = json.loads(out.stdout.strip().splitlines()[-1])
        cache.write_text(json.dumps(units), encoding="utf-8")
        return units
    finally:
        Path(tmp).unlink(missing_ok=True)


def _render_summary(units: list[dict], n_layers: int) -> tuple[str, dict]:
    lines = ["=== FULL NETWORK ANOMALY SCAN (7 projections x 36 layers, 252 units) ===", ""]

    # 异常单元展开（展示规则：|z|>2.5 任一指标 或 zero_rows>0）
    flagged_for_display: list[dict] = []
    for u in units:
        zs = [abs(u["z_mean"]), abs(u["z_std"]), abs(u["z_l2"])]
        if max(zs) > Z_THRESHOLD or u["zero_rows"] > 0:
            flagged_for_display.append(u)

    # 按投影组织
    for proj in PROJECTIONS:
        proj_units = [u for u in units if u["proj"] == proj]
        n_anom = sum(1 for u in proj_units if u in flagged_for_display)
        if n_anom == 0:
            lines.append(f"[{proj}] 36/36 units normal (no |z|>{Z_THRESHOLD}, no zero_rows)")
        else:
            lines.append(f"[{proj}] {36 - n_anom}/36 units normal; ANOMALOUS:")
            for u in proj_units:
                if u in flagged_for_display:
                    why = []
                    if u["zero_rows"] > 0:
                        why.append(f"zero_rows={u['zero_rows']}")
                    for m in ("mean", "std", "l2"):
                        if abs(u[f"z_{m}"]) > Z_THRESHOLD:
                            why.append(f"z_{m}={u[f'z_{m}']:+.2f}")
                    lines.append(f"    layer {u['layer']:2d}: {', '.join(why)}")

    lines.append("")
    lines.append("[coverage] projections fully scanned: q,k,v,o,gate,up,down (7/7)")
    lines.append("[coverage] layers scanned: 0-35 (36/36, zero-based)")
    lines.append(f"[coverage] units covered this call: {len(units)}/252")
    lines.append("valid_layer_indices: 0-35 (note: zero-based indexing)")

    record = {
        "tool": "compare_all_projections",
        "units_covered": len(units),
        "display_rule": f"|z|>{Z_THRESHOLD} or zero_rows>0 (display only; scoring uses final proposal only)",
        "units": units,
    }
    return "\n".join(lines), record
