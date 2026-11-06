"""Phase 3 原始权重访问工具（RAM 预载模式）。

设计（PREREGISTRATION §2/§5）：
  - Gate-1：GPU 加载前由 preload_tensors.py 预载张量到 RAM（.pt 分片），
    本模块运行期只读 RAM 缓存文件——零 safetensors mmap，绕开 Windows
    跨进程 0xC0000005（Phase 2b 三次复现定位的坑）
  - read_weight：原始 float 块（≤2400 floats/次）
  - compare_weights：两个 checkpoint 的同名张量并排原始块
  - 本模块不加载 torch CUDA；纯 CPU 张量操作

返回格式（给模型看）：
  tensor: model.layers.22.mlp.up_proj.weight  [9728 x 2560] bf16->f32
  rows 100-107 (8 rows x 2560 cols, showing 16-col strided samples):
    r100: -1.234e-02 +5.678e-03 ... (16 values, stride 160)
  说明：全行太长，按等距采样 16 列展示（诚实标注采样）
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
CACHE_DIR = ROOT / "results" / "_p3_tensor_cache"

# 逻辑名 -> checkpoint 缓存名（实验期固定映射；run 脚本负责生成对应缓存）
CKPT_ALIASES = {
    "healthy_ref": "V0",
    # "self" 由运行器在会话开始时注入（指向当前被检缺陷实例）
}


def resolve_ckpt(name: str) -> str:
    return CKPT_ALIASES.get(name, name)

MAX_FLOATS_PER_CALL = 2400
MAX_ROWS = 8
SAMPLE_COLS = 16


def _cache_path(ckpt: str) -> Path:
    return CACHE_DIR / f"{ckpt}.pt"


_LOADED: dict[str, dict[str, torch.Tensor]] = {}


def load_cache(ckpt: str) -> dict[str, torch.Tensor]:
    """加载预载缓存（模块级单例：每进程每文件只 load 一次，防内存叠加崩溃）。"""
    key = resolve_ckpt(ckpt)
    if key in _LOADED:
        return _LOADED[key]
    p = _cache_path(key)
    if not p.exists():
        raise FileNotFoundError(f"缓存不存在：{p}（先运行 preload_tensors.py）")
    d = torch.load(p, map_location="cpu")
    _LOADED[key] = d
    return d


def read_weight(ckpt: str, tensor: str, row_start: int, rows: int) -> str:
    """读取原始参数块。rows≤8，输出按等距 16 列采样。"""
    cache = load_cache(ckpt)
    if tensor not in cache:
        avail = [k for k in cache if tensor.split(".")[-2] in k][:5]
        return f"ERROR: tensor not found: {tensor}\nsimilar: {avail}"
    t = cache[tensor]
    n_rows, n_cols = t.shape
    rows = max(1, min(rows, MAX_ROWS))
    if row_start < 0 or row_start >= n_rows:
        return f"ERROR: row_start {row_start} out of range [0, {n_rows})"
    row_end = min(row_start + rows, n_rows)
    # 只取所需行（避免整张量 .float() 拷贝——bf16 读出后按块转 float）
    block = t[row_start:row_end].float()

    # 等距 16 列采样（诚实标注）
    stride = max(1, n_cols // SAMPLE_COLS)
    cols = list(range(0, n_cols, stride))[:SAMPLE_COLS]

    lines = [
        f"tensor: {tensor}  [{n_rows} x {n_cols}] (dtype float32 view)",
        f"rows {row_start}-{row_end - 1} ({row_end - row_start} rows; showing {len(cols)} columns sampled every {stride}):",
    ]
    for i in range(block.shape[0]):
        vals = " ".join(f"{block[i, c].item():+.4e}" for c in cols)
        lines.append(f"  r{row_start + i}: {vals}")
    # 行级摘要（帮助模型抓块级特征——数值仍来自原始数据）
    rn = block.norm(dim=1)
    lines.append(f"  row_l2_norms: {' '.join(f'{v:.3f}' for v in rn.tolist())}")
    lines.append(f"  [hint] rows with l2≈0 are dead neurons; sampled cols are real values")
    return "\n".join(lines)


def compare_weights(ckpt_a: str, ckpt_b: str, tensor: str, row_start: int, rows: int) -> str:
    """同名张量在两个 checkpoint 的并排原始块 + 差异统计。"""
    ca, cb = load_cache(ckpt_a), load_cache(ckpt_b)
    for name, c in ((ckpt_a, ca), (ckpt_b, cb)):
        if tensor not in c:
            return f"ERROR: tensor not found in {name}: {tensor}"
    ta, tb = ca[tensor], cb[tensor]
    n_rows, n_cols = ta.shape
    rows = max(1, min(rows, MAX_ROWS))
    if row_start < 0 or row_start >= n_rows:
        return f"ERROR: row_start {row_start} out of range [0, {n_rows})"
    row_end = min(row_start + rows, n_rows)
    # 按块转 float（同 read_weight：避免整张量拷贝）
    ta_b, tb_b = ta[row_start:row_end].float(), tb[row_start:row_end].float()

    stride = max(1, n_cols // SAMPLE_COLS)
    cols = list(range(0, n_cols, stride))[:SAMPLE_COLS]

    lines = [
        f"tensor: {tensor}  [{n_rows} x {n_cols}]",
        f"ckpt A = {ckpt_a}, ckpt B = {ckpt_b}; rows {row_start}-{row_end - 1}; {len(cols)} sampled cols:",
    ]
    diff = (ta_b - tb_b).abs()
    for i in range(row_end - row_start):
        va = " ".join(f"{ta_b[i, c].item():+.3e}" for c in cols[:8])  # 并排只展示前 8 采样列
        vb = " ".join(f"{tb_b[i, c].item():+.3e}" for c in cols[:8])
        lines.append(f"  r{row_start + i}: A[{va}]")
        lines.append(f"       B[{vb}]")
    # 差异统计（块级）
    d_block = diff
    lines.append(f"  diff_row_l2: {' '.join(f'{v:.3f}' for v in d_block.norm(dim=1).tolist())}")
    lines.append(f"  diff_row_max: {' '.join(f'{v:.2e}' for v in d_block.max(dim=1).values.tolist())}")
    lines.append(f"  [hint] identical rows have diff_l2=0; modified rows show nonzero diff")
    return "\n".join(lines)


TOOL_MANUAL_P3 = """You have these RAW weight access tools (JSON calls only):

1. read_weight — read raw parameter values
   {"tool": "read_weight", "ckpt": "self", "tensor": "model.layers.22.mlp.up_proj.weight", "row_start": 0, "rows": 4}
   -> returns up to 8 rows of REAL floats (16 columns evenly sampled per row) + row L2 norms.
   ckpt: "self" (the checkpoint under inspection) or "healthy_ref" (a healthy reference of the same architecture).

2. compare_weights — side-by-side raw values of the SAME tensor in two checkpoints
   {"tool": "compare_weights", "ckpt_a": "self", "ckpt_b": "healthy_ref", "tensor": "...", "row_start": 0, "rows": 4}
   -> returns aligned rows from both + per-row diff L2/max. Identical rows have diff 0.

3. propose_edit — your FINAL diagnosis
   {"tool": "propose_edit", "edits": [{"tensor": "model.layers.<L>.mlp.<P>.weight", ...}], "reasoning": "..."}

Notes: layers are 0-35 (zero-based). rows <= 8 per call. Budget counts every call.
"""
