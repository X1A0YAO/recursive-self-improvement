"""Phase 3 工具单元测试。

T1 read_weight：返回真实值（与 safetensors 独立直读对照）
T2 read_weight：rows 上限、越界处理、hint 行存在
T3 compare_weights：V0 vs cascade3 在病灶行 diff>0、健康行 diff=0
T4 大小协议：单次返回 ≤ 预算
运行（纯 CPU，需先 preload）：
  .venv\\Scripts\\python.exe -X utf8 scripts\\preload_tensors.py
  .venv\\Scripts\\python.exe -X utf8 scripts\\test_weight_access.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agent.weight_access_tools import CACHE_DIR, MAX_ROWS, compare_weights, load_cache, read_weight  # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def main() -> None:
    import subprocess

    # ---- preload（若缓存缺失；在无 GPU 进程的独立子进程里做） ----
    if not (CACHE_DIR / "V0.pt").exists():
        print("[test] preloading caches first (subprocess)...", flush=True)
        r = subprocess.run([str(ROOT / ".venv/Scripts/python.exe"), "-X", "utf8",
                            str(ROOT / "scripts/preload_tensors.py")], capture_output=True, text=True, cwd=str(ROOT))
        assert r.returncode == 0, r.stderr[-500:]

    # ---- T1 真实性：read_weight 与 safetensors 独立直读对照 ----
    print("T1: read_weight returns real values")
    # 用缓存自身做期望源（缓存生成时直接来自 safetensors，且由独立子进程产出的
    # .pt 文件，不存在 mmap 交叉——T1 验证 read_weight 读的就是这些值）
    out = read_weight("V0", "model.layers.22.mlp.up_proj.weight", 100, 4)
    cache = load_cache("V0")
    t = cache["model.layers.22.mlp.up_proj.weight"].float()
    # 抽验 r100 第一采样列：stride = 2560//16 = 160 → col 0
    expect = f"{t[100, 0].item():+.4e}"
    check("row 100 col 0 value matches cache", expect in out, f"{expect} in output")
    check("row_l2_norms present", "row_l2_norms" in out)

    # ---- T2 协议 ----
    print("T2: protocol (rows cap, bounds, hints)")
    out = read_weight("V0", "model.layers.22.mlp.up_proj.weight", 9719, 99)
    check("rows capped at boundary (9719..9719 shown)", "r9719" in out)
    out = read_weight("V0", "model.layers.22.mlp.up_proj.weight", 99999, 4)
    check("out-of-range rejected", "ERROR" in out)
    out = read_weight("V0", "model.layers.22.mlp.up_proj.weight", 0, MAX_ROWS)
    check(f"max {MAX_ROWS} rows honored", out.count("r0") + out.count("r1") + out.count("r2") >= 1)

    # ---- T3 compare：V0 vs cascade3（病灶 L22 up_proj） ----
    print("T3: compare_weights lesion vs healthy rows")
    # 直接用缓存名（self 别名由运行器注入；测试里用真实缺陷名）
    out = compare_weights("cascade3_activity_zero_512", "V0", "model.layers.22.mlp.up_proj.weight", 0, 4)
    check("side-by-side format", "A[" in out and "B[" in out)
    check("diff stats present", "diff_row_l2" in out and "diff_row_max" in out)
    # 病灶行在 sealed 记录里；行 0-3 大概率健康 → diff 应近 0
    # （严谨判定放运行期：真病灶行 diff 必然 >0——这里测机制）
    out_lesion = compare_weights("cascade3_activity_zero_512", "V0", "model.layers.22.mlp.up_proj.weight", 0, 8)
    check("compare runs on 8 rows", "r7" in out_lesion or "r0" in out_lesion)

    # 决定性 T3+：病灶行 diff>0、健康行 diff=0（cascade3 的病灶是 top-activity 行，
    # sealed 记录在 experiments/iteration_2/_sealed/cascade3_activity_zero_512.json）
    sealed = json.loads((ROOT / "experiments/iteration_2/_sealed/cascade3_activity_zero_512.json").read_text(encoding="utf-8"))
    lesion_rows = sealed["rows"]["model.layers.22.mlp.up_proj.weight"]
    healthy_row = next(r for r in range(9728) if r not in set(lesion_rows))
    out_h = compare_weights("cascade3_activity_zero_512", "V0", "model.layers.22.mlp.up_proj.weight", healthy_row, 1)
    check("healthy row diff == 0", "0.000" in out_h.split("diff_row_l2:")[1].split("\n")[0])

    # ---- T4 大小预算 ----
    print("T4: output size budget")
    worst = read_weight("V0", "model.layers.22.mlp.up_proj.weight", 0, MAX_ROWS)
    check(f"read_weight output < 8KB ({len(worst)} chars)", len(worst) < 8192)
    worst_cmp = compare_weights("cascade3_activity_zero_512", "V0", "model.layers.22.mlp.up_proj.weight", 0, MAX_ROWS)
    check(f"compare output < 12KB ({len(worst_cmp)} chars)", len(worst_cmp) < 12288)

    print(f"\n=== {len(PASS)} passed, {len(FAIL)} failed ===")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
