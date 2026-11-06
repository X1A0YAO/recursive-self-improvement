"""审计：实验三期间工具是否真的返回了数据？

关键问题：tool_compare_from_model / tool_tensor_stats_from_model 在
device_map=auto + 磁盘卸载下访问 .data 是否抛 meta tensor 错误？
若是，实验三中模型收到的是错误串——结论需重写。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from model_utils import load_model  # noqa: E402
from agent.diagnosis_tools import tool_compare_from_model, tool_tensor_stats_from_model  # noqa: E402

DEFECT = ROOT / "checkpoints/p2/cascade3_activity_zero_512"


def main() -> None:
    model, tok = load_model(str(DEFECT))
    model.eval()

    # 1. 参数设备普查
    devices: dict[str, int] = {}
    meta_params = []
    for n, p in model.named_parameters():
        d = str(p.device)
        devices[d] = devices.get(d, 0) + 1
        if p.device.type == "meta":
            meta_params.append(n)
    print("[audit] parameter devices:", devices)
    print(f"[audit] meta params: {len(meta_params)}")
    for n in meta_params[:10]:
        print("   meta:", n)

    # 2. 逐层测试 up_proj 的可访问性（病灶投影）
    ok, fail = [], []
    for L in range(36):
        name = f"model.layers.{L}.mlp.up_proj.weight"
        p = dict(model.named_parameters())[name]
        if p.device.type == "meta":
            fail.append(L)
        else:
            ok.append(L)
    print(f"[audit] up_proj accessible layers: {len(ok)}, meta (inaccessible): {fail}")

    # 3. 重放 A/B 臂的实际调用（q_proj 全层 / 单层 stats）
    print("\n[audit] replay compare(q_proj, layers 0-35):")
    try:
        r = tool_compare_from_model(model, "q_proj", list(range(36)))
        print(f"  OK, {len(r)} chars; first 120: {r[:120]!r}")
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")

    print("\n[audit] replay tensor_stats(layer=22, q_proj):")
    try:
        r = tool_tensor_stats_from_model(model, 22, "q_proj")
        print(f"  OK, {len(r)} chars; first 120: {r[:120]!r}")
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")

    print("\n[audit] replay tensor_stats(layer=22, up_proj):  # 病灶层病灶投影")
    try:
        r = tool_tensor_stats_from_model(model, 22, "up_proj")
        print(f"  OK, {len(r)} chars")
        print("  " + r)
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")

    # 4. 若有 meta 层：测试 safetensors 直读方案（修复路径）
    if fail:
        print("\n[audit] fallback test: read layer via safetensors file directly")
        from safetensors import safe_open

        f = DEFECT / "model.safetensors"
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            t = sf.get_tensor(f"model.layers.{fail[0]}.mlp.up_proj.weight")
            print(f"  direct read of L{fail[0]} up_proj: shape={list(t.shape)}, mean={t.float().mean():.6f} OK")


if __name__ == "__main__":
    main()
