"""CPU 直读版审计：实验三的工具在当时应该返回什么。"""
import json
from pathlib import Path
from safetensors import safe_open
import statistics as st

D = Path("checkpoints/p2/cascade3_activity_zero_512")
sf = safe_open(str(D / "model.safetensors"), framework="pt", device="cpu")

def stats_of(name):
    t = sf.get_tensor(name).float()
    rn = t.norm(dim=1)
    return {
        "mean": t.mean().item(), "std": t.std().item(), "l2": t.norm().item(),
        "zero_rows": int((rn == 0).sum()),
    }

# 1. B 臂实际查询的：q_proj 全层 compare（病灶层 22/28/30 在 q_proj 上应该完全干净）
print("=== q_proj L0-35（B 臂看到的）===")
rows = []
for L in range(36):
    rows.append((L, stats_of(f"model.layers.{L}.self_attn.q_proj.weight")))
means = [r["mean"] for _, r in rows]
mu, sd = st.mean(means), st.stdev(means)
for L, r in rows:
    z = (r["mean"] - mu) / sd
    mark = " <<<" if L in (22, 28, 30) else ""
    if abs(z) > 1.5 or L in (22, 28, 30):
        print(f"  L{L:2d}: mean={r['mean']:+.5f} z={z:+.2f} zero_rows={r['zero_rows']}{mark}")

# 2. 病灶投影 up_proj 的同类 compare（B 臂从未查过）
print("\n=== up_proj L0-35（B 臂从未看到；病灶在此）===")
rows = []
for L in range(36):
    rows.append((L, stats_of(f"model.layers.{L}.mlp.up_proj.weight")))
for L, r in rows:
    mark = " <<< LESION" if L in (22, 28, 30) else ""
    print(f"  L{L:2d}: zero_rows={r['zero_rows']:4d}  std={r['std']:.5f}{mark}")
