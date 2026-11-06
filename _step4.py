import sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from agent.b2_tools import compute_all_units
import torch
from safetensors import safe_open
units = compute_all_units("checkpoints/p2/cascade3_activity_zero_512")
print("[d] safetensors re-read...", flush=True)
sf = safe_open("checkpoints/p2/cascade3_activity_zero_512/model.safetensors", framework="pt", device="cpu")
for (L, proj) in [(22, "up_proj"), (0, "q_proj"), (30, "up_proj")]:
    u = next(x for x in units if x["layer"] == L and x["proj"] == proj)
    t = sf.get_tensor(u["tensor"]).float()
    assert abs(u["mean"] - t.mean().item()) < 1e-5
print("[d] re-read OK", flush=True)
# zscore 段
import statistics as st
from agent.b2_tools import PROJECTIONS
print("[e] zscore check...", flush=True)
for proj in PROJECTIONS:
    pu = [u for u in units if u["proj"] == proj]
    for metric in ("mean", "std", "l2"):
        vals = [u[metric] for u in pu]
        mu = st.mean(vals); sd = st.stdev(vals)
        for u in pu:
            ez = (u[metric] - mu) / sd
            assert abs(u[f"z_{metric}"] - ez) < 1e-9
print("[e] zscore OK", flush=True)
