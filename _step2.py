import sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from agent.b2_tools import compute_all_units
print("[v0] computing...", flush=True)
units = compute_all_units("checkpoints/V0")
print(f"[v0] OK {len(units)}", flush=True)
