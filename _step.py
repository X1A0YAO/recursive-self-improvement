import sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
print("[a] importing compute_all_units...", flush=True)
from agent.b2_tools import compute_all_units
print("[b] computing defect units...", flush=True)
units = compute_all_units("checkpoints/p2/cascade3_activity_zero_512")
print(f"[c] OK {len(units)}", flush=True)
