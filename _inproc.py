# 最小子进程复现：直接进程内调 _compute_all_units_inproc
import sys
sys.path.insert(0, ".")
from agent.b2_tools import _compute_all_units_inproc
print("[inproc] computing...", flush=True)
units = _compute_all_units_inproc("checkpoints/p2/cascade3_activity_zero_512", 36)
print(f"[inproc] OK {len(units)} units", flush=True)
