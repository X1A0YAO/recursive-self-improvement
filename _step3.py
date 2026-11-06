import sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from agent.b2_tools import compare_all_projections
print("[sub] calling subprocess version...", flush=True)
text, rec = compare_all_projections("checkpoints/p2/cascade3_activity_zero_512")
print(f"[sub] OK {len(text)} chars, {rec['units_covered']} units", flush=True)
