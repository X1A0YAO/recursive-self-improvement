import sys, subprocess, tempfile, json
from pathlib import Path
sys.path.insert(0, ".")
from agent.b2_tools import compute_all_units
units = compute_all_units("checkpoints/p2/cascade3_activity_zero_512")
print("[parent] units ok (mmap held open)", flush=True)
prog = """
import json
from safetensors.torch import load_file
# load_file 默认走内存加载路径（不是 persistent mmap）
t = load_file("checkpoints/p2/cascade3_activity_zero_512/model.safetensors", device="cpu")
n = "model.layers.22.mlp.up_proj.weight"
print(json.dumps({"ok": True, "mean": t[n].float().mean().item()}))
"""
with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
    f.write(prog); tmp = f.name
r = subprocess.run([r".venv\Scripts\python.exe", "-X", "utf8", tmp], capture_output=True, text=True, timeout=600)
Path(tmp).unlink()
print("[child exit]", r.returncode)
print("[child out]", r.stdout.strip()[:200])
print("[child err]", r.stderr.strip()[-300:] if r.stderr else "")
