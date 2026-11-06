import sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from agent.b2_tools import compute_all_units
units = compute_all_units("checkpoints/p2/cascade3_activity_zero_512")
print("[parent] units ok", flush=True)

# 子进程重读（写成临时文件执行）
import subprocess, tempfile, json
from pathlib import Path
prog = """
import sys, json
from safetensors import safe_open
sf = safe_open("checkpoints/p2/cascade3_activity_zero_512/model.safetensors", framework="pt", device="cpu")
names = ["model.layers.22.mlp.up_proj.weight", "model.layers.0.self_attn.q_proj.weight", "model.layers.30.mlp.up_proj.weight"]
out = {}
for n in names:
    t = sf.get_tensor(n).float()
    out[n] = {"mean": t.mean().item(), "std": t.std().item(), "l2": t.norm().item(), "zero_rows": int((t.norm(dim=1)==0).sum())}
print(json.dumps(out))
"""
with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
    f.write(prog); tmp = f.name
r = subprocess.run([r".venv\Scripts\python.exe", "-X", "utf8", tmp], capture_output=True, text=True, check=True, timeout=600)
Path(tmp).unlink()
out = json.loads(r.stdout.strip().splitlines()[-1])
print("[parent] subprocess re-read ok:", list(out.keys()))
