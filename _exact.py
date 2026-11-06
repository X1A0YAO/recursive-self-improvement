import sys, json
from pathlib import Path
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from model_utils import load_model, generate

# 从 runner 模块里取真实的提示常量（不执行 main）
import importlib.util
spec = importlib.util.spec_from_file_location("runner", "scripts/run_exp3b_b2a.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["runner"] = mod
try:
    spec.loader.exec_module(mod)
except SystemExit:
    pass

manual = mod.TOOL_MANUAL_B2A.replace("{max_rounds}", "10")
system = mod.ARM_B2A_PROMPT.replace("{max_rounds}", "10").replace("{tool_manual}", manual)
print(f"[repro] system prompt: {len(system)} chars", flush=True)

m, t = load_model("checkpoints/p2/cascade3_activity_zero_512")
m.eval()
msgs = [{"role":"user","content":system}]
p = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
print(f"[repro] templated: {len(p)} chars", flush=True)
g = generate(m, t, p, max_new_tokens=512, enable_thinking=False)
print(f"[repro] OK resp {len(g['response'])} chars: {g['response'][:150]!r}", flush=True)
