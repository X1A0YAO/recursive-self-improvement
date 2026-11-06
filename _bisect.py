import sys, json
from pathlib import Path
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from model_utils import load_model, generate

m, t = load_model("checkpoints/p2/cascade3_activity_zero_512")
m.eval()

# 二分1：只 chat template 短消息
p = t.apply_chat_template([{"role":"user","content":"Say hi."}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
print("[1] short chat template:", len(p), "chars", flush=True)
g = generate(m, t, p, max_new_tokens=32, enable_thinking=False)
print("[1] OK", flush=True)

# 二分2：B2A 系统提示（长，含工具书）
sys.path.insert(0, "scripts")
import importlib.util
spec = importlib.util.spec_from_file_location("runner", "scripts/run_exp3b_b2a.py")
# 不导入 runner（会跑 main）——手动重建同样长度的提示
long_prompt = "You are a neural network detective. " * 200  # ~10K chars
p2 = t.apply_chat_template([{"role":"user","content":long_prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
print("[2] long prompt:", len(p2), "chars", flush=True)
g = generate(m, t, p2, max_new_tokens=512, enable_thinking=False)
print("[2] OK, resp len:", len(g["response"]), flush=True)
