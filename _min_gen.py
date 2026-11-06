import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from model_utils import load_model, generate
m, t = load_model("checkpoints/p2/cascade3_activity_zero_512")
m.eval()
print("[min] model loaded, generating...", flush=True)
g = generate(m, t, "Hello, how are you?", max_new_tokens=32, enable_thinking=False)
print("[min] OK:", g["response"][:80])
