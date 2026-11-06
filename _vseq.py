# 验证：GPU 加载前 load cascade3 缓存 → 再加载 GPU 模型 → 调 compare
import sys
sys.path.insert(0, ".")
from agent.weight_access_tools import CKPT_ALIASES, load_cache
CKPT_ALIASES["self"] = "cascade3_activity_zero_512"
print("[1] pre-loading caches (before GPU)...", flush=True)
load_cache("self"); load_cache("healthy_ref")
print("[2] caches in RAM", flush=True)
sys.path.insert(0, "scripts")
from model_utils import load_model
m, t = load_model("checkpoints/p2/cascade3_activity_zero_512")
m.eval()
print("[3] GPU model loaded", flush=True)
from agent.weight_access_tools import compare_weights
out = compare_weights("self", "healthy_ref", "model.layers.22.mlp.up_proj.weight", 0, 8)
print(f"[4] compare OK ({len(out)} chars); diff line present: {'diff_row_l2' in out}", flush=True)
