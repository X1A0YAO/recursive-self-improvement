import sys, json
from pathlib import Path
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from model_utils import load_model, generate
from agent.b2_tools import compare_all_projections
from agent.diagnosis_tools import extract_tool_call

m, t = load_model("checkpoints/p2/cascade3_activity_zero_512")
m.eval()

manual = """You have tools:
1. {"tool": "compare_all_projections"}  # scan all 252 units
2. {"tool": "tensor_stats", "layer": 0, "proj": "q_proj"}
3. propose_edit to finish."""
system = f"You are a detective. First call compare_all_projections.\n\n{manual}"
msgs = [{"role":"user","content":system}]

for rnd in range(3):
    p = t.apply_chat_template(msgs[-14:], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    print(f"[round {rnd}] prompt {len(p)} chars", flush=True)
    g = generate(m, t, p, max_new_tokens=512, enable_thinking=False)
    print(f"[round {rnd}] gen OK, {len(g['response'])} chars", flush=True)
    call = extract_tool_call(g["response"])
    print(f"[round {rnd}] call: {json.dumps(call)[:100] if call else None}", flush=True)
    msgs.append({"role":"assistant","content": g["response"]})
    if call and call.get("tool") == "compare_all_projections":
        text, rec = compare_all_projections("checkpoints/p2/cascade3_activity_zero_512")
        msgs.append({"role":"user","content": f"TOOL RESULT:\n{text}\n\nRound {rnd+2}/10. Output ONE JSON tool call."})
    else:
        break
print("[done] full 2-round flow OK")
