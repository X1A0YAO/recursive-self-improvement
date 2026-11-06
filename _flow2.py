import sys, json
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from model_utils import load_model, generate
from agent.b2_tools import compare_all_projections
from agent.diagnosis_tools import extract_tool_call
import importlib.util
spec = importlib.util.spec_from_file_location("runner", "scripts/run_exp3b_b2a.py")
mod = importlib.util.module_from_spec(spec); sys.modules["runner"] = mod
spec.loader.exec_module(mod)

manual = mod.TOOL_MANUAL_B2A.replace("{max_rounds}", "10")
system = mod.ARM_B2A_PROMPT.replace("{max_rounds}", "10").replace("{tool_manual}", manual)
m, t = load_model("checkpoints/p2/cascade3_activity_zero_512")
m.eval()

msgs = [{"role":"user","content":system}]
for rnd in range(4):
    p = t.apply_chat_template(msgs[-14:], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    print(f"[r{rnd}] prompt {len(p)} chars", flush=True)
    g = generate(m, t, p, max_new_tokens=512, enable_thinking=False)
    print(f"[r{rnd}] gen OK ({len(g['response'])} ch)", flush=True)
    call = extract_tool_call(g["response"])
    print(f"[r{rnd}] call={json.dumps(call)[:120] if call else None}", flush=True)
    msgs.append({"role":"assistant","content": g["response"]})
    if not call:
        msgs.append({"role":"user","content":"Invalid. ONE JSON object."}); continue
    if call.get("tool") == "compare_all_projections":
        text, rec = compare_all_projections("checkpoints/p2/cascade3_activity_zero_512")
        print(f"[r{rnd}] aggregate result {len(text)} chars", flush=True)
        msgs.append({"role":"user","content": f"TOOL RESULT:\n{text}\n\nRound {rnd+2}/10. Output ONE JSON tool call."})
    elif call.get("tool") == "tensor_stats":
        from agent.b2_tools import _SfReader, tensor_name
        reader = _SfReader.acquire("checkpoints/p2/cascade3_activity_zero_512")
        tt = reader.get(tensor_name(int(call["layer"]), call["proj"])).float()
        zr = int((tt.norm(dim=1)==0).sum())
        r = f"tensor: {tensor_name(int(call['layer']), call['proj'])}\nmean={tt.mean():.6f} std={tt.std():.6f}\nzero_rows={zr}"
        msgs.append({"role":"user","content": f"TOOL RESULT:\n{r}\n\nRound {rnd+2}/10. Output ONE JSON tool call."})
    elif call.get("tool") == "propose_edit":
        print("[r] PROPOSAL:", json.dumps(call)[:300], flush=True)
        break
print("[flow] completed without crash")
