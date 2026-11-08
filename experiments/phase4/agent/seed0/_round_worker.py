
import json, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import torch
from model_utils import load_model, generate
from agent.diagnosis_tools import extract_tool_call
from agent.p4_tools import activation_probe, task_report
from agent.b2_tools import _cache_path_for, _render_summary

state = json.loads(open(sys.argv[1], encoding="utf-8").read())
import os
_mem = {0: int(os.environ.get("P4_MAX_GPU_MB", "6200")) * 1024 * 1024, "cpu": "24GiB"}
model, tokenizer = load_model(state["working_dir"], max_memory=_mem)
model.eval()

agent_msgs = state["agent_msgs"]
obs_used = 0
observations = []
outputs = []

def build(msgs):
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

while obs_used < 6:
    gen = generate(model, tokenizer, build(agent_msgs[-16:]), max_new_tokens=700, enable_thinking=False)
    call = extract_tool_call(gen["response"])
    agent_msgs.append({"role": "assistant", "content": gen["response"]})
    outputs.append(gen["response"][:600])
    if call is None:
        agent_msgs.append({"role": "user", "content": "Invalid. ONE JSON tool call. Or output your decide call now."})
        continue
    tool = call.get("tool")
    if tool == "task_report":
        result = task_report(state["train_results"], state["val_results"]); obs_used += 1
    elif tool == "weight_stats":
        units = json.loads(open(_cache_path_for(state["stats_cache_name"]), encoding="utf-8").read())
        result, _ = _render_summary(units, 36); obs_used += 1
    elif tool == "activation_probe":
        result = activation_probe(model, tokenizer, state["fail_prompts"], int(call["layer"]), str(call["proj"])); obs_used += 1
    elif tool == "decide":
        print(json.dumps({"decision": call, "outputs": outputs, "observations": observations, "agent_msgs": agent_msgs}))
        sys.exit(0)
    else:
        result = f"unknown tool: {tool}"
    observations.append(result[:2000])
    agent_msgs.append({"role": "user", "content": "TOOL RESULT:\n" + result[:3000]})

# 强制决定
agent_msgs.append({"role": "user", "content": "Observation budget used. Output ONE decide call now. Example: "
    '{"tool": "decide", "decision": "edit", "edits": [{"tensor": "model.layers.29.mlp.up_proj.weight", "op": "scale", "factor": 1.05}], "rationale": "..."} '
    "or stop."})
gen = generate(model, tokenizer, build(agent_msgs[-16:]), max_new_tokens=700, enable_thinking=False)
call = extract_tool_call(gen["response"])
outputs.append(gen["response"][:600])
print(json.dumps({"decision": call or {"decision": "stop", "rationale": "parse failure"}, "outputs": outputs, "observations": observations, "agent_msgs": agent_msgs}))
