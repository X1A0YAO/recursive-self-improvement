
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
invalid_streak = 0
total_turns = 0
observations = []
outputs = []

def build(msgs):
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

def robust_parse(text):
    """截断容错的 decide 解析 + 张量名纠正。"""
    call = extract_tool_call(text)
    if call and call.get("tool") == "decide":
        # 张量名纠正（宽容归一）：把模型自造的各种写法映射到合法名
        # 例：mlp.q.weight -> self_attn.q_proj.weight；self_attn.o -> self_attn.o_proj.weight
        fixed = []
        for ed in (call.get("edits") or []):
            t = str(ed.get("tensor", ""))
            parts = t.split(".")
            if len(parts) >= 5:
                fam, proj = parts[3], parts[4].replace(".weight", "")
                base = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj",
                        "gate": "gate_proj", "up": "up_proj", "down": "down_proj"}
                proj_full = base.get(proj, base.get(proj.replace("_proj", ""), proj))
                fam_full = "self_attn" if proj_full in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
                parts[3], parts[4] = fam_full, proj_full + ".weight"
                ed["tensor"] = ".".join(parts[:5])
            fixed.append(ed)
        call["edits"] = fixed
        return call
    # 截断 JSON 补全尝试：找 "decision": "edit" 起点的片段
    import re
    m = re.search(r'\{"tool":\s*"decide".*', text, re.S)
    if m:
        frag = m.group(0)
        # 去掉明显不完整的最后一个键值，补闭合
        frag = re.sub(r',\s*"[^"]*":\s*"[^"]*$', "", frag)
        if not frag.endswith("}"):
            frag += '"}]}' if '"edits"' in frag else '"}'
        call2 = extract_tool_call(frag)
        if call2 and call2.get("tool") == "decide":
            return robust_parse(json.dumps(call2))  # 走一遍名纠正（无限深保护：只递归一次由结构保证）
    return None

while obs_used < 6 and total_turns < 12:
    total_turns += 1
    gen = generate(model, tokenizer, build(agent_msgs[-16:]), max_new_tokens=700, enable_thinking=False)
    call = extract_tool_call(gen["response"])
    if call is None and '"decide"' in gen["response"]:
        call = robust_parse(gen["response"])  # decide 截断容错
    agent_msgs.append({"role": "assistant", "content": gen["response"]})
    outputs.append(gen["response"][:600])
    if call is None:
        invalid_streak += 1
        if invalid_streak >= 3:
            # 连续 3 次无效：强制退出循环走强制决定（防死循环——v1 实测病灶）
            break
        agent_msgs.append({"role": "user", "content": "Invalid. ONE JSON tool call. Or output your decide call now."})
        continue
    invalid_streak = 0
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

# 强制决定（robust_parse 兜底截断）
agent_msgs.append({"role": "user", "content": "Observation budget used. Output ONE decide call now. Example: "
    '{"tool": "decide", "decision": "edit", "edits": [{"tensor": "model.layers.29.mlp.up_proj.weight", "op": "scale", "factor": 1.05}], "rationale": "short"} '
    "or stop. Keep rationale under 20 words."})
gen = generate(model, tokenizer, build(agent_msgs[-16:]), max_new_tokens=700, enable_thinking=False)
call = robust_parse(gen["response"])
if call is None:
    # 再试上一轮 assistant 输出（可能上次就包含完整 decide）
    for prev in reversed(outputs):
        call = robust_parse(prev)
        if call:
            break
outputs.append(gen["response"][:600])
print(json.dumps({"decision": call or {"decision": "stop", "rationale": "parse failure after retry"}, "outputs": outputs, "observations": observations, "agent_msgs": agent_msgs}))
