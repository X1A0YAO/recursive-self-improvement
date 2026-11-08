"""Phase 4.2 Agent RSI：轮间子进程隔离版。

教训（run_p4_agent.py 20h 实测）：单进程长会话在共享 GPU 上不可行——
CPU-offload 生成极慢 + 驱动长时劣化（CUDA unknown error）。
本版每轮生成放独立子进程（每轮重载模型 ~1min），主进程只做编排，
轮间状态落盘，天然断点续跑。

用法：python scripts/run_p4_agent_v2.py --seed 0 --max-rounds 6
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
PY = ROOT / ".venv/Scripts/python.exe"

# 每轮子进程脚本：加载模型 → 跑一轮 Agent 交互（观察+决定）→ 落盘轮产物
ROUND_SCRIPT = r'''
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
'''


def gpu_free_mb() -> int:
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(r.stdout.strip().splitlines()[0])


def wait_gpu(min_mb: int = 2900, max_wait_s: int = 14400) -> bool:
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        if gpu_free_mb() >= min_mb:
            return True
        print(f"[v2] GPU busy ({gpu_free_mb()}MB), wait 60s", flush=True)
        time.sleep(60)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rounds", type=int, default=6)
    args = ap.parse_args()

    workdir = ROOT / f"experiments/phase4/agent/seed{args.seed}"
    ckpt = workdir / "working"
    traj = workdir / "trajectory_v2.json"

    if traj.exists():
        state = json.loads(traj.read_text(encoding="utf-8"))
        print(f"[v2] resume round {state['round']}", flush=True)
    else:
        # 复用 v1 的初始化产物（工作副本+缓存已建好）
        v1_traj = workdir / "trajectory.json"
        if not (ckpt / "model.safetensors").exists():
            raise SystemExit("working checkpoint missing — run v1 once first (or copy)")
        from split_ifeval_p4 import SEED  # noqa: F401  (确认模块可导入)

        train_bl = json.loads((ROOT / "experiments/phase4/_train_eval/train_V0.json").read_text(encoding="utf-8"))
        val_bl = json.loads((ROOT / "experiments/phase4/baseline/val_V0.json").read_text(encoding="utf-8"))
        val_data = [json.loads(x) for x in (ROOT / "benchmarks/ifeval_p4/validation/data.jsonl").read_text(encoding="utf-8").strip().splitlines()]
        fail_uids = {r["uid"] for r in val_bl["results"] if r["score"] < 1}
        fail_prompts = [d["prompt"] for d in val_data if d["uid"] in fail_uids][:3]

        # Agent 系统提示
        from run_p4_agent import AGENT_PROMPT

        state = {
            "seed": args.seed,
            "round": 0,
            "val_acc": val_bl["accuracy"],
            "best_val": val_bl["accuracy"],
            "train_results": train_bl["results"],
            "val_results": val_bl["results"],
            "fail_prompts": fail_prompts,
            "budget_used": {"rel_delta_l2": 0.0},
            "working_dir": str(ckpt),
            "stats_cache_name": f"p4_seed{args.seed}",
            "rounds_log": [],
            "edits_applied": 0,
            "agent_msgs": [{"role": "user", "content": AGENT_PROMPT.format(train_acc=0.63, val_acc=val_bl["accuracy"], max_rounds=args.max_rounds)}],
            "last_feedback": None,
        }

    # 子进程脚本落盘（一次）
    script_path = workdir / "_round_worker.py"
    script_path.write_text(ROUND_SCRIPT, encoding="utf-8")

    while state["round"] < args.max_rounds:
        print(f"\n[v2] === round {state['round']} (val={state['val_acc']:.3f}) ===", flush=True)
        if state.get("last_feedback"):
            state["agent_msgs"].append({"role": "user", "content": state["last_feedback"]})

        # 轮状态喂给子进程
        tmp_state = workdir / "_round_state.json"
        tmp_state.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

        if not wait_gpu():
            print("[v2] GPU never freed; aborting (state saved)", flush=True)
            traj.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            return

        import os as _os
        r = subprocess.run([str(PY), "-X", "utf8", str(script_path), str(tmp_state)],
                           capture_output=True, text=True, cwd=str(ROOT), timeout=14400,
                           env={**_os.environ, "P4_MAX_GPU_MB": "2600"})
        # 解析子进程 stdout 最后一行 JSON
        try:
            out = json.loads(r.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            print(f"[v2] round worker failed: rc={r.returncode} stderr={r.stderr[-300:]}", flush=True)
            traj.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            return
        dec = out["decision"]
        state["agent_msgs"] = out["agent_msgs"]
        rl = {"round": state["round"], "decision": dec, "outputs": out["outputs"], "val_before": state["val_acc"]}
        print(f"[v2] decision: {dec.get('decision')} | {str(dec.get('rationale',''))[:100]}", flush=True)

        action = dec.get("decision")
        if action == "stop":
            state["rounds_log"].append(rl)
            state["round"] += 1
            break
        if action == "explore":
            state["last_feedback"] = "Explore logged. Continue observing or decide."
            state["rounds_log"].append(rl)
            state["round"] += 1
            traj.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            continue
        if action == "edit":
            from agent.p4_tools import apply_edit, rollback

            msg, new_budget = apply_edit(ckpt, dec.get("edits", []), state["round"], state["budget_used"])
            rl["apply_result"] = msg
            print(f"[v2] apply: {msg}", flush=True)
            if msg.startswith("APPLIED"):
                state["budget_used"] = new_budget
                state["edits_applied"] += 1
                # Val 重评（子进程）
                val_out = workdir / f"val_r{state['round']}.json"
                if not wait_gpu():
                    print("[v2] GPU blocked before val eval; aborting", flush=True)
                    break
                rr = subprocess.run([str(PY), "-X", "utf8", str(ROOT / "scripts/eval_ifeval_p4.py"),
                                     "--model-dir", str(ckpt), "--split", "validation", "--output", str(val_out)],
                                    capture_output=True, text=True, cwd=str(ROOT), timeout=14400,
                                    env={**__import__("os").environ, "P4_EVAL_MAX_GPU_MB": "2600"})
                if rr.returncode != 0:
                    print(f"[v2] val eval failed: {rr.stderr[-300:]}", flush=True)
                    break
                val_new = json.loads(val_out.read_text(encoding="utf-8"))
                new_acc = val_new["accuracy"]
                delta = new_acc - state["val_acc"]
                rl["val_after"] = new_acc
                rl["val_delta"] = round(delta, 4)
                if new_acc >= state["val_acc"]:
                    state["val_acc"] = new_acc
                    state["best_val"] = max(state["best_val"], new_acc)
                    state["val_results"] = val_new["results"]
                    fb = f"EDIT KEPT. Val accuracy {new_acc:.3f} ({delta:+.3f})."
                else:
                    rollback(ckpt, state["round"])
                    rl["rollback"] = True
                    fb = f"EDIT ROLLED BACK. Val dropped to {new_acc:.3f} ({delta:+.3f}). Weights restored."
                # 更新失败 prompts（保留 3 条）
                fuids = {x["uid"] for x in state["val_results"] if x["score"] < 1}
                state["fail_prompts"] = [d["prompt"] for d in val_data if d["uid"] in fuids][:3]
                state["last_feedback"] = fb + f" Budget used {state['budget_used']['rel_delta_l2']:.4f}/0.05; rounds left {args.max_rounds - state['round'] - 1}."
                print(f"[v2] val_after={new_acc:.3f} ({delta:+.3f})", flush=True)

        state["rounds_log"].append(rl)
        state["round"] += 1
        traj.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    traj.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[v2] DONE val={state['val_acc']:.3f} best={state['best_val']:.3f} edits={state['edits_applied']}", flush=True)


if __name__ == "__main__":
    main()
