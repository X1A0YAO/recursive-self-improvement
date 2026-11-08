"""Phase 4.2：Agent RSI 主循环（Observe → Analyze → Propose → Apply → Evaluate → Decide）。

每轮结构（对抗提案悬崖 C4）：
  1. 给 Agent：task_report + 上轮反馈（若有）
  2. Agent 调观察工具（weight_stats / activation_probe），≤3 次/轮
  3. Agent 输出结构化决定：
     {"decision":"edit", "edits":[...], "rationale":"..."} 或
     {"decision":"stop", "rationale":"..."} 或
     {"decision":"explore", "query": "...", "rationale":"..."}
  4. edit → apply_edit → Val 评估（180 题）→ 反馈分数与决定 keep/rollback
  5. 循环 ≤6 轮

用法：
  python scripts/run_p4_agent.py --seed 0 --max-rounds 6
  python scripts/run_p4_agent.py --seed 0 --resume   # 从已有轨迹续跑
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from agent.diagnosis_tools import extract_tool_call  # noqa: E402
from model_utils import load_model  # noqa: E402

AGENT_PROMPT = """You are a Qwen3-4B model. You are being tested on INSTRUCTION FOLLOWING
(writing responses that obey formatting constraints: word counts, case rules, bullet formats,
keywords, titles, etc.). Your current accuracy: TRAIN {train_acc:.1%}, VALIDATION {val_acc:.1%}.

Your job: improve your own instruction-following ability by MODIFYING YOUR OWN WEIGHTS.
No gradients, no fine-tuning — you inspect your weights and decide direct numerical edits.

WHAT WE KNOW: instruction-following failures are usually behavioral habits (wrong casing,
wrong length, wrong markers), not knowledge gaps. Such habits often live in specific
mid-to-late MLP layers. Small weight nudges CAN change such habits (scaling a tensor by
0.9-1.1 rarely destroys general ability).

TOOLS (JSON calls only):
1. {{"tool": "task_report"}} — per-type failure statistics + failing train samples
2. {{"tool": "weight_stats"}} — cross-layer statistics for all 7 projections
3. {{"tool": "activation_probe", "layer": <int>, "proj": "<proj>"}} — activation stats
   on failing validation prompts
4. {{"tool": "decide", "decision": "edit", "edits": [{{"tensor": "model.layers.<L>.mlp.<P>.weight",
   "op": "scale", "factor": <float>}}], "rationale": "..."}} — apply edits
   (ops: scale | row_scale | zero_rows | add_low_rank; <=2 tensors/round)
5. {{"tool": "decide", "decision": "stop", "rationale": "..."}} — keep current weights, finish
6. {{"tool": "decide", "decision": "explore", "rationale": "..."}} — take another
   observation round without editing

PROCESS each round: call at most 3 observation tools, THEN one decide call.
Budget: {max_rounds} rounds total, cumulative weight change <= 0.05 relative L2.
After each edit round you will receive the new VALIDATION accuracy and may keep or
rollback (your next decide can be "edit" to replace, or "stop" to keep).

Honesty rule: base every claim on tool data you actually received.
"""


def run_round(model, tokenizer, agent, state: dict) -> dict:
    """一轮交互：返回轮记录。agent 是对话消息列表（累积）。"""
    from agent.p4_tools import EDIT_BUDGET, apply_edit, activation_probe, task_report, weight_stats

    round_log = {"round": state["round"], "observations": [], "decision": None, "val_before": state["val_acc"]}
    obs_used = 0

    # 反馈上一轮结果
    if state.get("last_feedback"):
        agent.append({"role": "user", "content": state["last_feedback"]})

    while obs_used < 6:  # 观察上限 6（历史教训：4 太紧导致无决定空间）
        prompt = tokenizer.apply_chat_template(agent[-16:], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        from model_utils import generate

        gen = generate(model, tokenizer, prompt, max_new_tokens=700, enable_thinking=False)
        call = extract_tool_call(gen["response"])
        agent.append({"role": "assistant", "content": gen["response"]})
        round_log.setdefault("outputs", []).append(gen["response"][:600])

        if call is None:
            agent.append({"role": "user", "content": "Invalid. ONE JSON tool call. If you have seen enough, output your decide call now."})
            continue
        tool = call.get("tool")

        if tool == "task_report":
            result = task_report(state["train_results"], state["val_results"])
            obs_used += 1
        elif tool == "weight_stats":
            # 从会话专属缓存读（GPU 运行期零 mmap）
            from agent.b2_tools import _cache_path_for, _render_summary

            cache = _cache_path_for(state["stats_cache_name"])
            units = json.loads(cache.read_text(encoding="utf-8"))
            result, _ = _render_summary(units, 36)
            obs_used += 1
        elif tool == "activation_probe":
            failing_prompts = state["fail_prompts"]
            result = activation_probe(model, tokenizer, failing_prompts, int(call["layer"]), str(call["proj"]))
            obs_used += 1
        elif tool == "decide":
            round_log["decision"] = call
            return round_log
        else:
            result = f"unknown tool: {tool}"
        agent.append({"role": "user", "content": f"TOOL RESULT:\n{result[:3000]}"})

    # 观察用尽仍无决定——强制决定（给大 token + 明确 schema 提示）
    agent.append({"role": "user", "content": "Observation budget used. Output ONE decide call now. Examples:\n"
                  '{"tool": "decide", "decision": "edit", "edits": [{"tensor": "model.layers.29.mlp.up_proj.weight", "op": "scale", "factor": 1.05}], "rationale": "..."}\n'
                  'or {"tool": "decide", "decision": "stop", "rationale": "..."}'})
    from model_utils import generate as _g

    prompt = tokenizer.apply_chat_template(agent[-16:], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    gen = _g(model, tokenizer, prompt, max_new_tokens=700, enable_thinking=False)
    call = extract_tool_call(gen["response"])
    if call is None:
        # 二次机会：截断的 JSON 尝试补全
        raw = gen["response"]
        if '"decision"' in raw:
            # 找到最大的 {...} 片段补闭合
            import re

            m = re.search(r'\{.*"decision"', raw, re.S)
            if m:
                frag = raw[m.start():]
                # 截到最后一个完整 key-value 再闭
                call = extract_tool_call(frag + '"}]}' if '"edits"' in frag else '"}')
    round_log["decision"] = call or {"decision": "stop", "rationale": "parse failure after retry"}
    return round_log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    workdir = ROOT / f"experiments/phase4/agent/seed{args.seed}"
    ckpt = workdir / "working"
    traj_path = workdir / "trajectory.json"

    if args.resume and traj_path.exists():
        state = json.loads(traj_path.read_text(encoding="utf-8"))
        state.setdefault("stats_cache_name", f"p4_seed{args.seed}")
        print(f"[agent] resume from round {state['round']}", flush=True)
    else:
        workdir.mkdir(parents=True, exist_ok=True)
        if ckpt.exists():
            shutil.rmtree(ckpt)
        # 工作副本 = V0 单文件化（复制分片并合并为单文件，供 apply_edit 就地更新）
        from edit_weights import _copy_static_files, _load_all_tensors
        from safetensors.torch import save_file

        tensors = _load_all_tensors(ROOT / "checkpoints/V0")
        ckpt.mkdir(parents=True)
        _copy_static_files(ROOT / "checkpoints/V0", ckpt)
        save_file(tensors, str(ckpt / "model.safetensors"), metadata={"format": "pt"})
        del tensors

        # 关键顺序：GPU 加载前为工作副本预计算 b2 统计缓存。
        # 用 seed 区分的缓存键（多 seed 工作副本同名 working 会互相覆盖）。
        # 直接 inproc 计算（父进程尚未加载 GPU/torch CUDA）。
        from agent.b2_tools import _cache_path_for
        import json as _json

        cache = _cache_path_for(f"p4_seed{args.seed}")
        if not cache.exists():
            from agent.b2_tools import _compute_all_units_inproc

            print("[agent] computing weight_stats cache inproc (pre-GPU)...", flush=True)
            units = _compute_all_units_inproc(ckpt, 36)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(_json.dumps(units), encoding="utf-8")
        print(f"[agent] cache ready: {cache.name}", flush=True)
        _stats_cache_name = f"p4_seed{args.seed}"

        # Train/Val 基线结果（V0 的——复用 baseline 产物）
        train_bl = json.loads((ROOT / "experiments/phase4/_train_eval/train_V0.json").read_text(encoding="utf-8"))
        val_bl = json.loads((ROOT / "experiments/phase4/baseline/val_V0.json").read_text(encoding="utf-8"))
        # 失败题 prompts（Val）
        val_data = [json.loads(x) for x in (ROOT / "benchmarks/ifeval_p4/validation/data.jsonl").read_text(encoding="utf-8").strip().splitlines()]
        fail_uids = {r["uid"] for r in val_bl["results"] if r["score"] < 1}
        fail_prompts = [d["prompt"] for d in val_data if d["uid"] in fail_uids][:6]

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
            "stats_cache_name": _stats_cache_name,
            "rounds_log": [],
            "last_feedback": None,
            "edits_applied": 0,
        }

    model, tokenizer = load_model(str(ckpt))
    model.eval()

    agent_msgs = [{"role": "user", "content": AGENT_PROMPT.format(train_acc=0.63, val_acc=state["val_acc"], max_rounds=args.max_rounds)}]

    # PYTHONUNBUFFERED 由外层设；逐轮写 trajectory 快照防长会话丢进度
    while state["round"] < args.max_rounds:
        print(f"\n[agent] === round {state['round']} (val={state['val_acc']:.3f}) ===", flush=True)
        rl = run_round(model, tokenizer, agent_msgs, state)
        dec = rl["decision"] or {}
        action = dec.get("decision")
        print(f"[agent] decision: {action} | {str(dec.get('rationale',''))[:100]}", flush=True)

        if action == "stop":
            state["last_feedback"] = None
            state["rounds_log"].append(rl)
            state["round"] += 1
            break
        if action == "explore":
            state["last_feedback"] = "Explore round logged. Continue with observation or decide."
            state["rounds_log"].append(rl)
            state["round"] += 1
            traj_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            continue
        if action == "edit":
            from agent.p4_tools import apply_edit, rollback

            msg, new_budget = apply_edit(ckpt, dec.get("edits", []), state["round"], state["budget_used"])
            print(f"[agent] apply_edit: {msg}", flush=True)
            rl["apply_result"] = msg
            if msg.startswith("APPLIED"):
                state["budget_used"] = new_budget
                state["edits_applied"] += 1
                # Val 重评（子进程跑评测避免 GPU 进程内存叠加——直接子进程调本脚本评估）
                import subprocess

                r = subprocess.run(
                    [str(ROOT / ".venv/Scripts/python.exe"), "-X", "utf8", str(ROOT / "scripts/eval_ifeval_p4.py"),
                     "--model-dir", str(ckpt), "--split", "validation",
                     "--output", str(workdir / f"val_r{state['round']}.json")],
                    capture_output=True, text=True, cwd=str(ROOT), timeout=7200,
                )
                val_new = json.loads((workdir / f"val_r{state['round']}.json").read_text(encoding="utf-8"))
                new_acc = val_new["accuracy"]
                delta = new_acc - state["val_acc"]
                rl["val_after"] = new_acc
                rl["val_delta"] = round(delta, 4)
                keep = new_acc >= state["val_acc"]
                if keep:
                    state["val_acc"] = new_acc
                    state["best_val"] = max(state["best_val"], new_acc)
                    state["val_results"] = val_new["results"]
                    fb = f"EDIT KEPT. Validation accuracy: {new_acc:.3f} ({delta:+.3f}). "
                else:
                    rb = rollback(ckpt, state["round"])
                    rl["rollback"] = True
                    fb = f"EDIT ROLLED BACK ({rb}). Validation accuracy dropped to {new_acc:.3f} ({delta:+.3f}). Weights restored. "
                # 更新失败 prompts
                fail_uids = {x["uid"] for x in state["val_results"] if x["score"] < 1}
                val_data = [json.loads(x) for x in (ROOT / "benchmarks/ifeval_p4/validation/data.jsonl").read_text(encoding="utf-8").strip().splitlines()]
                state["fail_prompts"] = [d["prompt"] for d in val_data if d["uid"] in fail_uids][:6]
                state["last_feedback"] = fb + (
                    f"Remaining budget: rel_delta {state['budget_used']['rel_delta_l2']:.4f}/0.05, "
                    f"rounds left {args.max_rounds - state['round'] - 1}."
                )
                print(f"[agent] val_after={new_acc:.3f} ({delta:+.3f}) keep={keep}", flush=True)

        state["rounds_log"].append(rl)
        state["round"] += 1
        traj_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    traj_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[agent] DONE. final val={state['val_acc']:.3f} (best {state['best_val']:.3f}), edits={state['edits_applied']}", flush=True)


if __name__ == "__main__":
    main()
