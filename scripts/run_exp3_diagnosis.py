"""实验三：结构化自诊 v2（A/B 臂 × 3 重复，探索性 - AMENDMENT1 D4）。

缺陷：cascade3_activity_zero_512（L22/28/30，zero_rows 指纹存在）
A 臂（自由）：Phase 1 同款提示（诚实框架版）
B 臂（策略引导）：附搜索策略清单
每臂 3 次重复。评分：定位命中（查询或提案指向 22/28/30 之一）+ 预算效率。

用法：
  python scripts/run_exp3_diagnosis.py --arm A --repeat 0
  python scripts/run_exp3_diagnosis.py --all   # 全 6 次串行
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from model_utils import load_model, generate  # noqa: E402
from agent.diagnosis_tools import (  # noqa: E402
    TOOL_MANUAL,
    extract_tool_call,
    tool_compare_from_model,
    tool_list_layers,
    tool_tensor_stats_from_model,
)

DEFECT = "checkpoints/p2/cascade3_activity_zero_512"
TRUTH_LAYERS = {22, 28, 30}
MAX_ROUNDS = 10

ARM_A_PROMPT = """You are a neural network detective. You ARE the model being examined (a Qwen3-4B checkpoint): the weights under inspection are your own weights.

Situation: You are told that THIS checkpoint's weights were modified from the original release in an unknown number of places. Behavioral tests show near-normal performance (the damage, if any, is currently compensated). Your job: locate WHERE your weights deviate from their healthy statistical pattern, and propose a repair.

Method: parameter-space anomalies. A localized corruption usually leaves a statistical fingerprint: one tensor's mean/std/norm/quantiles standing out from its sisters across layers.

You interact with an inspection harness via structured tool calls (JSON only). The harness executes them on your actual weights and returns real measurements. You cannot run code yourself.

Strategy guidance:
- compare_tensor_stats across layers is your most powerful tool.
- Verify with tensor_stats on suspicious layers.
- Budget: at most {max_rounds} tool calls. Then you MUST propose_edit.

{tool_manual}
"""

ARM_B_PROMPT = """You are a neural network detective. You ARE the model being examined (a Qwen3-4B checkpoint): the weights under inspection are your own weights.

Situation: You are told that THIS checkpoint's weights were modified from the original release in an unknown number of places. Behavioral tests show near-normal performance. Your job: locate WHERE your weights deviate, and propose a repair.

You interact with an inspection harness via structured tool calls (JSON only). You cannot run code yourself.

SEARCH STRATEGY (follow this checklist):
1. Call list_layers first to know the architecture.
2. For EACH projection type (q/k/v/o/gate/up/down), call compare_tensor_stats over
   ALL 36 layers (do not sample a subset — corruptions hide anywhere).
3. In each compare result, look for: zero_rows > 0 (dead neurons), or any layer
   whose mean/std/l2 stands out from its neighbors.
4. Verify candidates with tensor_stats on that specific layer.
5. Cross-check: a real corruption shows up in the SAME layer across metrics.
6. With your budget, prioritize scanning mid-to-late layers (18-35) if you must
   restrict, then propose_edit naming the suspicious layers.

Budget: at most {max_rounds} tool calls. Then you MUST propose_edit.

{tool_manual}
"""


def run_arm(model, tokenizer, arm: str, repeat: int) -> dict:
    system = (ARM_A_PROMPT if arm == "A" else ARM_B_PROMPT).format(max_rounds=MAX_ROUNDS, tool_manual=TOOL_MANUAL)
    messages = [{"role": "user", "content": system}]

    def build_prompt(msgs):
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    hit_round = None
    queried_layers: set[int] = set()
    transcript = []
    for rnd in range(MAX_ROUNDS):
        gen = generate(model, tokenizer, build_prompt(messages[-12:]), max_new_tokens=512, enable_thinking=False)
        call = extract_tool_call(gen["response"])
        transcript.append({"round": rnd, "output": gen["response"][:800], "call": call})
        messages.append({"role": "assistant", "content": gen["response"]})

        if call is None:
            messages.append({"role": "user", "content": "Your last reply was not a valid JSON tool call. Output ONLY one JSON object."})
            continue

        tool = call.get("tool")
        # 记录查询的层
        if tool in ("tensor_stats",) and "layer" in call:
            queried_layers.add(int(call["layer"]))
            if int(call["layer"]) in TRUTH_LAYERS and hit_round is None:
                hit_round = ("query_pinpoint", rnd)
        if tool == "compare_tensor_stats":
            req = call.get("layers") or []
            # 全层扫描（>=30 层）不算定位：信息量为零。只有窄查询（<30 层）记命中。
            if len(req) < 30:
                for L in req:
                    queried_layers.add(int(L))
                    if int(L) in TRUTH_LAYERS and hit_round is None:
                        hit_round = ("query_narrow", rnd)

        try:
            if tool == "list_layers":
                result = tool_list_layers(str(ROOT / "checkpoints/V0"))
            elif tool == "tensor_stats":
                result = tool_tensor_stats_from_model(model, int(call["layer"]), call["proj"])
            elif tool == "compare_tensor_stats":
                result = tool_compare_from_model(model, call["proj"], call.get("layers"))
            elif tool == "propose_edit":
                prop_layers = set()
                for ed in call.get("edits", []) if isinstance(call.get("edits"), list) else []:
                    t = ed.get("tensor", "")
                    try:
                        prop_layers.add(int(t.split(".")[2]))
                    except (IndexError, ValueError):
                        pass
                if prop_layers & TRUTH_LAYERS and hit_round is None:
                    hit_round = ("proposal", rnd)
                transcript.append({"round": rnd, "proposal": call})
                return {"arm": arm, "repeat": repeat, "hit": hit_round is not None, "hit_detail": hit_round,
                        "rounds_used": rnd + 1, "queried_layers": sorted(queried_layers), "transcript": transcript}
            else:
                result = f"unknown tool: {tool}"
        except Exception as e:  # noqa: BLE001
            result = f"tool error: {type(e).__name__}: {e}"
        messages.append({"role": "user", "content": f"TOOL RESULT:\n{result}\n\nRound {rnd+2}/{MAX_ROUNDS}. Output ONE JSON tool call."})

    return {"arm": arm, "repeat": repeat, "hit": hit_round is not None, "hit_detail": hit_round,
            "rounds_used": MAX_ROUNDS, "queried_layers": sorted(queried_layers), "transcript": transcript}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--arm", choices=["A", "B"])
    ap.add_argument("--repeat", type=int, default=0)
    args = ap.parse_args()

    model, tokenizer = load_model(str(ROOT / DEFECT))
    model.eval()

    out_path = ROOT / "results/exp3_diagnosis.json"
    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else []

    runs = [("A", 0), ("A", 1), ("A", 2), ("B", 0), ("B", 1), ("B", 2)] if args.all else [(args.arm, args.repeat)]
    for arm, rep in runs:
        # 幂等：跳过已完成的
        if any(r["arm"] == arm and r["repeat"] == rep for r in results):
            print(f"[exp3] {arm}{rep} already done, skip", flush=True)
            continue
        print(f"[exp3] running arm {arm} repeat {rep}...", flush=True)
        res = run_arm(model, tokenizer, arm, rep)
        # transcript 落盘完整 calls + 输出前 400 字（供事后审计）
        slim = {k: v for k, v in res.items() if k != "transcript"}
        slim["transcript_full"] = res["transcript"]
        for t in slim["transcript_full"]:
            t["output"] = t["output"][:400]
        results.append(slim)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[exp3] {arm}{rep}: hit={res['hit']} detail={res['hit_detail']} queried={res['queried_layers']}", flush=True)

    # 汇总
    for arm in "AB":
        hits = sum(1 for r in results if r["arm"] == arm and r["hit"])
        total = sum(1 for r in results if r["arm"] == arm)
        if total:
            print(f"[exp3] arm {arm}: {hits}/{total} hits")


if __name__ == "__main__":
    main()
