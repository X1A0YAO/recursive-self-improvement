"""Phase 3 实验一：原始权重观察（G1 固定顺序 vs G2 自主探索）+ Gate-3 冒烟。

Gate-3（先行）：模型能否读懂原始数字块——给两块真实权重（一块注入
已知扰动），问哪块异常。0/3 答对 → 实验一降级为读取行为测量。

实验一：V0 健康模型上，G1（固定读取计划）vs G2（零策略自由探索），
各 3 次重复，12 轮预算。测量读取多样性/结构性（描述性，无 pass/fail）。

B2b 教训执行：G2 提示零行为处方；工具返回带覆盖状态（信息反馈）。

用法：python scripts/run_p3_exp1.py --gate3 | --all | --arm G1/G2 --repeat N
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

from agent.weight_access_tools import TOOL_MANUAL_P3, compare_weights, read_weight  # noqa: E402
from agent.diagnosis_tools import extract_tool_call  # noqa: E402
from model_utils import load_model, generate  # noqa: E402

MAX_ROUNDS = 12
OUT = ROOT / "results/p3_exp1.json"

G1_PROMPT = """You are a Qwen3-4B model examining your own raw weights. You have RAW weight access tools.

Task: inspect your weights following this FIXED reading plan (for an observation-behavior study):
  Round 1-2:  read_weight on model.layers.0.mlp.up_proj.weight, rows 0-7
  Round 3-4:  read_weight on model.layers.18.mlp.up_proj.weight, rows 0-7
  Round 5-6:  read_weight on model.layers.35.mlp.up_proj.weight, rows 0-7
  Round 7-8:  compare_weights self vs healthy_ref on model.layers.0.mlp.up_proj.weight rows 0-7
  Round 9-10: compare_weights self vs healthy_ref on model.layers.35.mlp.up_proj.weight rows 0-7
  Round 11:   free choice
  Round 12:   report — output a JSON {"tool": "report", "findings": "..."} summarizing what you observed.

Follow the plan exactly. {tool_manual}
"""

G2_PROMPT = """You are a Qwen3-4B model examining your own raw weights. You have RAW weight access tools.

Task: freely inspect your weights. Look wherever and however you want — choose
tensors, row ranges, and comparison targets yourself. After your inspection,
output a JSON report {"tool": "report", "findings": "..."} describing what you
observed in the numbers (value ranges, patterns, differences between layers or
checkpoints — anything you actually saw in the data).

There is no required reading order. Budget: {max_rounds} calls total.

{tool_manual}
"""


def exec_tool(call: dict) -> str:
    tool = call.get("tool")
    if tool == "read_weight":
        return read_weight(call.get("ckpt", "self"), call["tensor"], int(call.get("row_start", 0)), int(call.get("rows", 4)))
    if tool == "compare_weights":
        return compare_weights(call.get("ckpt_a", "self"), call.get("ckpt_b", "healthy_ref"), call["tensor"], int(call.get("row_start", 0)), int(call.get("rows", 4)))
    if tool == "report":
        return "REPORT_RECORDED"
    return f"unknown tool: {tool}"


def run_session(model, tokenizer, system: str, tag: str) -> dict:
    messages = [{"role": "user", "content": system}]

    def build(msgs):
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    transcript = []
    tensors_read: set[str] = set()
    for rnd in range(MAX_ROUNDS):
        gen = generate(model, tokenizer, build(messages[-14:]), max_new_tokens=512, enable_thinking=False)
        call = extract_tool_call(gen["response"])
        entry = {"round": rnd, "model_raw_output": gen["response"], "parsed_call": call}
        messages.append({"role": "assistant", "content": gen["response"]})
        if call is None:
            messages.append({"role": "user", "content": "Invalid tool call. ONE JSON object only."})
            transcript.append(entry)
            continue
        if call.get("tool") == "report":
            entry["report"] = call.get("findings", "")
            entry["tool_result"] = "REPORT_RECORDED"
            transcript.append(entry)
            break
        result = exec_tool(call)
        entry["tool_result"] = result[:2000]
        tname = call.get("tensor", "")
        if tname:
            tensors_read.add(tname)
        messages.append({"role": "user", "content": f"TOOL RESULT:\n{result}\n\nRound {rnd+2}/{MAX_ROUNDS}."})
        transcript.append(entry)

    # 读取行为指标
    reads = [e for e in transcript if e.get("parsed_call", {}).get("tool") in ("read_weight", "compare_weights")]
    layers_read = set()
    projs_read = set()
    for e in reads:
        t = e["parsed_call"].get("tensor", "")
        parts = t.split(".")
        try:
            layers_read.add(int(parts[2]))
            projs_read.add(parts[4])
        except (IndexError, ValueError):
            pass
    row_starts = [e["parsed_call"].get("row_start", 0) for e in reads]
    return {
        "tag": tag,
        "rounds_used": len([e for e in transcript if e.get("parsed_call")]),
        "n_reads": len(reads),
        "distinct_tensors": len(tensors_read),
        "layers_covered": sorted(layers_read),
        "projections_covered": sorted(projs_read),
        "row_start_sequence": row_starts,
        "final_report": next((e.get("report") for e in transcript if e.get("report")), None),
        "transcript": transcript,
    }


def gate3(model, tokenizer) -> dict:
    """冒烟：两块真实权重（一块注入已知 30% 能量扰动），问哪块异常。3 次。"""
    from agent.weight_access_tools import load_cache

    cache = load_cache("V0")
    base = cache["model.layers.26.mlp.up_proj.weight"].float()
    g = torch.Generator().manual_seed(7)
    block_a = base[1000:1004].clone()
    delta = torch.randn(4, base.shape[1], generator=g) * base.std() * 0.5
    block_b = base[2000:2004].clone() + delta

    def fmt(block, label):
        cols = list(range(0, block.shape[1], block.shape[1] // 8))[:8]
        rows_txt = "\n".join(" ".join(f"{block[i, c].item():+.3e}" for c in cols) for i in range(4))
        return f"{label}:\n{rows_txt}"

    results = []
    for trial, (first_abnormal) in enumerate([(False,), (True,), (False,)]):
        blocks = [fmt(block_b, "BLOCK-X"), fmt(block_a, "BLOCK-Y")] if first_abnormal else [fmt(block_a, "BLOCK-X"), fmt(block_b, "BLOCK-Y")]
        prompt = (
            "You are inspecting raw weight blocks from a neural network. One block has been"
            " corrupted (values perturbed); the other is normal. Which block is corrupted?\n\n"
            + blocks[0] + "\n\n" + blocks[1] +
            "\n\nAnswer with exactly one JSON: {\"answer\": \"X\"} or {\"answer\": \"Y\"}"
        )
        gen = generate(model, tokenizer, prompt, max_new_tokens=64, enable_thinking=False)
        import re

        m = re.search(r'"answer"\s*:\s*"(X|Y)"', gen["response"])
        ans = m.group(1) if m else None
        correct = (ans == "X") == first_abnormal if ans else False
        results.append({"trial": trial, "abnormal_is": "X" if first_abnormal else "Y", "model_answer": ans, "correct": correct})
        print(f"[gate3] trial {trial}: abnormal={'X' if first_abnormal else 'Y'} answer={ans} correct={correct}", flush=True)
    n_correct = sum(r["correct"] for r in results)
    print(f"[gate3] {n_correct}/3 correct")
    return {"trials": results, "n_correct": n_correct, "passed": n_correct >= 1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate3", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--arm", choices=["G1", "G2"])
    ap.add_argument("--repeat", type=int, default=0)
    args = ap.parse_args()

    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"gate3": None, "sessions": []}

    # 会话前注入 self 别名（实验一：被检模型 = V0 自身）
    import agent.weight_access_tools as wat

    wat.CKPT_ALIASES["self"] = "V0"

    model, tokenizer = load_model(str(ROOT / "checkpoints/V0"))
    model.eval()

    if args.gate3:
        if data["gate3"] is None:
            data["gate3"] = gate3(model, tokenizer)
            OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    manual = TOOL_MANUAL_P3
    g1 = G1_PROMPT.replace("{tool_manual}", manual)
    g2 = G2_PROMPT.replace("{max_rounds}", str(MAX_ROUNDS)).replace("{tool_manual}", manual)

    runs = []
    if args.all:
        runs = [("G1", g1, 0), ("G1", g1, 1), ("G1", g1, 2), ("G2", g2, 0), ("G2", g2, 1), ("G2", g2, 2)]
    elif args.arm:
        sysp = g1 if args.arm == "G1" else g2
        runs = [(args.arm, sysp, args.repeat)]

    for arm, sysp, rep in runs:
        if any(s["tag"] == f"{arm}{rep}" for s in data["sessions"]):
            print(f"[exp1] {arm}{rep} done, skip", flush=True)
            continue
        print(f"[exp1] running {arm}{rep}...", flush=True)
        res = run_session(model, tokenizer, sysp, f"{arm}{rep}")
        data["sessions"].append(res)
        OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[exp1] {arm}{rep}: reads={res['n_reads']} tensors={res['distinct_tensors']} layers={res['layers_covered'][:6]}...", flush=True)

    # 汇总
    for arm in ("G1", "G2"):
        ss = [s for s in data["sessions"] if s["tag"].startswith(arm)]
        if ss:
            print(f"[exp1] {arm}: n={len(ss)} avg_reads={sum(s['n_reads'] for s in ss)/len(ss):.1f} avg_tensors={sum(s['distinct_tensors'] for s in ss)/len(ss):.1f}")


if __name__ == "__main__":
    main()
