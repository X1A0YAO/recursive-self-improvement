"""Phase 1 主实验：受控缺陷的自我诊断与修复。

流程：
  1. 用 V0-defect 跑 optimization 子集 -> 收集错误样本（行为证据）
  2. 自我诊断循环（diagnosis_tools）：模型看错误样本 + 工具说明书，
     逐轮生成工具调用，系统执行并回填 —— 全程记录
  3. 模型提交 propose_edit -> edit_protocol 校验 -> Weight Editor 执行
  4. Evaluator 复评 V1-repaired（optimization + validation + test）
  5. 报告：修复效果、副作用、诊断轨迹

关键约束：
  - Agent 的提示词中绝不包含 defect_record（密封信息）
  - 模型只能用 JSON 工具调用，不能执行代码
  - 每次实验的完整轨迹落盘，可复现

用法：
  python scripts/run_phase1.py --defect checkpoints/V0-defect --workdir experiments/iteration_1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_utils import generate, load_model  # noqa: E402


# ---------------------------------------------------------------------------
# 阶段 1：收集行为证据（错误样本）
# ---------------------------------------------------------------------------


def collect_failures(model, tokenizer, n: int = 12) -> list[dict]:
    """跑 optimization 里的样本，返回错误/最差的样本作为行为证据。"""
    from evaluate import load_split, score_one

    examples = load_split(Path("benchmarks/optimization/data.jsonl"))
    # 均衡抽样：math/logic/general 各来一些（instruction/repetition 评分慢）
    by_group: dict[str, list[dict]] = {}
    for ex in examples:
        by_group.setdefault(ex["group"], []).append(ex)
    picked: list[dict] = []
    for g in ("math", "logic", "general"):
        picked.extend(by_group.get(g, [])[: max(1, n // 3)])

    failures = []
    for ex in picked:
        gen = generate(
            model, tokenizer, ex["prompt"], max_new_tokens=512, enable_thinking=False
        )
        truncated = gen["num_new_tokens"] >= 512
        s = score_one(ex, gen["response"], truncated)
        failures.append(
            {
                "uid": ex["uid"],
                "group": ex["group"],
                "prompt": ex["prompt"][:500],
                "response": gen["response"][:800],
                "score": s["score"],
                "info": s.get("info", {}),
            }
        )
    return failures


# ---------------------------------------------------------------------------
# 阶段 2：自我诊断循环
# ---------------------------------------------------------------------------

DIAG_SYSTEM_PROMPT = """You are a neural network detective. You ARE the model being examined (a Qwen3-4B checkpoint): the weights under inspection are your own weights.

Situation: You are told that THIS checkpoint's weights were modified from the original release in an unknown number of places. Behavioral tests show near-normal performance (the damage, if any, is currently compensated). Your job: locate WHERE your weights deviate from their healthy statistical pattern, and propose a repair.

Method: parameter-space anomalies. A localized corruption usually leaves a statistical fingerprint: one tensor's mean/std/norm/quantiles standing out from its sisters across layers.

You interact with an inspection harness via structured tool calls (JSON only). The harness executes them on your actual weights and returns real measurements. You cannot run code yourself.

Strategy guidance:
- compare_tensor_stats across layers is your most powerful tool.
- Verify with tensor_stats on suspicious layers.
- Budget: at most {max_rounds} tool calls. Then you MUST propose_edit.

{tool_manual}

Behavioral test summary (for reference, near-normal):
{evidence}
"""


def run_diagnosis_loop(model, tokenizer, failures: list[dict], workdir: Path, max_rounds: int = 10, config_dir: str = "models/Qwen3-4B") -> tuple[dict, list[dict]]:
    args_dir_for_config = config_dir  # list_layers 需要一个含 config.json 的目录
    from agent.diagnosis_tools import (
        PROJ_CHOICES,
        TOOL_MANUAL,
        DiagnosisLog,
        extract_tool_call,
        tool_activation_probe,
        tool_compare_from_model,
        tool_list_layers,
        tool_tensor_stats_from_model,
    )

    log = DiagnosisLog()

    evidence = "\n\n".join(
        f"[{f['uid']} / {f['group']} / score={f['score']}]\nPROMPT: {f['prompt'][:300]}\nYOUR RESPONSE (excerpt): {f['response'][:400]}"
        for f in failures
    )
    system = DIAG_SYSTEM_PROMPT.format(
        max_rounds=max_rounds,
        tool_manual=TOOL_MANUAL,
        evidence=evidence,
    )

    messages: list[dict] = [{"role": "user", "content": system}]
    transcript: list[dict] = []

    def build_prompt(msgs: list[dict]) -> str:
        """把消息历史渲染成单一 prompt（chat template 一次成型）。"""
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    for round_i in range(max_rounds):
        gen = generate(model, tokenizer, build_prompt(messages[-12:]), max_new_tokens=512, enable_thinking=False)
        call = extract_tool_call(gen["response"])
        transcript.append({"round": round_i, "model_output": gen["response"], "parsed_call": call})
        print(f"[diag round {round_i}] -> {json.dumps(call, ensure_ascii=False)[:200] if call else 'NO CALL'}", flush=True)
        messages.append({"role": "assistant", "content": gen["response"]})

        if call is None:
            messages.append({"role": "user", "content": "Your last reply was not a valid JSON tool call. Output ONLY one JSON object with one tool invocation."})
            continue

        tool = call.get("tool")
        try:
            if tool == "list_layers":
                result = tool_list_layers(args_dir_for_config)
            elif tool == "tensor_stats":
                result = tool_tensor_stats_from_model(model, int(call["layer"]), call["proj"])
            elif tool == "compare_tensor_stats":
                result = tool_compare_from_model(model, call["proj"], call.get("layers"))
            elif tool == "activation_probe":
                result = tool_activation_probe(
                    None, int(call["layer"]), call["proj"],
                    call.get("prompts", ["Hello world"])[:4], tokenizer=tokenizer, model=model,
                )
            elif tool == "propose_edit":
                return call, transcript
            else:
                result = f"unknown tool: {tool}"
        except Exception as e:  # noqa: BLE001
            result = f"tool error: {type(e).__name__}: {e}"

        messages.append({"role": "user", "content": f"TOOL RESULT:\n{result}\n\nContinue investigating. Round {round_i + 2}/{max_rounds}. Remember: output ONE JSON tool call."})

    # 预算耗尽，强制要求提案
    messages.append({"role": "user", "content": "BUDGET EXHAUSTED. You must now output a propose_edit JSON (single object)."})
    gen = generate(model, tokenizer, build_prompt(messages[-12:]), max_new_tokens=512, enable_thinking=False)
    call = extract_tool_call(gen["response"])
    transcript.append({"round": max_rounds, "model_output": gen["response"], "parsed_call": call})
    return (call or {"tool": "propose_edit", "error": "failed_to_parse"}, transcript)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--defect", default="checkpoints/V0-defect")
    ap.add_argument("--workdir", default="experiments/iteration_1")
    ap.add_argument("--max-rounds", type=int, default=10)
    ap.add_argument("--n-evidence", type=int, default=12)
    ap.add_argument("--skip-diagnosis", action="store_true", help="跳过诊断，直接复评（调试用）")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    print(f"[phase1] loading defect model {args.defect}", flush=True)
    model, tokenizer = load_model(args.defect)

    if not args.skip_diagnosis:
        print("[phase1] stage 1: collecting behavioral evidence", flush=True)
        failures = collect_failures(model, tokenizer, n=args.n_evidence)
        (workdir / "evidence.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        n_fail = sum(1 for f in failures if f["score"] < 0.5)
        print(f"[phase1] evidence: {n_fail}/{len(failures)} scored below 0.5", flush=True)

        print("[phase1] stage 2: self-diagnosis loop", flush=True)
        proposal, transcript = run_diagnosis_loop(
            model, tokenizer, failures, workdir, max_rounds=args.max_rounds,
            config_dir=args.defect,  # defect 目录含 config.json
        )
        (workdir / "diagnosis_transcript.json").write_text(
            json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (workdir / "proposal.json").write_text(
            json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[phase1] proposal: {json.dumps(proposal, ensure_ascii=False)[:400]}", flush=True)

    print("[phase1] done. Next: validate proposal via edit_protocol, apply, re-evaluate.", flush=True)


if __name__ == "__main__":
    main()
