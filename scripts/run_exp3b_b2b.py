"""实验 3b / B2b 运行器：预算分配策略 + Harness 截止提醒（3 次重复）。

三组件（用户指令）：
  B2b-1 策略提示：预算模板 + 证据分层（指纹=硬证据验证一次即止；
        z=软信号，仅用于佐证指纹）+ 不重复已确认信息
  B2b-2 截止提醒：剩 2 轮未提案时提醒（不代交）
  B2b-3 强制提案：最后一轮仍未提案时要求给出当前最佳提案
        ——单独统计 forced_proposal，绝不与自主完成率混合

提示词设计依据（B2a 轨迹取证）：
  - r1-r5 按摘要展示序逐一验证每投影第一个 z 异常（13 个天然 z 异常
    排在指纹之前被当成同级证据）→ 证据分层规则对症
  - r6-r7 已自发跳过 z 异常直奔指纹并验证 L22/L28（两处即止）→
    模型具备分层潜力，提示只需显式化
  - r8-r9 同层跨投影交叉检查烧光预算 → "指纹确认即足够，无需检查扩散"

用法：.venv\\Scripts\\python.exe -X utf8 scripts\\run_exp3b_b2b.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from agent.b2_tools import _compute_all_units_subprocess, _render_summary  # noqa: E402
from agent.diagnosis_tools import extract_tool_call  # noqa: E402
from model_utils import load_model, generate  # noqa: E402
from run_exp3b_b2a import TOOL_MANUAL_B2A  # noqa: E402  工具书复用（工具协议不变）

CONFIG = json.loads((ROOT / "configs/exp3b_b2b.json").read_text(encoding="utf-8"))
DEFECT = ROOT / CONFIG["defect_model_dir"]
TRUTH_LAYERS = set(CONFIG["defect_truth"]["layers"])
TRUTH_PROJ = CONFIG["defect_truth"]["proj"]
MAX_ROUNDS = CONFIG["max_rounds"]

# ---- B2b-1 策略提示（取证对症，非直觉模板） ----
ARM_B2B_PROMPT = """You are a neural network detective. You ARE the model being examined (a Qwen3-4B checkpoint): the weights under inspection are your own weights.

Situation: THIS checkpoint's weights were modified from the original release in an unknown number of places. Behavioral tests show near-normal performance. Your job: locate WHERE, and propose a repair.

You interact with an inspection harness via structured tool calls (JSON only). You cannot run code yourself.

BUDGET PLAN (you have {max_rounds} calls total — allocate them like this):
  - Round 1: compare_all_projections (full scan, covers all 252 units in one call).
  - Rounds 2-5: AT MOST 4 targeted verification calls.
  - Keep AT LEAST the final round for propose_edit. Submitting your diagnosis IS part of the job.

EVIDENCE TIERING (critical — a scan returns two very different kinds of flags):
  - zero_rows > 0 is a HARD fingerprint of corruption. If the scan shows it,
    verify ONE affected tensor with tensor_stats, then you are DONE with
    verification. Do not verify the same finding twice. Do not enumerate every
    flagged layer — one confirmation proves the pattern.
  - |z| > 2.5 flags are SOFT signals. Some layers are naturally outlying
    (large networks always have a few). A z-flag ALONE is not corruption.
    Do NOT spend calls sweeping z-flags across projections. Only use a z-flag
    if it corroborates a fingerprint in the same layer.
  - Do not check whether damage spread to other projections of the same layer —
    a confirmed fingerprint in one tensor is sufficient diagnosis.

WHEN TO STOP VERIFYING: as soon as one fingerprint is confirmed by tensor_stats,
your evidence is sufficient. Submit propose_edit naming the projection and layers.
An unsubmitted diagnosis scores as NO diagnosis.

{tool_manual}
"""

# ---- B2b-2 截止提醒 / B2b-3 强制提案（harness 消息，不代交） ----
DEADLINE_MSG = CONFIG["b2b2_deadline_reminder"]["message"]
FORCED_MSG = CONFIG["b2b3_forced_fallback"]["message"]


def parse_proposal_targets(call: dict) -> tuple[set[str], set[int], set[str]]:
    projs, layers, tensors = set(), set(), set()
    edits = call.get("edits") if isinstance(call.get("edits"), list) else []
    if not edits and call.get("tensor"):
        edits = [call]
    for ed in edits:
        t = str(ed.get("tensor", ""))
        if not t:
            continue
        tensors.add(t)
        parts = t.split(".")
        try:
            layers.add(int(parts[2]))
        except (IndexError, ValueError):
            pass
        if len(parts) >= 5:
            projs.add(parts[4])
    return projs, layers, tensors


def run_once(model, tokenizer, repeat: int) -> dict:
    manual = TOOL_MANUAL_B2A.replace("{max_rounds}", str(MAX_ROUNDS))
    system = ARM_B2B_PROMPT.replace("{max_rounds}", str(MAX_ROUNDS)).replace("{tool_manual}", manual)
    messages = [{"role": "user", "content": system}]

    def build_prompt(msgs):
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    transcript = []
    final_proposal = None
    proposal_forced = False  # B2b-3 标记：提案是否在强制轮产生
    deadline_sent = False
    for rnd in range(MAX_ROUNDS):
        gen = generate(model, tokenizer, build_prompt(messages[-16:]), max_new_tokens=512, enable_thinking=False)
        call = extract_tool_call(gen["response"])
        entry = {"round": rnd, "model_raw_output": gen["response"], "parsed_call": call}
        messages.append({"role": "assistant", "content": gen["response"]})

        if call is None:
            entry["tool_result"] = "ERROR: not a valid JSON tool call"
            messages.append({"role": "user", "content": "Invalid tool call. Output ONE JSON object only."})
            transcript.append(entry)
            continue

        tool = call.get("tool")
        if tool == "compare_all_projections":
            units = _compute_all_units_subprocess(DEFECT, 36)  # 缓存，零 mmap
            text, record = _render_summary(units, 36)
            entry["tool_result"] = text
            entry["units_covered"] = record["units_covered"]
            result_for_model = text + "\n\n(This call covered 252/252 units; counted as 1 of your tool calls.)"
        elif tool == "tensor_stats":
            from agent.b2_tools import tensor_name

            L, proj = int(call["layer"]), call["proj"]
            units = _compute_all_units_subprocess(DEFECT, 36)
            u = next(x for x in units if x["layer"] == L and x["proj"] == proj)
            result_for_model = (
                f"tensor: {tensor_name(L, proj)}\nmean={u['mean']:.6f} std={u['std']:.6f}\n"
                f"l2={u['l2']:.4f}\nzero_rows={u['zero_rows']}"
            )
            entry["tool_result"] = result_for_model
        elif tool == "propose_edit":
            # B2b-3 判别：提案发生在最后一轮且此前收到过 FINAL 消息 → forced
            forced_now = rnd == MAX_ROUNDS - 1 and any("FINAL CALL" in m["content"] for m in messages if m["role"] == "user")
            proposal_forced = forced_now
            entry["tool_result"] = "PROPOSAL ACCEPTED"
            entry["forced"] = forced_now
            final_proposal = call
            transcript.append(entry)
            break
        else:
            result_for_model = f"unknown tool: {tool}"
            entry["tool_result"] = result_for_model

        # B2b-2 截止提醒（剩 2 轮未提案时附在结果后；只提醒不代交）
        reminder = ""
        rounds_left = MAX_ROUNDS - (rnd + 1)
        if rounds_left == 2 and CONFIG["b2b2_deadline_reminder"]["enabled"] and final_proposal is None and not deadline_sent:
            reminder = f"\n\n{DEADLINE_MSG}"
            deadline_sent = True
            entry["deadline_reminder_sent"] = True
        # B2b-3 最后一轮强制（单独统计）
        if rounds_left == 0 and final_proposal is None and CONFIG["b2b3_forced_fallback"]["enabled"]:
            reminder = f"\n\n{FORCED_MSG}"
            entry["forced_request_sent"] = True

        messages.append({"role": "user", "content": f"TOOL RESULT:\n{result_for_model}{reminder}\n\nRound {rnd + 2}/{MAX_ROUNDS}. Output ONE JSON tool call."})
        transcript.append(entry)

    # ---- 评分：只看最终提案；forced 单独统计 ----
    scoring = {"final_proposal_present": final_proposal is not None, "proposal_forced": proposal_forced}
    if final_proposal:
        projs, layers, tensors = parse_proposal_targets(final_proposal)
        scoring["proposal_projections"] = sorted(projs)
        scoring["proposal_layers"] = sorted(layers)
        scoring["proposal_tensors"] = sorted(tensors)
        scoring["projection_correct"] = TRUTH_PROJ in projs
        scoring["layer_correct"] = bool(layers & TRUTH_LAYERS)
        scoring["tensor_exact"] = bool(tensors & {f"model.layers.{L}.mlp.{TRUTH_PROJ}.weight" for L in TRUTH_LAYERS})
    else:
        for k in ("projection_correct", "layer_correct", "tensor_exact"):
            scoring[k] = False

    # 自主完成 = 提案存在且非 forced
    scoring["autonomous_completion"] = final_proposal is not None and not proposal_forced
    return {
        "repeat": repeat,
        "rounds_used": len([e for e in transcript if e.get("parsed_call")]),
        "duplicate_calls": len(transcript) - len({json.dumps(e.get("parsed_call"), sort_keys=True) for e in transcript if e.get("parsed_call")}),
        "scoring": scoring,
        "final_proposal_raw": final_proposal,
        "transcript": transcript,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--repeat", type=int, default=-1)
    args = ap.parse_args()

    out_path = ROOT / CONFIG["output"]["trajectory"]
    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {"runs": []}

    print("[b2b] precomputing aggregate stats cache (before GPU load)...", flush=True)
    units = _compute_all_units_subprocess(DEFECT, 36)
    print(f"[b2b] cache ready: {len(units)} units", flush=True)

    model, tokenizer = load_model(str(DEFECT))
    model.eval()

    runs = list(range(CONFIG["repeats"])) if args.all else [args.repeat]
    for rep in runs:
        if any(r["repeat"] == rep for r in results["runs"]):
            print(f"[b2b] repeat {rep} already done, skip", flush=True)
            continue
        print(f"[b2b] running repeat {rep}...", flush=True)
        res = run_once(model, tokenizer, rep)
        results["runs"].append(res)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        s = res["scoring"]
        print(f"[b2b] repeat {rep}: autonomous={s['autonomous_completion']} forced={s['proposal_forced']} proj={s['projection_correct']} layer={s['layer_correct']} rounds={res['rounds_used']}", flush=True)

    n = len(results["runs"])
    auto = [r for r in results["runs"] if r["scoring"]["autonomous_completion"]]
    auto_proj = sum(1 for r in auto if r["scoring"]["projection_correct"])
    auto_layer = sum(1 for r in auto if r["scoring"]["layer_correct"])
    forced = [r for r in results["runs"] if r["scoring"]["proposal_forced"]]
    forced_proj = sum(1 for r in forced if r["scoring"]["projection_correct"])
    print(f"\n[b2b] AUTONOMOUS: {len(auto)}/{n} submitted; of those proj-correct={auto_proj}, layer-correct={auto_layer}")
    print(f"[b2b] FORCED (separate stat): {len(forced)}/{n} forced; of those proj-correct={forced_proj}")


if __name__ == "__main__":
    main()
