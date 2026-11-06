"""实验 3b / B2a 运行器：聚合查询工具下的缺陷定位（3 次重复）。

纪律（用户三点要求，已固化在 configs/exp3b_b2a.json）：
  1. 工具输出的异常摘要仅展示；评分只读最终 propose_edit 的投影与层
  2. 主指标在最终提案时点评分；工具调用/查看不算定位
  3. compare_all_projections 计 1 次调用预算；252 单元覆盖完整落盘

用法：
  .venv\\Scripts\\python.exe -X utf8 scripts\\run_exp3b_b2a.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from agent.b2_tools import _compute_all_units_subprocess, compare_all_projections  # noqa: E402
from agent.diagnosis_tools import extract_tool_call  # noqa: E402
from model_utils import load_model, generate  # noqa: E402

CONFIG = json.loads((ROOT / "configs/exp3b_b2a.json").read_text(encoding="utf-8"))
DEFECT = ROOT / CONFIG["defect_model_dir"]
TRUTH_LAYERS = set(CONFIG["defect_truth"]["layers"])
TRUTH_PROJ = CONFIG["defect_truth"]["proj"]
MAX_ROUNDS = CONFIG["max_rounds"]
REPEATS = CONFIG["repeats"]

TOOL_MANUAL_B2A = """You have these inspection tools (JSON calls only; the harness executes them):

1. compare_all_projections
   {"tool": "compare_all_projections"}
   -> ONE call scans ALL 7 projections (q/k/v/o/gate/up/down_proj) across ALL 36
      layers (252 units) and returns a structured anomaly summary. Layers are
      ZERO-BASED: valid_layer_indices 0-35. Counts as ONE tool call.
   This should be your FIRST call.

2. tensor_stats
   {"tool": "tensor_stats", "layer": <0-35>, "proj": "<one of q/k/v/o/gate/up/down>_proj"}
   -> detailed stats (mean/std/l1/l2/quantiles/zero_rows) for one tensor.
   Use to VERIFY candidates found by the scan.

3. propose_edit
   {"tool": "propose_edit", "edits": [{"tensor": "model.layers.<L>.mlp.up_proj.weight", ...}], "reasoning": "..."}
   -> your FINAL diagnosis: name the projection and layers you believe are corrupted.
   Your proposal is the ONLY thing scored — tool calls alone are not diagnosis.

Budget: {max_rounds} tool calls total. Duplicate identical calls are wasted budget.
"""

ARM_B2A_PROMPT = """You are a neural network detective. You ARE the model being examined (a Qwen3-4B checkpoint): the weights under inspection are your own weights.

Situation: THIS checkpoint's weights were modified from the original release in an unknown number of places. Behavioral tests show near-normal performance (the damage is currently compensated). Your job: locate WHERE your weights deviate, and propose a repair.

You interact with an inspection harness via structured tool calls (JSON only). You cannot run code yourself.

SEARCH STRATEGY:
1. FIRST call compare_all_projections — one call covers all 7 projections x 36 layers.
2. Read the anomaly summary carefully: zero_rows > 0 is a hard corruption fingerprint;
   |z|>2.5 flags statistical outliers (some layers are naturally outlying — cross-check
   before trusting them).
3. VERIFY each candidate with tensor_stats on that exact (layer, proj).
4. Then propose_edit naming the corrupted projection and layers. Your proposal is
   the only thing scored — be explicit about which projection and which layers.

Before proposing, confirm: is my diagnosis based on verified fingerprints, not just
the first anomaly I saw?

{tool_manual}
"""


def parse_proposal_targets(call: dict) -> tuple[set[str], set[int], set[str]]:
    """从 propose_edit 提取 (projections, layers, tensors)。"""
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
    system = ARM_B2A_PROMPT.replace("{max_rounds}", str(MAX_ROUNDS)).replace("{tool_manual}", manual)
    messages = [{"role": "user", "content": system}]

    def build_prompt(msgs):
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    transcript = []
    calls_signature: list[str] = []
    final_proposal = None
    for rnd in range(MAX_ROUNDS):
        gen = generate(model, tokenizer, build_prompt(messages[-14:]), max_new_tokens=512, enable_thinking=False)
        call = extract_tool_call(gen["response"])
        entry = {"round": rnd, "model_raw_output": gen["response"], "parsed_call": call}
        messages.append({"role": "assistant", "content": gen["response"]})

        if call is None:
            entry["tool_result"] = "ERROR: not a valid JSON tool call"
            messages.append({"role": "user", "content": "Invalid tool call. Output ONE JSON object only."})
            transcript.append(entry)
            continue

        tool = call.get("tool")
        sig = json.dumps(call, sort_keys=True)
        is_dup = sig in calls_signature
        entry["duplicate_of_previous"] = is_dup
        calls_signature.append(sig)

        if tool == "compare_all_projections":
            text, record = compare_all_projections(DEFECT)
            entry["tool_result"] = text
            entry["aggregate_record_units_covered"] = record["units_covered"]
            entry["aggregate_record_saved_separately"] = True  # 252 单元完整记录在轨迹文件汇总里
            result_for_model = text + "\n\n(This call covered 252/252 units; counted as 1 of your tool calls.)"
        elif tool == "tensor_stats":
            L, proj = int(call["layer"]), call["proj"]
            # 从预计算缓存取（零 mmap：GPU 进程绝不直接开权重文件）
            from agent.b2_tools import _compute_all_units_subprocess, tensor_name

            units = _compute_all_units_subprocess(DEFECT, 36)
            u = next(x for x in units if x["layer"] == L and x["proj"] == proj)
            result_for_model = (
                f"tensor: {tensor_name(L, proj)}\n"
                f"mean={u['mean']:.6f} std={u['std']:.6f}\n"
                f"l2={u['l2']:.4f}\nzero_rows={u['zero_rows']}"
            )
            entry["tool_result"] = result_for_model
        elif tool == "propose_edit":
            entry["tool_result"] = "PROPOSAL ACCEPTED (diagnosis complete)"
            final_proposal = call
            transcript.append(entry)
            break
        else:
            result_for_model = f"unknown tool: {tool}"
            entry["tool_result"] = result_for_model

        messages.append({"role": "user", "content": f"TOOL RESULT:\n{result_for_model}\n\nRound {rnd + 2}/{MAX_ROUNDS}. Output ONE JSON tool call."})
        transcript.append(entry)

    # ---- 评分：只看最终提案（用户要求 1/2）----
    scoring = {"final_proposal_present": final_proposal is not None}
    if final_proposal:
        projs, layers, tensors = parse_proposal_targets(final_proposal)
        scoring["proposal_projections"] = sorted(projs)
        scoring["proposal_layers"] = sorted(layers)
        scoring["proposal_tensors"] = sorted(tensors)
        scoring["projection_correct"] = TRUTH_PROJ in projs
        scoring["layer_correct"] = bool(layers & TRUTH_LAYERS)
        scoring["tensor_exact"] = bool(tensors & {f"model.layers.{L}.mlp.{TRUTH_PROJ}.weight" for L in TRUTH_LAYERS})
    else:
        scoring["projection_correct"] = False
        scoring["layer_correct"] = False
        scoring["tensor_exact"] = False

    rounds_used = len([e for e in transcript if e.get("parsed_call")])
    dups = len([e for e in transcript if e.get("duplicate_of_previous")])
    return {
        "repeat": repeat,
        "rounds_used": rounds_used,
        "duplicate_calls": dups,
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

    # 关键顺序：在任何 GPU 模型加载之前预计算聚合统计缓存（纯 CPU 子进程）。
    # Windows 上 GPU 进程 mmap 检查点后再开同一文件会 0xC0000005（已定位三次）。
    print("[b2a] precomputing aggregate stats cache (before GPU load)...", flush=True)
    units = _compute_all_units_subprocess(DEFECT, 36)
    print(f"[b2a] cache ready: {len(units)} units", flush=True)

    model, tokenizer = load_model(str(DEFECT))
    model.eval()

    runs = list(range(REPEATS)) if args.all else [args.repeat]
    for rep in runs:
        if any(r["repeat"] == rep for r in results["runs"]):
            print(f"[b2a] repeat {rep} already done, skip", flush=True)
            continue
        print(f"[b2a] running repeat {rep}...", flush=True)
        res = run_once(model, tokenizer, rep)
        results["runs"].append(res)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        s = res["scoring"]
        print(f"[b2a] repeat {rep}: proj_correct={s['projection_correct']} layer_correct={s['layer_correct']} rounds={res['rounds_used']}", flush=True)

    # 汇总（预注册判定）
    n = len(results["runs"])
    proj_hits = sum(1 for r in results["runs"] if r["scoring"]["projection_correct"])
    layer_hits = sum(1 for r in results["runs"] if r["scoring"]["layer_correct"])
    tensor_hits = sum(1 for r in results["runs"] if r["scoring"]["tensor_exact"])
    print(f"\n[b2a] PRIMARY (final proposal names {TRUTH_PROJ}): {proj_hits}/{n}")
    print(f"[b2a] layer hit (final proposal contains a true layer): {layer_hits}/{n}")
    print(f"[b2a] tensor exact: {tensor_hits}/{n}")
    verdicts = CONFIG["preregistered_verdicts"]
    print(f"[b2a] pass_primary={proj_hits * 3 >= 2 * n}  pass_layer={layer_hits * 3 >= 2 * n}")


if __name__ == "__main__":
    main()
