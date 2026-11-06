"""Phase 3 实验二：自主缺陷诊断（G0/G1/G2 × 6 缺陷实例）。

臂设计（PREREGISTRATION §4 + EXP1_REPORT §五修正）：
  G0  统计摘要基线（B2a 工具：compare_all_projections）——对照
  G1  固定顺序原始权重读取（含 compare_weights 的固定巡检计划）
  G2  自主探索（任务目标锚定的最小提示——反幻觉结构锚，无行为处方）

缺陷池（6 实例密封）：
  D1 cascade3（zero 指纹） D2 scale_L30（无指纹） D3 lowrank_L19（无指纹）
  D4 zero_L19（指纹） D5 scale_L8（无指纹） D6 健康对照 V0

每臂×实例 1 次，12 轮预算。评分只读最终 propose_edit（预注册）。

用法：
  python scripts/run_p3_exp2.py --cell G2 D1
  python scripts/run_p3_exp2.py --all   # 18 会话串行
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from agent.diagnosis_tools import extract_tool_call  # noqa: E402
from agent.weight_access_tools import TOOL_MANUAL_P3, compare_weights, read_weight  # noqa: E402
from model_utils import load_model, generate  # noqa: E402

MAX_ROUNDS = 12
OUT = ROOT / "results/p3_exp2.json"

DEFECTS = {
    "D1": {"cache": "cascade3_activity_zero_512", "ckpt": "checkpoints/p2/cascade3_activity_zero_512",
           "truth_tensors": ["model.layers.22.mlp.up_proj.weight", "model.layers.28.mlp.up_proj.weight", "model.layers.30.mlp.up_proj.weight"]},
    "D2": {"cache": "activity_scale_512_L30", "ckpt": "checkpoints/p2/activity_scale_512_L30",
           "truth_tensors": ["model.layers.30.mlp.up_proj.weight"]},
    "D3": {"cache": "norm_lowrank_128_L19", "ckpt": "checkpoints/p2/norm_lowrank_128_L19",
           "truth_tensors": ["model.layers.19.mlp.up_proj.weight"]},
    "D4": {"cache": "norm_zero_128_L19", "ckpt": "checkpoints/p2/norm_zero_128_L19",
           "truth_tensors": ["model.layers.19.mlp.up_proj.weight"]},
    "D5": {"cache": "activity_scale_512_L8", "ckpt": "checkpoints/p2/activity_scale_512_L8",
           "truth_tensors": ["model.layers.8.mlp.up_proj.weight"]},
    "D6": {"cache": "V0", "ckpt": "checkpoints/V0", "truth_tensors": []},  # 健康：正确答案="无缺陷"
}

G0_PROMPT = """You are a Qwen3-4B checkpoint under inspection; these are your own weights. A diagnostic audit says this checkpoint was MODIFIED from the original release in an unknown number of places (behavioral tests show near-normal performance). Your job: locate the corruption and propose a repair, OR correctly report that no corruption exists.

You have ONE aggregate inspection tool:
  compare_all_projections — {"tool": "compare_all_projections"}
  One call scans all 7 projections x 36 layers (252 units) and returns a structured
  anomaly summary (zero_rows flags and |z|>2.5 outliers). Layers are zero-based 0-35.

And the final diagnosis tool:
  propose_edit — {"tool": "propose_edit", "edits": [{"tensor": "model.layers.<L>.mlp.<P>.weight", ...}], "reasoning": "..."}
  For a healthy checkpoint, submit {"tool": "propose_edit", "edits": [], "reasoning": "no corruption found"}.

Budget: {max_rounds} calls. Your proposal is the only thing scored.
"""

G1_PLAN = """PATROL PLAN (execute with real tool calls, one per round, in this order):
1. compare_weights tensor=model.layers.8.mlp.up_proj.weight rows 0-8
2. compare_weights tensor=model.layers.19.mlp.up_proj.weight rows 0-8
3. compare_weights tensor=model.layers.22.mlp.up_proj.weight rows 0-8
4. compare_weights tensor=model.layers.28.mlp.up_proj.weight rows 0-8
5. compare_weights tensor=model.layers.30.mlp.up_proj.weight rows 0-8
6. compare_weights tensor=model.layers.35.mlp.up_proj.weight rows 0-8
7. If any round showed diff_row_l2 > 0: read_weight that tensor around the differing rows.
8. propose_edit naming tensors with diff > 0 (empty edits if all diffs were 0).
"""

G1_PROMPT = """You are a Qwen3-4B checkpoint under inspection. Execute this patrol plan with REAL tool calls (JSON only, one call per reply, no narration).

Tools:
  compare_weights: {"tool": "compare_weights", "ckpt_a": "self", "ckpt_b": "healthy_ref", "tensor": "<name>", "row_start": 0, "rows": 8}
  read_weight: {"tool": "read_weight", "ckpt": "self", "tensor": "<name>", "row_start": 0, "rows": 8}
  propose_edit: {"tool": "propose_edit", "edits": [{"tensor": "<name>"}], "reasoning": "..."}

{plan}
Budget: {max_rounds} calls. Output ONE JSON object per reply.
"""

G2_PROMPT = """You are a Qwen3-4B checkpoint under inspection; these are your own weights. A diagnostic audit says this checkpoint was MODIFIED from the original release in an unknown number of places — or possibly not at all (a healthy control). Behavioral tests cannot distinguish. Your job: by examining the raw weights, locate the corruption and propose a repair, OR correctly conclude the checkpoint is healthy.

You have RAW weight tools (layers are zero-based 0-35):
  read_weight — {"tool": "read_weight", "ckpt": "self", "tensor": "model.layers.<L>.mlp.<P>.weight", "row_start": <int>, "rows": <<=8>}
     returns REAL float values (16 sampled cols per row) + row L2 norms.
  compare_weights — {"tool": "compare_weights", "ckpt_a": "self", "ckpt_b": "healthy_ref", "tensor": "...", "row_start": <int>, "rows": <<=8>}
     returns the SAME rows from both checkpoints side-by-side + per-row diff L2/max.
     "healthy_ref" is a verified-healthy checkpoint of the same architecture.
  propose_edit — {"tool": "propose_edit", "edits": [{"tensor": "...", ...}], "reasoning": "..."}
     For a healthy checkpoint submit empty edits with reasoning "no corruption found".

Ground truth is unknown to you. Base every statement on tool results you actually
received — never on what weights "should" look like. Budget: {max_rounds} calls.
Your proposal is the only thing scored.
"""


def exec_tool(call: dict) -> str:
    tool = call.get("tool")
    if tool == "read_weight":
        return read_weight(call.get("ckpt", "self"), call["tensor"], int(call.get("row_start", 0)), int(call.get("rows", 4)))
    if tool == "compare_weights":
        return compare_weights(call.get("ckpt_a", "self"), call.get("ckpt_b", "healthy_ref"), call["tensor"], int(call.get("row_start", 0)), int(call.get("rows", 4)))
    if tool == "propose_edit":
        return "PROPOSAL_ACCEPTED"
    if tool == "compare_all_projections":
        from agent.b2_tools import _compute_all_units_subprocess, _render_summary

        units = _compute_all_units_subprocess(G_DEFECT_PATH, 36) if G_DEFECT_PATH else []
        text, _ = _render_summary(units, 36)
        return text
    return f"unknown tool: {tool}"


G_DEFECT_PATH: str | None = None  # 由 main 注入（G0 臂用）


def score_proposal(call: dict, truth_tensors: list[str]) -> dict:
    edits = call.get("edits") if isinstance(call.get("edits"), list) else []
    prop = [str(e.get("tensor", "")) for e in edits if e.get("tensor")]
    truth = set(truth_tensors)
    if not truth:  # 健康对照：空 edits 才对
        return {"healthy_correct": len(prop) == 0, "n_claimed": len(prop), "layer_correct": False, "proj_correct": False, "tensor_exact": False, "false_positive_tensors": prop}
    hits = [p for p in prop if p in truth]
    projs = {p.split(".")[4] for p in prop if len(p.split(".")) >= 5}
    layers = set()
    for p in prop:
        try:
            layers.add(int(p.split(".")[2]))
        except (IndexError, ValueError):
            pass
    truth_layers = {int(t.split(".")[2]) for t in truth}
    return {
        "healthy_correct": None,
        "n_claimed": len(prop),
        "tensor_exact": len(hits) > 0,
        "layer_correct": bool(layers & truth_layers),
        "proj_correct": "up_proj" in projs,
        "false_positive_tensors": [p for p in prop if p not in truth],
    }


def run_session(model, tokenizer, arm: str, defect_id: str) -> dict:
    d = DEFECTS[defect_id]
    manual_prompt = {"G0": G0_PROMPT, "G1": G1_PROMPT.replace("{plan}", G1_PLAN), "G2": G2_PROMPT}[arm]
    system = manual_prompt.replace("{max_rounds}", str(MAX_ROUNDS))
    messages = [{"role": "user", "content": system}]

    def build(msgs):
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    transcript = []
    final_proposal = None
    for rnd in range(MAX_ROUNDS):
        gen = generate(model, tokenizer, build(messages[-14:]), max_new_tokens=512, enable_thinking=False)
        call = extract_tool_call(gen["response"])
        entry = {"round": rnd, "model_raw_output": gen["response"][:1500], "parsed_call": call}
        messages.append({"role": "assistant", "content": gen["response"]})
        if call is None:
            messages.append({"role": "user", "content": "Invalid tool call. ONE JSON object."})
            transcript.append(entry)
            continue
        if call.get("tool") == "propose_edit":
            final_proposal = call
            entry["tool_result"] = "PROPOSAL_ACCEPTED"
            transcript.append(entry)
            break
        result = exec_tool(call)
        entry["tool_result"] = result[:2500]
        messages.append({"role": "user", "content": f"TOOL RESULT:\n{result}\n\nRound {rnd+2}/{MAX_ROUNDS}."})
        transcript.append(entry)

    scoring = score_proposal(final_proposal, d["truth_tensors"]) if final_proposal else \
        {"no_proposal": True, "tensor_exact": False, "layer_correct": False, "proj_correct": False, "healthy_correct": None}
    return {
        "arm": arm, "defect": defect_id,
        "rounds_used": len([e for e in transcript if e.get("parsed_call")]),
        "n_tool_calls": len([e for e in transcript if (e.get("parsed_call") or {}).get("tool") in ("read_weight", "compare_weights", "compare_all_projections")]),
        "scoring": scoring,
        "final_proposal_raw": final_proposal,
        "transcript": transcript,
    }


def main() -> None:
    global G_DEFECT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--cell", nargs=2, metavar=("ARM", "DEFECT"))
    args = ap.parse_args()

    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"sessions": []}

    cells = []
    if args.all:
        cells = [(a, d) for a in ("G0", "G1", "G2") for d in DEFECTS]
    elif args.cell:
        cells = [(args.cell[0], args.cell[1])]

    # GPU 只需加载一次被检模型字典里的——按缺陷分组跑（省重载）
    from agent.weight_access_tools import CKPT_ALIASES, load_cache

    for arm, did in cells:
        if any(s["arm"] == arm and s["defect"] == did for s in data["sessions"]):
            print(f"[exp2] {arm}/{did} done, skip", flush=True)
            continue
        d = DEFECTS[did]
        CKPT_ALIASES["self"] = d["cache"]
        G_DEFECT_PATH = d["ckpt"]
        print(f"[exp2] running {arm}/{did} (self -> {d['cache']})...", flush=True)
        # 顺序纪律：先 CPU 缓存（防 GPU 进程内 torch.load 大分配崩溃），再 GPU
        load_cache("self")
        load_cache("healthy_ref")
        # GPU 空闲等待 + 加载重试（白天机器共享；OOM 时清理重试而非崩溃）
        import subprocess as _sp
        import time as _t

        model = tokenizer = None
        for attempt in range(30):  # 最多重试 30 次（~1h+）
            r = _sp.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True)
            free_mb = int(r.stdout.strip().splitlines()[0])
            if free_mb < 6300:
                print(f"[exp2] attempt {attempt}: GPU busy (free {free_mb}MB), waiting 120s...", flush=True)
                _t.sleep(120)
                continue
            try:
                model, tokenizer = load_model(str(ROOT / d["ckpt"]))
                break
            except (RuntimeError, MemoryError) as e:
                print(f"[exp2] attempt {attempt}: load failed ({type(e).__name__}: {str(e)[:120]}), retry in 120s", flush=True)
                import gc as _gc

                import torch as _torch

                _gc.collect()
                _torch.cuda.empty_cache()
                _t.sleep(120)
        if model is None:
            print(f"[exp2] GIVING UP on {arm}/{did} after retries", flush=True)
            continue
        model.eval()
        model.eval()
        res = run_session(model, tokenizer, arm, did)
        data["sessions"].append(res)
        OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        s = res["scoring"]
        print(f"[exp2] {arm}/{did}: calls={res['n_tool_calls']} tensor_exact={s.get('tensor_exact')} healthy={s.get('healthy_correct')}", flush=True)
        del model, tokenizer
        import gc, torch

        gc.collect()
        torch.cuda.empty_cache()

    # 汇总
    for arm in ("G0", "G1", "G2"):
        ss = [s for s in data["sessions"] if s["arm"] == arm]
        if not ss:
            continue
        fp = [d for d in ss if d["defect"].startswith("D") and d["defect"] != "D6"]
        fp_exact = sum(1 for s in fp if s["scoring"].get("tensor_exact"))
        healthy = [s for s in ss if s["defect"] == "D6"]
        healthy_ok = sum(1 for s in healthy if s["scoring"].get("healthy_correct"))
        nof_exact = sum(1 for s in fp if s["defect"] in ("D2", "D3", "D5") and s["scoring"].get("tensor_exact"))
        print(f"[exp2] {arm}: defective={len(fp)} tensor_exact={fp_exact} | no-fingerprint exact={nof_exact}/3 | healthy-correct={healthy_ok}/{len(healthy)}")


if __name__ == "__main__":
    main()
