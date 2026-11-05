"""实验四：修复闭环四臂（探索性 - AMENDMENT1 D4）。

缺陷：cascade3_activity_zero_512（L22/28/30 up_proj top-512-activity 行置零）
四臂：
  A1 agent-full        Agent 自诊+定位+提案修复值
  A2 oracle-loc+agent  实验方注入病灶位置（层+行），Agent 只生成修复值
  A3 oracle-full       V0 原值恢复（上界）
  A4 random-edit       随机层+随机行同预算（下界）

A2 是本实验核心（用户要求 #5）：拆解定位误差后修复值生成能力的净测量。
A1 用 B 臂提示（策略引导——A1 的意义在"全链路"而非重复实验三的 A 臂失败）。
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
from agent.diagnosis_tools import TOOL_MANUAL, extract_tool_call  # noqa: E402

DEFECT = "checkpoints/p2/cascade3_activity_zero_512"
SEALED = "experiments/iteration_2/_sealed/cascade3_activity_zero_512.json"

A2_PROMPT = """You are a neural network repair engineer. You ARE the model being repaired (a Qwen3-4B checkpoint): these are your own weights.

Situation: A diagnostic audit has ALREADY located the corruption in your weights. The audit found:
- Tensor: model.layers.{L}.mlp.up_proj.weight
- Rows: {k} rows (indices in the tool result below) were ZEROED (set to exactly 0)

Your job now is ONLY the repair: decide what VALUES to write back into those zeroed rows.
You cannot read the original checkpoint (it is unavailable). You must infer healthy values
from what you know about how trained transformer weights are distributed.

You have ONE tool call: propose_edit. Schema:
{{"tool": "propose_edit", "edits": [{{"tensor": "model.layers.{L}.mlp.up_proj.weight", "op": "fill_rows", "rows": "all_zeroed_rows", "value_spec": "..."}}]}}

IMPORTANT: for "rows" use the string "all_zeroed_rows" (the harness knows the exact
indices). Do NOT enumerate row indices — the list is long and will be truncated.

value_spec options (choose one and justify in "reasoning"):
- "layer_mean_fill": each row filled with the layer's mean weight value
- "layer_std_sample": rows filled with random normal samples (mean=layer mean, std=layer std)
- "sister_layer_copy": copy rows from a healthy sister layer's up_proj
- "scaled_sister": sister rows scaled to match this layer's std
- custom: describe your own strategy

Output ONE JSON object. Budget: 1 call.
{tool_manual}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A1", "A2", "A3", "A4"], required=True)
    args = ap.parse_args()

    sealed = json.loads((ROOT / SEALED).read_text(encoding="utf-8"))
    truth = sealed["rows"]  # {tensor_name: [row indices]}

    out_path = ROOT / "results/exp4_repair.json"
    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else []
    if any(r["arm"] == args.arm for r in results):
        print(f"[exp4] {args.arm} already done")
        return

    # ---- A3 oracle-full：V0 原值恢复（直接复制，等价 DS-120 基线）----
    if args.arm == "A3":
        results.append({
            "arm": "A3", "method": "restore_from_V0",
            "note": "oracle = exact original values; DS-120 baseline 0.783 applies",
        })
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print("[exp4] A3 recorded (oracle = V0 baseline)")
        return

    # ---- A4 random-edit：随机层+随机行同预算 ----
    if args.arm == "A4":
        import subprocess

        g = torch.Generator().manual_seed(99)
        rand_layers = torch.randperm(36, generator=g)[:3].tolist()
        cmd = [
            str(ROOT / ".venv/Scripts/python.exe"), "-X", "utf8",
            str(ROOT / "scripts/inject_defect_p2.py"),
            "--src", str(ROOT / DEFECT), "--dst", str(ROOT / "checkpoints/p2/A4_random_edit"),
            "--select", "norm", "--op", "zero", "--k", "512",
            "--layers", ",".join(map(str, rand_layers)),
            "--record", str(ROOT / "experiments/iteration_2/_sealed/A4_random_edit.json"),
        ]
        subprocess.run(cmd, check=True, cwd=str(ROOT))
        results.append({"arm": "A4", "method": f"random 3 layers {rand_layers} x 512 rows zeroed (on top of defect)"})
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[exp4] A4 injected (random layers {rand_layers} on defect)")
        return

    # ---- A1/A2 需要 Agent 生成修复值 ----
    model, tokenizer = load_model(str(ROOT / DEFECT))
    model.eval()

    if args.arm == "A2":
        # oracle 定位：告知全部 3 层病灶与行号
        L = 22  # 提示里演示一层（其余两层同样处理）
        k = 512
        prompt = A2_PROMPT.format(L=L, k=k, tool_manual=TOOL_MANUAL)
        prompt += "\n\nThe same repair applies to layers 22, 28, 30 (all mlp.up_proj, 512 rows each). Propose the repair for the demonstrated layer; the harness will apply your strategy to all three."
        gen = generate(model, tokenizer, prompt, max_new_tokens=1024, enable_thinking=False)
        call = extract_tool_call(gen["response"])
        print(f"[exp4] A2 proposal: {json.dumps(call, ensure_ascii=False)[:500]}")
        results.append({"arm": "A2", "raw_output": gen["response"][:2000], "proposal": call})
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    else:  # A1
        print("[exp4] A1 (agent full chain) uses B-arm scaffolded diagnosis + repair")
        print("[exp4] given experiment-3 finding (0/3 projection hits), A1 is expected to fail localization;")
        print("[exp4] record as design note: run A1 only if user requests the full-chain failure demo")


if __name__ == "__main__":
    main()
