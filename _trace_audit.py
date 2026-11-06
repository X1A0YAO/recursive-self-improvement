import json, re

TRUTH_L = {22, 28, 30}
TRUTH_PROJ = "up_proj"

def classify(call):
    if not call: return "NO_CALL"
    t = call.get("tool")
    if t == "compare_all_projections": return "SCAN"
    if t == "tensor_stats":
        hit = call.get("proj") == TRUTH_PROJ and int(call.get("layer", -1)) in TRUTH_L
        return f"VERIFY_{'LESION' if hit else 'OTHER'}"
    if t == "propose_edit": return "PROPOSE"
    return t

for fname, tag in [("results/exp3b_b2a.json", "FORMAL"), ("results/exp3b_b2a_explore12.json", "EXPLORE12")]:
    data = json.loads(open(fname, encoding="utf-8").read())
    runs = data["runs"] if isinstance(data, dict) else [data]
    for r in runs:
        print(f"=== {tag} repeat {r['repeat']} ===")
        for e in r["transcript"]:
            c = e.get("parsed_call")
            cat = classify(c)
            # 模型输出里 JSON 之外的推理文本（决策依据）
            out = e.get("model_raw_output", "")
            m = re.search(r"\{.*\}", out, re.S)
            reasoning = (out[:m.start()] if m else out)[:400].strip()
            print(f"  r{e['round']:2d} {cat:14s} | {json.dumps(c, ensure_ascii=False)[:90] if c else ''}")
            if reasoning:
                print(f"       思考: {reasoning[:350]!r}")
        print()
