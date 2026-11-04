"""逐题对比 V0 与 defect_v3，看退化分布。"""

import json

quick = json.loads(open("results/quick_defect_v3.json", encoding="utf-8").read())
base = json.loads(open("results/baseline_V0.json", encoding="utf-8").read())
base_scores = {r["uid"]: r["score"] for r in base["splits"]["optimization"]["results"]}

print(f"{'uid':22s} {'V0':>4s} {'v3':>4s}  response_excerpt")
for r in quick["results"]:
    b = base_scores.get(r["uid"], -1)
    mark = " <<<" if r["score"] < b else ""
    print(f"{r['uid']:22s} {b:4.1f} {r['score']:4.1f}  {r['response'][:80]!r}{mark}")
mean_b = sum(base_scores[r['uid']] for r in quick['results']) / len(quick['results'])
mean_d = sum(r['score'] for r in quick['results']) / len(quick['results'])
print(f"\nV0={mean_b:.3f}  defect_v3={mean_d:.3f}  delta={mean_d-mean_b:+.3f}")
