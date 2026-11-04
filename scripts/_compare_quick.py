"""对比 quick_eval 样本在基线与缺陷模型上的得分。"""

import json

quick = json.loads(open("results/quick_defect.json", encoding="utf-8").read())
base = json.loads(open("results/baseline_V0.json", encoding="utf-8").read())

base_scores = {r["uid"]: r["score"] for r in base["splits"]["optimization"]["results"]}

print(f"{'uid':22s} {'group':8s} {'V0':>5s} {'defect':>7s}")
d_sum = b_sum = 0.0
n = 0
for r in quick["results"]:
    uid = r["uid"]
    b = base_scores.get(uid)
    if b is None:
        continue
    print(f"{uid:22s} {r['group']:8s} {b:5.2f} {r['score']:7.2f}")
    b_sum += b
    d_sum += r["score"]
    n += 1
print(f"\nV0 mean: {b_sum/n:.3f}   defect mean: {d_sum/n:.3f}   delta: {(d_sum-b_sum)/n:+.3f}")
