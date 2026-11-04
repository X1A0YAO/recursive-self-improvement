"""离线重评：用已保存的响应重新跑 IFEval 官方检查器（修复导入后）。

直接读取 results/baseline_V0.json，对 instruction 组重评，写回并输出对比。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import _ifeval_check  # noqa: E402


def rescore_file(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    changes = []
    for split, block in data["splits"].items():
        for r in block["results"]:
            if r.get("extract") != "ifeval_strict":
                continue
            # 从 benchmarks 找回原始样本（instruction_id_list/kwargs）
            uid = r["uid"]
            bench_file = Path("benchmarks") / split / "data.jsonl"
            ex = None
            for line in bench_file.open(encoding="utf-8"):
                row = json.loads(line)
                if row["uid"] == uid:
                    ex = row
                    break
            assert ex is not None, f"benchmark example not found: {uid}"
            old = r["score"]
            ok, detail = _ifeval_check(ex, r["response"])
            r["score"] = 1.0 if ok else 0.0
            r["info"]["ifeval"] = detail
            if old != r["score"]:
                changes.append((uid, old, r["score"]))
        # 重算 summary
        by_group = {}
        for r in block["results"]:
            by_group.setdefault(r["group"], []).append(r["score"])
        summary = {g: {"n": len(v), "mean": round(sum(v) / len(v), 4)} for g, v in by_group.items()}
        allv = [r["score"] for r in block["results"]]
        summary["overall"] = {"n": len(allv), "mean": round(sum(allv) / len(allv), 4)}
        block["summary"] = summary
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"path": path, "changed": len(changes), "details": changes[:20]}


if __name__ == "__main__":
    for p in sys.argv[1:]:
        out = rescore_file(p)
        print(f"\n== {p}: {out['changed']} scores changed ==")
        for uid, old, new in out["details"]:
            print(f"  {uid}: {old} -> {new}")
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        for split, block in data["splits"].items():
            s = block["summary"]
            print(f"  {split}: overall={s['overall']['mean']}  instruction={s.get('instruction', {}).get('mean')}")
