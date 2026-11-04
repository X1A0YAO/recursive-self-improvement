"""建立 V0 参数索引：列出全部参数张量与统计量。

输出 results/param_index_V0.json 与人类可读摘要。
只读操作，不加载整个模型到 GPU（safetensors 流式读取）。
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open


def human(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/Qwen3-4B")
    ap.add_argument("--output", default="results/param_index_V0.json")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        index = json.loads(index_file.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = sorted(p.name for p in model_dir.glob("*.safetensors"))
        weight_map = None

    entries = []
    total_params = 0
    dtype_counts: dict[str, int] = dict()

    for shard in shards:
        path = model_dir / shard
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                numel = t.numel()
                total_params += numel
                dtype_counts[str(t.dtype)] = dtype_counts.get(str(t.dtype), 0) + numel
                entries.append(
                    {
                        "name": key,
                        "shape": list(t.shape),
                        "dtype": str(t.dtype),
                        "numel": numel,
                        "shard": shard,
                    }
                )

    entries.sort(key=lambda e: e["name"])

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_dir),
        "num_tensors": len(entries),
        "total_params": total_params,
        "total_params_human": human(total_params),
        "dtype_counts": dtype_counts,
        "tensors": entries,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"tensors={len(entries)}  total={human(total_params)}  -> {out}")

    # 控制台摘要：按模块类型分组
    from collections import defaultdict

    by_type = defaultdict(int)
    for e in entries:
        parts = e["name"].split(".")
        mod = ".".join(parts[:-1])
        mod_type = "other"
        if "self_attn" in e["name"]:
            mod_type = "attention"
        elif "mlp" in e["name"]:
            mod_type = "mlp"
        elif "norm" in e["name"] or "embed" in e["name"]:
            mod_type = "norm_or_embed"
        by_type[mod_type] += e["numel"]
    for k, v in sorted(by_type.items()):
        print(f"  {k:15s} {human(v)}")


if __name__ == "__main__":
    main()
