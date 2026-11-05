"""捕获指定层 up_proj 的任务集活跃度向量（预注册 §0 定义）。

用法：
  python scripts/capture_activity.py --layers 26
产出：results/activity_L{L}.pt（[9728] float32 CPU tensor）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from model_utils import load_model  # noqa: E402
from neuron_activity import capture_layer_activity, load_reference_prompts  # noqa: E402

ROOT = Path(__file__).parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", required=True, help="逗号分隔层号")
    ap.add_argument("--model-dir", default="checkpoints/V0")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    prompts = load_reference_prompts()
    print(f"[activity] {len(prompts)} reference prompts (optimization math)", flush=True)

    model, tokenizer = load_model(str(ROOT / args.model_dir))
    model.eval()
    for L in layers:
        a = capture_layer_activity(model, tokenizer, prompts, L)
        out = ROOT / f"results/activity_L{L}.pt"
        torch.save(a, out)
        stats = {
            "layer": L,
            "n_neurons": a.shape[0],
            "mean": a.mean().item(),
            "median": a.median().item(),
            "q90": a.quantile(0.9).item(),
            "q99": a.quantile(0.99).item(),
            "max": a.max().item(),
        }
        print(f"[activity] L{L}: mean={stats['mean']:.4f} median={stats['median']:.4f} q99={stats['q99']:.4f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
