"""lesion 标定 v2：按行范数选 top-k 重要神经元置零，测 logprob。"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from model_utils import load_model  # noqa: E402

PROBE_TEXTS = [
    "The firefighters need to fundraise for a total of 9 hours to reach their goal.",
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball.",
    "Answer this multiple-choice question with the letter only.",
    "Let's break this down step by step and find the final answer.",
]

MODULE = "model.layers.26.mlp.up_proj"
SEED = 178798052


@torch.no_grad()
def mean_logprob(model, tokenizer, texts):
    total, n = 0.0, 0
    for t in texts:
        ids = tokenizer(t, return_tensors="pt").to(model.device)
        out = model(**ids, labels=ids["input_ids"])
        total += out.logits.shape[1] * out.loss.item()
        n += out.logits.shape[1]
    return total / n


def main():
    print("[calib2] loading V0", flush=True)
    model, tok = load_model("checkpoints/V0")
    base_lp = mean_logprob(model, tok, PROBE_TEXTS)
    print(f"[calib2] V0 logprob/token = {base_lp:.4f}", flush=True)

    W0 = dict(model.named_parameters())[MODULE + ".weight"].data.clone()
    n_rows = W0.shape[0]
    # 行范数 top-k（重要神经元）
    row_norms = W0.float().norm(dim=1)
    topk_idx = torch.argsort(row_norms, descending=True)

    for k in (32, 64, 128, 256):
        rows = topk_idx[:k]
        W = W0.clone()
        W[rows] = 0
        model.get_submodule(MODULE).weight.data = W
        lp = mean_logprob(model, tok, PROBE_TEXTS)
        print(f"[calib2] top-k={k:4d}  logprob/token={lp:.4f}  delta={lp-base_lp:+.4f}", flush=True)
        model.get_submodule(MODULE).weight.data = W0


if __name__ == "__main__":
    main()
