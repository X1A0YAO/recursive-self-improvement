"""鎵板姩鏁忔劅搴︽壂鎻忥細涓嶅悓 sigma_frac 涓嬬殑 PPL 鍙樺寲锛堜笉钀界洏 checkpoint锛屽唴瀛樹腑鏀癸級銆?
鐢ㄤ竴灏忔鍥哄畾鏂囨湰娴?log姒傜巼锛屽揩閫熷畾浣?鏈夋劅闃堝€?銆?"""

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

TENSOR = "model.layers.26.mlp.up_proj.weight"
MODULE = "model.layers.26.mlp.up_proj"


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
    print("[scan] loading V0", flush=True)
    model, tok = load_model("checkpoints/V0")
    base_lp = mean_logprob(model, tok, PROBE_TEXTS)
    print(f"[scan] V0 mean logprob/token = {base_lp:.4f}", flush=True)

    W0 = dict(model.named_parameters())[TENSOR].data.clone()

    import math

    for frac in (0.3, 0.45, 0.6, 0.8, 1.0):
        g = torch.Generator(device="cpu").manual_seed(2021599929)
        m_, n_ = W0.shape
        r = 4
        U = torch.randn(m_, r, generator=g, dtype=torch.float32)
        V = torch.randn(n_, r, generator=g, dtype=torch.float32)
        U = U.to(W0.device)
        V = V.to(W0.device)
        U, _ = torch.linalg.qr(U)
        V, _ = torch.linalg.qr(V)
        delta = (U @ V.T) / math.sqrt(r) * (frac * W0.float().norm().item())
        model.get_submodule(MODULE).weight.data = (W0.float() + delta).to(W0.dtype)
        lp = mean_logprob(model, tok, PROBE_TEXTS)
        print(f"[scan] frac={frac:.2f}  logprob/token={lp:.4f}  delta={lp-base_lp:+.4f}", flush=True)
        model.get_submodule(MODULE).weight.data = W0  # 鎭㈠


if __name__ == "__main__":
    main()
