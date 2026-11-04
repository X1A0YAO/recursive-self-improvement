"""模型加载与推理工具（Phase 0：只读使用）。

- 从本地目录加载 Qwen3-4B，模型目录绝不写入。
- device_map="auto" 由 accelerate 决定 GPU/CPU 分布（8GB 显存需 offload）。
- 提供 capture_activations()：注册 hook 抓取指定层的输出，供 Inspector 使用。
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Iterable, Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-4B"


def load_model(model_dir: str | Path, dtype=torch.bfloat16, **kwargs):
    """加载模型与 tokenizer。默认 BF16 + auto device map。"""
    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        device_map="auto",
        **kwargs,
    )
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    do_sample: bool = False,
    enable_thinking: bool | None = None,
) -> dict:
    """单条生成。默认贪心解码以保证可复现。

    enable_thinking: Qwen3 特有。None=用 chat template 默认（开启思考模式）；
    False 时通过 template 参数关闭思考。
    """
    if enable_thinking is None:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature

    output_ids = model.generate(**inputs, **gen_kwargs)
    # 去掉 prompt 部分
    new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    generated_text = tokenizer.decode(new_ids, skip_special_tokens=True)

    # Qwen3 思考模式：<think>...</think> 区块分离
    think, answer = "", generated_text
    if "</think>" in generated_text:
        idx = generated_text.index("</think>")
        think = generated_text[: idx + len("</think>")]
        answer = generated_text[idx + len("</think>") :]

    return {
        "prompt": prompt,
        "thinking": think.strip(),
        "response": answer.strip(),
        "full_text": generated_text,
        "num_new_tokens": int(new_ids.shape[0]),
    }


@contextlib.contextmanager
def capture_activations(
    model,
    module_names: Iterable[str],
) -> Iterator[dict[str, torch.Tensor]]:
    """注册 forward hook，抓取指定子模块的输出激活。

    用法：
        with capture_activations(model, ["model.layers.0.mlp.down_proj"]) as cap:
            model(**inputs)
        acts = cap  # dict: name -> tensor
    """
    handles = []
    captured: dict[str, torch.Tensor] = {}

    def hook(name):
        def _hook(_module, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor):
                captured[name] = t.detach().to("cpu", copy=True)

        return _hook

    for name in module_names:
        mod = model.get_submodule(name)
        handles.append(mod.register_forward_hook(hook(name)))

    try:
        yield captured
    finally:
        for h in handles:
            h.remove()
