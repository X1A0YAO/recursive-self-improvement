"""Phase 1：Self-Diagnosis 工具协议（Agent 的"眼睛和手"）。

设计：目标模型自身（V0-defect）通过生成**结构化工具调用**（JSON）来检查自身参数，
系统执行查询并把结果以文本形式喂回模型。模型不能执行任意代码。

工具目录（Agent 可用）：
  list_layers()                          -> 所有层与模块概览（不含任何偏差信息）
  tensor_stats(layer, proj)              -> 指定张量的 mean/std/norms/分位数
  compare_tensor_stats(layer, proj, ...)  -> 跨层对比同一 proj 的统计（发现异常层的关键工具）
  activation_probe(layer, proj, prompts)  -> 用少量 probe 提示抓激活统计
  propose_edit(...)                       -> 提交修复提案（走 edit_protocol 校验）

主循环（run_self_diagnosis）：
  1. 用缺陷模型跑 optimization 子集，收集错误样本
  2. 构造诊断提示（错误样本 + 工具说明书），模型生成工具调用
  3. 执行工具，结果回填，循环若干轮
  4. 模型给出最终修复提案（结构化 JSON）
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# 工具实现（系统侧执行，Agent 只能请求）
# ---------------------------------------------------------------------------

PROJ_CHOICES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _tensor_path(model_dir: str | Path, layer: int, proj: str) -> tuple[str, Path, str]:
    """定位张量所在 shard。"""
    model_dir = Path(model_dir)
    name = (
        f"model.layers.{layer}.mlp.{proj}.weight"
        if proj in ("gate_proj", "up_proj", "down_proj")
        else f"model.layers.{layer}.self_attn.{proj}.weight"
    )
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        index = json.loads(index_file.read_text(encoding="utf-8"))
        shard = index["weight_map"].get(name)
        if shard is None:
            raise KeyError(f"张量不存在: {name}")
        return name, model_dir / shard, name
    # 单文件
    return name, model_dir / "model.safetensors", name


def tool_list_layers(model_dir: str | Path) -> str:
    """概览：层数、每类张量形状。不含统计值（避免泄露异常位置）。"""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(str(model_dir))
    n = cfg.num_hidden_layers
    h = cfg.hidden_size
    kv = getattr(cfg, "kv_head_num", None) or getattr(cfg, "num_key_value_heads", h)
    head_dim = getattr(cfg, "head_dim", h // cfg.num_attention_heads)
    inter = getattr(cfg, "intermediate_size", None)
    lines = [
        f"num_hidden_layers = {n}",
        f"hidden_size = {h}",
        f"attention projections: q_proj[{h}x{h}] k_proj[{h}x{kv*head_dim}] v_proj[{h}x{kv*head_dim}] o_proj[{h}x{h}]",
        f"mlp projections: gate_proj[{inter}x{h}] up_proj[{inter}x{h}] down_proj[{h}x{inter}]",
    ]
    return "\n".join(lines)


@torch.no_grad()
def tool_tensor_stats_from_model(model, layer: int, proj: str) -> str:
    """从已加载模型直接取张量统计（避免磁盘路径问题）。"""
    if proj in ("gate_proj", "up_proj", "down_proj"):
        name = f"model.layers.{layer}.mlp.{proj}.weight"
    else:
        name = f"model.layers.{layer}.self_attn.{proj}.weight"
    t = dict(model.named_parameters())[name].data.float().cpu()
    flat = t.flatten()
    step = max(1, t.numel() // 1_000_000)
    qs = torch.quantile(flat[::step], torch.tensor([0.01, 0.5, 0.99]))
    lines = [
        f"tensor: {name}",
        f"shape: {list(t.shape)}",
        f"mean={t.mean():.6f}  std={t.std():.6f}",
        f"l1={t.abs().mean():.6f}  l2={t.norm():.4f}  fro_norm/element={t.norm()/t.numel()**0.5:.6f}",
        f"q01={qs[0]:.6f}  q50={qs[1]:.6f}  q99={qs[2]:.6f}",
        f"max_abs={t.abs().max():.6f}",
        f"zero_rows={int((t.abs().sum(dim=1) == 0).sum()) if t.dim() == 2 else 'n/a'}",
    ]
    return "\n".join(lines)


@torch.no_grad()
def tool_compare_from_model(model, proj: str, layers: list[int] | None = None) -> str:
    """跨层对比同一投影统计（内存版）。"""
    n = model.config.num_hidden_layers
    want = layers if layers is not None else list(range(n))
    rows = []
    params = dict(model.named_parameters())
    for L in want:
        if proj in ("gate_proj", "up_proj", "down_proj"):
            name = f"model.layers.{L}.mlp.{proj}.weight"
        else:
            name = f"model.layers.{L}.self_attn.{proj}.weight"
        t = params[name].data.float().cpu()
        step = max(1, t.numel() // 1_000_000)
        q99 = torch.quantile(t.flatten()[::step], torch.tensor(0.99))
        zero_rows = int((t.abs().sum(dim=1) == 0).sum()) if t.dim() == 2 else 0
        rows.append(
            f"layer {L:2d}: mean={t.mean():+.5f} std={t.std():.5f} "
            f"l1={t.abs().mean():.5f} l2={t.norm():.3f} q99={q99:.5f} zero_rows={zero_rows}"
        )
    return "\n".join(rows)


@torch.no_grad()
def tool_tensor_stats(model_dir: str | Path, layer: int, proj: str) -> str:
    name, shard, _ = _tensor_path(model_dir, layer, proj)
    with safe_open(str(shard), framework="pt", device="cpu") as f:
        t = f.get_tensor(name).float()
    qs = torch.quantile(t.flatten()[:: max(1, t.numel() // 1_000_000)], torch.tensor([0.01, 0.5, 0.99]))
    lines = [
        f"tensor: {name}",
        f"shape: {list(t.shape)}  dtype: bf16(fp32 view)",
        f"mean={t.mean():.6f}  std={t.std():.6f}",
        f"l1={t.abs().mean():.6f}  l2={t.norm():.4f}  fro_norm/element={t.norm()/t.numel()**0.5:.6f}",
        f"q01={qs[0]:.6f}  q50={qs[1]:.6f}  q99={qs[2]:.6f}",
        f"max_abs={t.abs().max():.6f}",
    ]
    return "\n".join(lines)


@torch.no_grad()
def tool_compare_tensor_stats(model_dir: str | Path, proj: str, layers: list[int] | None = None) -> str:
    """跨层对比同一投影的统计——发现"哪个层不对劲"的核心工具。"""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(str(model_dir))
    n = cfg.num_hidden_layers
    want = layers if layers is not None else list(range(n))
    rows = []
    for L in want:
        name, shard, _ = _tensor_path(model_dir, L, proj)
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            t = f.get_tensor(name).float()
        rows.append(
            f"layer {L:2d}: mean={t.mean():+.5f} std={t.std():.5f} "
            f"l1={t.abs().mean():.5f} l2={t.norm():.3f} q99={torch.quantile(t.flatten()[:: max(1, t.numel() // 1_000_000)], torch.tensor(0.99)):.5f}"
        )
    return "\n".join(rows)


@torch.no_grad()
def tool_activation_probe(model_dir: str | Path, layer: int, proj: str, prompts: list[str], tokenizer=None, model=None) -> str:
    """抓取指定模块输出激活统计（复用已加载的 model，避免重复加载）。"""
    assert model is not None, "需要传入已加载的 model"
    if proj in ("gate_proj", "up_proj", "down_proj"):
        mod_name = f"model.layers.{layer}.mlp.{proj}"
    else:
        mod_name = f"model.layers.{layer}.self_attn.{proj}"
    handles = []
    captured: dict[str, torch.Tensor] = {}

    def hook(name):
        def _h(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor):
                captured[name] = t.detach().float().cpu()
        return _h

    mod = model.get_submodule(mod_name)
    handles.append(mod.register_forward_hook(hook(mod_name)))
    try:
        for p in prompts:
            inputs = tokenizer(p, return_tensors="pt").to(model.device)
            model(**inputs)
    finally:
        for h in handles:
            h.remove()

    if not captured:
        return "no activations captured"
    all_t = torch.cat(list(captured.values()), dim=0) if captured else None
    return (
        f"activations @ {mod_name} over {len(prompts)} prompts:\n"
        f"  mean={all_t.mean():.5f} std={all_t.std():.5f} "
        f"l2_per_token={all_t.norm()/all_t.shape[0]:.4f} "
        f"max_abs={all_t.abs().max():.4f}"
    )


TOOL_MANUAL = """你可以使用以下检查工具（每次回复只能包含一个工具调用，JSON 格式）：

1. {"tool": "list_layers"}
   概览模型结构。

2. {"tool": "tensor_stats", "layer": <int>, "proj": "<q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj>"}
   查看指定张量的统计量（mean/std/范数/分位数）。

3. {"tool": "compare_tensor_stats", "proj": "<proj>", "layers": [<int>, ...] (可选)}
   跨层对比同一投影的统计——找出哪个层的参数偏离了正常模式。

4. {"tool": "activation_probe", "layer": <int>, "proj": "<proj>", "prompts": ["...", ...]}
   用你自己的提示词探测指定模块的激活统计（最多 4 条提示）。

5. {"tool": "propose_edit", "tensor": "model.layers.<L>.mlp.<proj>.weight", "op": "scale"|"low_rank_add",
   "params": {...}, "rationale": "...", "hypothesis": "..."}
   提交最终修复提案。scale: {"factor": 0.5~2.0}；low_rank_add: {"alpha": ..., "u_hint": [...], "v_hint": [...]}（u/v 可省略，由系统用你的 rationale 中的方向说明生成）。

回复格式（严格遵守）：
只输出一个 JSON 对象，不要输出其他文本。"""


def extract_tool_call(text: str) -> dict | None:
    """从模型输出中提取第一个 JSON 工具调用。

    两种模式：
    1. 嵌套容错：找 "tool": 的最外层对象（支持 edits 数组内嵌套花括号——
       decide 调用的必需结构，旧正则无法匹配嵌套结构）
    2. 兼容旧行为：无嵌套的简单调用
    截断容错：若最外层 JSON 不完整，剥掉最后一个不完整键值再闭合重试。
    """
    # 模式 1：含嵌套的完整 JSON（平衡括号扫描）
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        end = -1
        for i in range(start, len(text)):
            c = text[i]
            if esc:
                esc = False
                continue
            if c == "\\":
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end != -1:
            cand = text[start : end + 1]
            try:
                obj = json.loads(cand)
                if isinstance(obj, dict) and "tool" in obj:
                    return obj
            except json.JSONDecodeError:
                # 尾逗号修复
                try:
                    fixed = re.sub(r",\s*([}\]])", r"\1", cand)
                    obj = json.loads(fixed)
                    if isinstance(obj, dict) and "tool" in obj:
                        return obj
                except json.JSONDecodeError:
                    pass
        start = text.find("{", start + 1)

    # 模式 2（截断容错）：找 "tool": "decide" 的起点，剥离不完整尾部后闭合
    m = re.search(r'\{\s*"tool"\s*:\s*"([a-z_]+)"', text)
    if m and m.group(1) in ("decide", "propose_edit"):
        frag = text[m.start():].rstrip()
        # 剥掉最后一个不完整的键值对（未闭合字符串/悬空逗号）
        frag = re.sub(r',\s*"[^"]*":\s*"[^"]*$', "", frag)  # 悬空 "k": "v...
        frag = re.sub(r',\s*"[^"]*":\s*\[[^\]]*$', "", frag)  # 悬空 "k": [...
        frag = re.sub(r",\s*$", "", frag)
        for closer in ('"}]}', '"}]', '"}', "}"):
            try:
                obj = json.loads(frag + closer)
                if isinstance(obj, dict) and obj.get("tool") in ("decide", "propose_edit"):
                    # 保底补全：edits 存在但缺 rationale 等字段可接受
                    return obj
            except json.JSONDecodeError:
                continue
    return None


@dataclass
class DiagnosisLog:
    rounds: list[dict] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.rounds.append({"role": role, "content": content})

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.rounds, ensure_ascii=False, indent=2), encoding="utf-8"
        )
