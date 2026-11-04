"""Self-Improvement Agent 骨架（Phase 1+ 启用）。

核心原则：
- Agent = 当前版本的目标模型本身（V0 分析 V0，V1 分析 V1 …）。
- 不用外部强模型代替目标模型做诊断与方案生成。
- Agent 的输出只能是结构化修改提案（edit_protocol），不能执行任意代码。
- Agent 可见：optimization split 的题目与自身作答（不含 test split、不含评分器内部）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from edit_protocol import validate_edit


@dataclass
class AgentContext:
    model_dir: str          # 当前自身 checkpoint（只读）
    tokenizer_dir: str
    optimization_data: str  # benchmarks/optimization/data.jsonl
    param_index: str        # results/param_index_V0.json
    workdir: str            # experiments/iteration_N


def load_optimization_examples(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_analysis_prompt(examples: list[dict], own_outputs: list[dict]) -> str:
    """Phase 2：把 optimization 集上的错误样本整理成自我分析提示。"""
    raise NotImplementedError("Phase 2 实现")


def main() -> None:
    raise NotImplementedError("Phase 1/2 按实验计划启用；Phase 0 不运行本模块")


if __name__ == "__main__":
    main()
