# Phase 4.1 Baseline 报告

- 日期：2026-11-07 19:30
- 模型：checkpoints/V0（Qwen3-4B BF16）
- 权重 SHA256（见 baseline/V0_sha256.txt）：
  - shard1 328A91D3122359D5…  shard2 6CD087B316306A68…  shard3 E4BF436957184F4E…

## 结果

| 集 | n | 准确率 |
|---|---|---|
| Validation | 180 | **0.6222** |
| Test | 180 | **0.6278** |

- Val/Test 差 0.6pp——分层划分（25 指令类型）质量良好
- 与 160 题探测（0.588）一致（子集方差内）

## 配置（锁定）

- eval_ifeval_p4.py：贪心解码、max_new_tokens=512、batch 8、
  官方检查器（本地 instructions*.py）、截断启发式（padding bug 已修）
- 逐题结果落盘（val_V0.json / test_V0.json，含响应与指令类型）
- 断点续跑验证：进度 jsonl 完整（180+180 行）

## 指令类型基线摘要（Test，按失败数排序前 8）

（见 test_V0.json by_instruction_type；Agent 将看到 Train 版的同类统计）

## Test 集锁定声明

自本报告起，test_V0.json 的逐题信息对 Agent 永不可见；
后续所有臂共用此基线做 McNemar 对比。Train/Val 的 baseline 结果
可供 Agent 观察（task_report 工具）。

## 下一步（4.2）

Agent RSI 循环实现（agent/p4_tools.py 已就位）：
pilot 1 轮 → 检查流程 → 全程（2 seed）。
