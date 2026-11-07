# Phase 4 预注册草案：自主权重修改与能力提升（Autonomous Weight Self-Improvement）

- 日期：2026-11-07 15:25（**草案——待用户确认后锁定，Baseline 前定稿**）
- 范式声明：真实任务驱动的自主改进。无人工注入缺陷、无 Oracle 定位、
  无人工指定修改方向。Agent 看到任务与验证反馈，自主决定是否改、
  改哪里、怎么改、留不留。
- 历史结论作为设计约束（非失败预设）：
  - C1（EXP1 Gate-3）：4B 无法从原始数值分辨扰动 → Agent 工具须提供
    结构化统计视图而非裸数字
  - C2（EXP2）：8 行×12 轮=0.1% 覆盖 → 工具须有张量级/块级粗粒度层
  - C3（Phase 2）：teacher-forced logprob ≠ 自由生成 → 一切反馈用
    自由生成评测
  - C4（B2a/B2b/EXP2）：提案悬崖（证据收集后不提交）→ 流程强制提案
    检查点 + 每轮必须输出结构化决定
  - C5（Phase 2 A2）：修复值≈随机 → 本实验不预设修改会成功；
    判定标准见 §7（含失败判定的科学价值条款）

## 1. 数据集（已验证可用）

**主数据集：官方 IFEval（google/IFEval，541 prompts，train 单切分）**

选择理由（对照用户标准 1-7）：
1. V0 基线 ~0.56（16 题探测；160 题探测进行中）——大量改进空间 ✓
   （ARC-Challenge 已验证被否决：V0=0.85 饱和，改进空间不足）
2. 官方检查器全自动评分（本地 scripts/instructions*.py 已部署且
   在 Phase 0 修正过 None-kwarg bug）✓
3. 规模适合：官方 541 题，划分后 val 180 题一轮评估 ~35 分钟 ✓
4. 严格二值分数（指令遵循/不遵循）✓
5. 独立划分：官方无 train/test 切分 → **按指令类型分层抽样自划分**
   （见 §2，25 种 instruction id 分层保证各类型分布一致）
6. 记忆风险低：IFEval 是指令格式约束任务，非知识检索 ✓
7. 贪心解码、固定 max_new_tokens=512、温度 0——完全确定 ✓

**数据划分（seed=20261107，分层抽样）**：

| 集 | n | 用途 |
|---|---|---|
| Train（Agent 分析用） | 181 | Agent 可看全部：题目+官方指令 id+V0 的对错 |
| Validation（Agent 反馈用） | 180 | Agent 可看：评测分数+逐题对错+失败题的指令 id（不看官方检查细节）|
| Test（锁定） | 180 | Agent 永不可见；仅最终评估 |

分层键：instruction_id_list 的首 id（25 类）。三集内类型分布一致。
划分脚本落盘划分清单（uid 列表+checksum），一次生成永不重划。

## 2. 模型与资产

- V0 = checkpoints/V0（BF16，SHA256 于 Phase 0 校验过，此处在
  baseline 时重算并记录）
- 工作副本：experiments/phase4/agent/checkpoints/ 下版本化
  （V4-r0, V4-r1, ...），原模型只读
- 回滚：每轮修改前保存 edit_manifest（修改前值完整记录），
  支持逐轮回滚

## 3. Agent 工具接口（复用+新构，遵 C1/C2/C4）

### 观察类（信息带宽优先，不替 Agent 决定）
- `task_report`：Train/Val 的逐题对错汇总、按指令类型的失败统计、
  失败样本的 prompt 摘录（无官方检查细节——那是测试信息）
- `weight_stats`：张量/层/投影统计（mean/std/norm 分布、跨层对比）——
  复用 b2_tools 聚合
- `activation_probe`：给定一小批 Val 失败题，返回指定层/投影的
  激活统计（复用 neuron_activity，Agent 指定层）
- **不提供** compare-to-healthy（本实验无"健康参照"——V0 就是起点）

### 修改类（执行 Agent 的结构化决定）
- `apply_edit`：schema 化提案 {tensor, op(zero/scale/add/lowrank/
  row_scale), rows?, factor?, budget}，校验后应用到工作副本
- 预算约束：每轮 ≤2 张量、全局累计 ΔL2/‖W‖ ≤ 0.05、总轮数 ≤6

### 流程类（对抗提案悬崖 C4）
- 每轮 Agent 必须输出一个结构化 JSON 决定：
  `{"decision": "edit"|"stop"|"explore", ..., "rationale": "..."}`
- explore 不消耗修改预算但消耗轮次（≤6 总轮）

## 4. 对照组（全部同初始权重/同划分/同协议）

| 组 | 内容 | 回答的问题 |
|---|---|---|
| A Baseline | V0 原样 | 基线 |
| B Agent RSI | Agent 自主循环 | 核心问题 |
| C Random Edit | 与 B 相同预算的随机修改（随机张量×随机行×同幅度分布） | B 是否优于随机 |
| D No-op | 相同流程（评估+工具调用+假编辑回滚）零实际修改 | 排除流程/波动效应 |
| E（预算允许）外部优化参考 | 轻量任务向量加法（如 Task Arithmetic 式）仅作可达性参考 | 任务是否有改进空间（非 RSI） |

## 5. 实验流程（B 组）

```
Round 0: task_report(Train+Val) → Agent 决定
Round k: [observe tools] → apply_edit → 自动 Val 评估(180 题)
         → 反馈给 Agent → keep/rollback 决定 → 下一轮
≤6 轮或 Agent 提前 stop
```
每轮完整落盘：观察调用、修改 manifest（前后值）、Val 分数、决定、
rationale 原文。

## 6. 统计与判定（运行前锁定）

### 主指标
ΔAcc_test = Acc_test(B) − Acc_test(A)，配对 McNemar（n=180）

### 预注册判定

| # | 判定 | 阈值 |
|---|---|---|
| S1 | B 在 Test 上显著提升 | McNemar 双侧 p<0.05 且 ΔAcc ≥ +3pp |
| S2 | B > C（优于随机） | B 的 ΔAcc 显著高于 C 的 ΔAcc（比例检验 p<0.05） |
| S3 | 无副作用 | DS-120（已有通用能力集）退化 <3pp |
| S4 | 可重复 | B 的 Val 提升轨迹在 ≥2 个 Agent 会话中同向（若预算允许跑 2 seed） |

### 失败判定的科学价值（预注册，防事后解释）
- F1：S1 ✗ 且 S2 ✓ → "自主修改≈随机扰动"（能力不存在）
- F2：S1 ✗ 且 E 组显示任务可达改进 >5pp → "空间存在但 4B 自主性不足"
- F3：S1 ✗ 且 E 组也无改进 → "任务在当前修改预算下不可达"（工具/预算限制）
- 三种失败的下一步完全不同——预注册区分它们

## 7. 计算预算

| 项 | 时 |
|---|---|
| Baseline：Val 180 + Test 180 | ~2.2h |
| B 组：≤6 轮 × (观察+修改+Val 180) | ~4h |
| C/D 组 | ~2h |
| E 组（可选） | ~1.5h |
| 最终 Test 评估 ×4 组 | ~4.5h |
| **合计** | **~14h GPU**（分批跑，夜间为主） |

## 8. 目录结构

```
experiments/phase4/
  PREREGISTRATION.md（本文件定稿）
  DATASET.md（划分清单+分层明细）
  baseline/（V0 的 Val/Test 结果+权重 SHA256）
  agent/（B 组轨迹、工作副本、edit manifests）
  controls/（C/D/E 组）
  results/（最终评估）
  reports/（阶段报告）
```

## 9. 阶段门（用户检查点）

- 4.0 完成：本草案 + 160 题探测结果 → **等用户确认**
- 4.1 Baseline（Val+Test 全量）→ 报告后停
- 4.2 B 组 pilot 1 轮 → 检查流程 → 全程
- 4.3-4.5 按序，每阶段独立报告

## 10. 待确认问题（用户决策）

1. 预算上限（每轮 ≤2 张量、6 轮、全局 ΔL2≤0.05）是否合适？
   更大预算=更大搜索空间=更难归因
2. E 组（外部优化参考）是否运行？（+1.5h GPU，但区分 F2/F3 必须）
3. Agent 会话重复（S4）是否需要？（+4h GPU）
