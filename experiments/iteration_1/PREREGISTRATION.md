# Phase 1 实验预注册：受控缺陷与可修改性验证

- 日期：2026-11-05（基线评测进行中写就，评测完成前定稿，防止事后改标准）
- 原则：本文件写明全部实验参数与成功标准；实验结束后按本文件判定，不得回头修改。

## 1. 实验设计

### 缺陷注入（V0 → V0-defect）

| 参数 | 值 | 理由 |
|---|---|---|
| 注入张量 | model.layers.18.mlp.down_proj.weight（ tentative，正式运行时从候选中随机抽选） | 中后层 MLP down_proj：参与度高、扰动有感 |
| 扰动形式 | 低秩加性噪声 W' = W + σ·(UVᵀ)/√r，U/V 列正交 | 低秩=少量方向受损，贴近"局部损伤"；可精确重放（记录 seed） |
| σ | 0.30 × 张量 std（sigma_frac=0.30） | 目标：明显退化但非毁灭（退化 15-40%） |
| rank | 4 | 自由度低，诊断可行 |
| seed | 运行时随机，记录密封 | 可复现 |

正式注入时层号与投影从以下候选均匀随机抽取（excluded: embed/norm/final 层，太容易或太难）：
layers ∈ {12..30}，proj ∈ {down_proj, up_proj, gate_proj, o_proj}。

密封记录：experiments/iteration_1/_sealed/defect_record.json
—— 绝不进入 Agent 提示词/工具结果；仅用于 oracle 对照与复现。

### 诊断协议（V0-defect 自我诊断）

- 行为证据：optimization 子集 ~12 条（math/logic/general 均衡），含模型自己的错误回答
- 工具：list_layers / tensor_stats / compare_tensor_stats / activation_probe / propose_edit
- 预算：≤10 轮工具调用，之后必须 propose_edit
- 提案约束（edit_protocol）：张量名必须存在于参数索引；|alpha| ≤ 0.5；scale ∈ [0.5, 2.0]；
  ≤8 张量；全局 ΔL2 ≤ 2.0
- Agent 可见信息：仅错误样本 + 工具返回值。不可见：密封记录、评测代码、test 集。

### 修复执行

- V1-repaired = V0-defect + Agent 提案（经 Weight Editor 安全检查后应用）
- 对照组：
  - oracle（E）：精确逆扰动（重放 delta 后回退）→ 上界，应≈V0
  - random（C）：随机位置等幅低秩修改 → 检验诊断价值
  - no-edit（D）：V0-defect 原样复评 → 评测噪声底

### 评测

- quick_eval（16 math + 4 logic）用于：缺陷确认、arm 筛查
- 全量（optimization/validation/test）用于：V0-defect、V1-repaired、oracle
- 贪心解码，max_new_tokens=768，与基线完全一致

## 2. 预注册成功标准

S1 缺陷可测量：V0-defect 在 quick_eval 上比 V0 退化 ≥ 10 个百分点
S2 自主定位：Agent 的 compare/tensor_stats 查询中，≥1 次命中真实缺陷张量
   （定义"命中"：查询的 (layer, proj) == 密封记录的 (layer, proj)）
S3 提案有效：V1-repaired 的 quick_eval 比 V0-defect 恢复 ≥ 30% 的退化幅度
S4 oracle 合理性：oracle 全量 optimization 得分与 V0 差 ≤ 2 个百分点（验证回滚机制）
S5 诊断价值：random arm 的恢复 < 30%（若 random 也"修好"了，说明任务有捷径，实验无效）
S6 无严重副作用：V1-repaired 在 test 上不比 V0-defect 更差超过 5 个百分点

## 3. 判定矩阵

- S1∧S2∧S3 全过 → Phase 1 成功：模型能定位并修复自身缺陷 → 进入 Phase 2
- S1∧¬S2∧S3 → 修复有效但定位存疑（可能靠 scale 粗调碰对）→ 分析提案内容后定
- S1∧S2∧¬S3 → 定位对但修改方向/幅度错 → 分析失败模式，Phase 1.5 迭代
- ¬S1 → 扰动太弱 → 加强 sigma 重新注入（最多 2 次）
- S5 失败 → 实验设计有捷径，重新设计（如改用更细粒度的评分）

## 4. 测量口径

- 主指标：optimization overall（quick_eval 用于流程内决策，最终以全量为准）
- 元能力（Phase 1 版）：定位命中率、工具轮数、提案通过校验率
- 全部轨迹（evidence/diagnosis transcript/proposal/edit manifest）落盘
