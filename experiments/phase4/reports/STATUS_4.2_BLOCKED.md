# Phase 4.2 中期状态报告（环境阻塞）

- 日期：2026-11-09 05:20
- 状态：Agent RSI 循环**因 GPU 环境不可行暂停**，等待用户决策

## 已完成（4.0-4.1 全部 + 4.2 部分）

1. **4.0**：工程审查、数据集验证（ARC 否决 0.85 饱和；IFEval 选定 0.588）、
   预注册草案（含失败分类学 F1/F2/F3）
2. **4.1 Baseline 锁定**：分层划分 181/180/180（25 指令类型，seed=20261107）、
   V0 SHA256、Train 0.630 / **Val 0.6222** / **Test 0.6278**（分层质量 0.6pp）
3. **4.2 工具与流程**：p4_tools（task_report/weight_stats/activation_probe/
   apply_edit+rollback）+ Agent 提示 + v1/v2 两代运行器
4. **Agent 行为验证**（v1 部分轨迹）：round 0 健康——task_report →
   weight_stats → activation_probe(L29 q / L33 k)，观察策略合理

## 阻塞详情（工程事实，非实验失败）

| 尝试 | 结果 |
|---|---|
| v1 单进程（20h） | round 0 未完成；CUDA unknown error（驱动长时劣化） |
| v2 子进程隔离 + 6300MB 等待 | GPU 常驻 5.1GB 永不满足阈值 |
| v2 + 2600MB cap 共存模式 | 冒烟通过（2.39GB 加载+生成 OK），实际轮 4h 超时——常驻服务波动挤压下退化为重 CPU offload，单次 700-token 生成 >1h |

根因：机器上出现了 **~5.1GB 常驻 GPU 进程**（疑似 astrbot/GPTSoVITS 服务，
非本项目）。Agent 会话需 ~10-20 次连续生成 × 700 token；在此环境下
单轮 4h+，6 轮实验需 24h+ 纯生成时间，且随时可能被进一步挤压。

## 已保存现场（全部可复用/续跑）

- experiments/phase4/agent/seed0/working/（工作副本 + 预计算缓存）
- trajectory_v2.json + _round_state.json（断点状态）
- _round_worker.py（轮子进程脚本）
- 全部 baseline/划分/工具代码（已 commit c240a42）

## 三个选项（等用户决策）

**A. 等待窗口**：用户停掉常驻 GPU 服务一段时间（如夜间 6h），v2 直接续跑
（状态在盘，`--resume` 即续）。最省事，实验设计零改动。

**B. 降配续跑**：接受极慢节奏（每轮 2-4h），v2 加自动重试无限等待，
跑多少算多少，几天出结果。科学上可行但时间成本高。

**C. 改协议适配**：把 Agent 生成的 max_new_tokens 从 700 降到 256、
观察工具上限 6→3，将会话压缩 ~3 倍。损失：Agent 决定质量可能下降
（历史教训：压缩提示曾导致解析失败）——需小预算验证。

不做的事：不偷偷改预注册指标；不宣称"实验完成"；不删现场。

## 对照组提醒

C（Random Edit）/ D（No-op）/ E（外部参考）不依赖 Agent 生成
（C/D 只需评测，E 需一次前向），可在当前环境排队跑——但为保持
"同初始权重同协议"的可比性，建议与 B 组同窗口执行。
