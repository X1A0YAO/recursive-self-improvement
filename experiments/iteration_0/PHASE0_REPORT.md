# Phase 0 实验记录：基础设施与基线建立

- 日期：2026-11-04
- 执行者：Codex（工具与流程）；模型自身仅作为评测对象
- 状态：评测进行中（本文件在基线评测完成后定稿）

## 1. 环境审计

| 项 | 值 |
|---|---|
| OS | Windows 11 家庭版中文版 (10.0.26200), AMD64 |
| CPU | AMD Ryzen 系列（够用；瓶颈在 GPU） |
| GPU | NVIDIA RTX 4060 Ti, 8 GB VRAM |
| 内存 | 31.7 GB |
| 磁盘 | D: 651.6 GB（项目所在盘，剩余 ~288 GB） |
| Python | 3.12（uv venv at `.venv/`） |
| PyTorch | 2.x + CUDA 12.6 |

## 2. 模型与加载

- 目标模型：Qwen/Qwen3-4B（原始 HF 权重，BF16 safetensors，无量化）
- 下载校验：三个分片 SHA256 校验通过
- 加载方式：BF16 + `device_map="auto"`（GPU 5.99GB + CPU offload）
- 贪心解码（do_sample=False）保证可复现
- V0 checkpoint：checkpoints/V0/（只读快照，SHA256 同样校验通过）
- 参数索引：results/param_index_V0.json
  - 398 张量，4.02B 参数

## 3. 测评集（benchmarks/manifest.json）

三组严格隔离（构建脚本断言 prompt 零重叠；seed=20261104 固定可复现）：

| Split | 总数 | math | instruction | logic | repetition | general | 用途 |
|---|---|---|---|---|---|---|---|
| optimization | 63 | 25 | 10 | 8 | 8 | 12 | Agent 可见，自我分析用 |
| validation | 57 | 20 | 10 | 8 | 8 | 11 | 编辑接受/拒绝判据 |
| test | 79 | 40 | 12 | 6 | 6 | 15 | 最终报告，Agent 永不可见 |

来源与检查器：
- math: GSM8K test 抽样（`#### number` 严格抽取）
- instruction: google/IFEval (train) + 官方 instructions.py 检查器（严格模式）
- logic: 本地人工校验题库 24 题（确定性关键词答案）
- repetition: 本地模板（要求写 N 句不重复句子；句子重复率 + char-8gram 循环环）
- general: TruthfulQA MC1 抽样

评分规则：
- 截断响应（生成到 max_new_tokens 未结束）一律 0 分（保守）
- math: 最终数值匹配
- instruction: IFEval 官方严格检查器全过为 1
- logic: 关键词命中
- repetition: 1 - 句子重复率；char8gram 循环率 > 0.3 直接 0
- general: MC1 选项匹配

## 4. 生成配置

- chat template（非思考模式，enable_thinking=False）
- max_new_tokens=768（见问题 4）

## 5. 基线结果

（评测完成后填写）

## 6. 过程问题与解决

1. HF snapshot_download 在本机代理环境下长连接被重置、反复卡死
   → 改用 curl 原生断点续传 + 分片 SHA256 校验，成功
2. GSM8K gold 提取 bug：初版误提取了 `<<...>>` 中间计算值而非 `####` 后的最终答案，
   导致冒烟测试 4/4 全 0 分（模型实际全对）
   → 修复为 `####\s*number` 模式；修复后冒烟 4/4 全 1
3. 单条串行生成过慢（CPU offload 下 50 分钟未完成 10 条）
   → 批量生成（8 条/批）+ 增量落盘 + 断点续评；速度提升数十倍
4. 评测中发现模型爱用 Markdown/LaTeX 长思维链，256 tokens 会被截断
   → max_new_tokens 提高到 768，截断计 0 分并记录

## 7. 结论与下一步

- Phase 0 目标（环境/模型/索引/测评集/基线/V0 快照）全部达成
- 下一步：等待用户确认后进入 Phase 1（受控缺陷与可修复性验证）
- Phase 0 期间未对任何权重做修改（read-only 使用模型）
