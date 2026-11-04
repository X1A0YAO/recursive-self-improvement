# Recursive Self-Improvement via Weight-Space Programming

An experimental framework for studying whether an LLM can analyze its own
parameters and directly edit its weights to improve itself 鈥?and whether the
improved model can improve itself further, forming a recursive positive loop.

> **Status: Phase 0 complete (baseline: opt 0.794 / val 0.881 / test 0.785). Phase 1 (controlled defect) authorized by user, running autonomously.**

## Research Questions

1. Can a model (Qwen3-4B) diagnose its own behavioral weaknesses by inspecting
   its weights and activations?
2. Can it propose structured, effective weight edits (no gradients, no
   backprop, no fine-tuning)?
3. Do edited models (V1, V2, ...) become better self-improvers (meta-capability
   M(W_t)) or only fix their own defects (capability C(W_t))?

Key distinction studied here:

| Mode | Description |
|------|-------------|
| External editing | Model A edits model B |
| Self-editing | Model A analyzes & edits **itself** |
| Recursive self-improvement (RSI) | Edited A' continues to edit itself |

## Hard Constraints

- **Forbidden:** gradient descent, backpropagation, optimizers, SFT/LoRA/QLoRA,
  any retraining of the target model.
- **Allowed:** reading/analyzing parameters & activations, statistics,
  matrix ops, direct numerical weight edits (e.g. low-rank updates W' = W + 伪uv岬€),
  within a structured edit protocol with budgets and safety checks.
- Original HF checkpoint is read-only; every edit creates a new versioned
  checkpoint under `checkpoints/`.
- The Evaluator is independent from the Self-Improvement Agent; the agent never
  sees evaluation code, test labels, or scoring logic.

## Repository Layout

```
weight-space-rsi/
鈹溾攢鈹€ models/Qwen3-4B/        # original HF weights (read-only, gitignored)
鈹溾攢鈹€ checkpoints/            # V0, V1, V2 ... versioned checkpoints (gitignored)
鈹溾攢鈹€ scripts/
鈹?  鈹溾攢鈹€ check_environment.py    # hardware/software inventory
鈹?  鈹溾攢鈹€ build_benchmarks.py     # optimization/validation/test datasets
鈹?  鈹溾攢鈹€ inference.py            # generation utilities
鈹?  鈹溾攢鈹€ inspect_weights.py      # Weight Inspector: tensor index & stats
鈹?  鈹溾攢鈹€ evaluate.py             # independent Evaluator
鈹?  鈹斺攢鈹€ edit_weights.py         # Weight Editor (applies validated proposals)
鈹溾攢鈹€ agent/
鈹?  鈹溾攢鈹€ self_improvement.py     # Self-Improvement Agent (runs on target model)
鈹?  鈹斺攢鈹€ edit_protocol.py        # structured edit proposal schema + validation
鈹溾攢鈹€ benchmarks/
鈹?  鈹溾攢鈹€ optimization/       # agent-visible, used for self-analysis
鈹?  鈹溾攢鈹€ validation/         # accept/reject decisions
鈹?  鈹斺攢鈹€ test/               # final reporting only, never agent-visible
鈹溾攢鈹€ experiments/            # per-iteration records (markdown + configs)
鈹溾攢鈹€ results/                # evaluation outputs (lightweight JSON/MD kept)
鈹斺攢鈹€ logs/                   # run logs (large logs gitignored)
```

## Environment

- Windows 11, Python 3.12 (uv venv at `.venv/`), PyTorch + CUDA.
- Target model: **Qwen/Qwen3-4B** (BF16 safetensors, no quantization).
- Hardware: NVIDIA RTX 4060 Ti 8GB, 32GB RAM 鈥?model loaded via
  `device_map="auto"` with CPU offload; inference-friendly, edit-friendly.

Setup:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv\Scripts\python.exe transformers accelerate safetensors datasets
```

## Running (Phase 0)

```powershell
# 1. environment report
.venv\Scripts\python.exe scripts\check_environment.py

# 2. download original model (read-only baseline)
.venv\Scripts\python.exe scripts\download_model.py

# 3. build isolated benchmark splits
.venv\Scripts\python.exe scripts\build_benchmarks.py

# 4. parameter index (Weight Inspector, read-only)
.venv\Scripts\python.exe scripts\inspect_weights.py

# 5. baseline evaluation of V0
.venv\Scripts\python.exe scripts\evaluate.py --splits optimization,validation,test
```

## Evaluation Design

Three strictly isolated splits (`benchmarks/manifest.json` records provenance
and per-example checksums):

| Split | Purpose | Visible to agent? |
|-------|---------|-------------------|
| optimization | self-analysis & edit hypothesis generation | yes |
| validation | accept/reject edited checkpoints | scores only |
| test | final capability reporting | never |

Groups per split: math (GSM8K), instruction (IFEval), logic (local verified
set), repetition (loop/degeneration probes), general (TruthfulQA MC1).

Decoding: greedy (deterministic) for reproducibility.

## Experiment Phases

- **Phase 0** 鈥?env, model load, param index, baseline eval, V0 checkpoint. *(current)*
- **Phase 1** 鈥?controlled reversible defect + self-diagnosis/repair, with
  random-edit and oracle control groups.
- **Phase 2** 鈥?genuine self-editing on optimization split, accept/reject by
  validation split.
- **Phase 3** 鈥?multi-generation recursion V0 鈫?V1 鈫?V2 鈫?..., measuring both
  C(W_t) (task capability) and M(W_t) (meta-capability).

Control arms (搂4 of the plan): A self-editing, B fixed external editor,
C random editing, D no-editing (noise floor), E oracle editing (upper bound).

## Progress Log

- 2026-11-04: Project scaffold created; environment audited (RTX 4060 Ti 8GB,
  32GB RAM); Python 3.12 venv + CUDA PyTorch being provisioned; benchmark
  builder, evaluator, edit protocol drafted. Model download & baseline eval
  pending.
