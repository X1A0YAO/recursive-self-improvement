# Recursive Self-Improvement via Weight-Space Programming

An experimental framework for studying whether an LLM can analyze its own
parameters and directly edit its weights to improve itself - and whether the
improved model can improve itself further, forming a recursive positive loop.

> **Status: COMPLETE (Phases 0-4). Conclusion: Qwen3-4B cannot achieve
> tool-mediated weight self-improvement - every link in the chain (defect
> visibility -> localization -> repair -> capability gain) fails with
> quantified mechanisms: behavioral redundancy (8-layer lesion = noise),
> statistical invisibility (z-score detection 0/8), norm != importance
> (2.3% overlap), observation hallucination, agent edits counterproductive
> vs random (+2.8pp random > 0 agent). Full evidence: reports under
> `experiments/`. All negative results preregistered and reproducible.**

## Research Positioning (vs. Takehi et al. 2026, "Sparks of Weight-Space Programming")

Full analysis in `RESEARCH_POSITIONING_REVIEW.md`. Four capability levels must
be distinguished (following Honnibal's critique of conflation in the original):

| Level | Meaning | Their work | This project |
|---|---|---|---|
| L1 raw-weight-in-context | raw floats in context, no tools | nets <=172K params only | not attempted (4B infeasible) |
| L2 tool-mediated analysis | structured queries to a harness | Python sandbox + forward passes | JSON tool protocol (stricter: no arbitrary code) |
| L3 direct weight editing | gradient-free writes | yes (0.016% of Qwen3.5) | yes (schema-validated, budgeted) |
| L4 recursive self-improvement | V0->V1->V2 chain | listed as future work | **core question** |

Key differences: (1) we study a 4B open model inspecting **its own** weights
(reflexive, RSI-relevant) rather than a frontier model inspecting third-party
networks; (2) their headline success (MATH-500 7.9->48.1) repaired a
behaviorally self-exposing defect (93% truncation/looping rate) using Python
and 50 batches of environment feedback, while we map the
**sub-behavioral-threshold defect** regime under preregistered statistics
(McNemar/Holm), sealed defect records, and oracle/random control arms; (3) their
own additional experiment (Gemma, which doesn't loop) failed to improve - the
hard region we deliberately probe with mechanisms (norm != importance;
statistical visibility limits) established by our audit.

## Research Questions

1. Can a model (Qwen3-4B) diagnose its own behavioral weaknesses by inspecting
   its weights and activations?
2. Can it propose structured, effective weight edits (no gradients, no
   backprop, no fine-tuning)?
3. Do edited models (V1, V2, ...) become better self-improvers (meta-capability
   M(W_t)) or only fix their own defects (capability C(W_t))?

| Mode | Description |
|------|-------------|
| External editing | Model A edits model B |
| Self-editing | Model A analyzes & edits **itself** |
| Recursive self-improvement (RSI) | Edited A' continues to edit itself |

## Hard Constraints

- **Forbidden:** gradient descent, backpropagation, optimizers, SFT/LoRA/QLoRA,
  any retraining of the target model.
- **Allowed:** reading/analyzing parameters & activations, statistics,
  matrix ops, direct numerical weight edits (e.g. low-rank updates
  `W' = W + UV`), within a structured edit protocol with budgets and
  safety checks.
- Original HF checkpoint is read-only; every edit creates a new versioned
  checkpoint under `checkpoints/`.
- The Evaluator is independent from the Self-Improvement Agent; the agent
  never sees evaluation code, test labels, or scoring logic.

## Repository Layout

```
weight-space-rsi/
|-- models/Qwen3-4B/        # original HF weights (read-only, gitignored)
|-- checkpoints/            # V0, V1, V2 ... versioned checkpoints (gitignored)
|-- scripts/                # all runners, evaluators, audits
|   |-- check_environment.py, download_model.py, build_benchmarks.py
|   |-- inspect_weights.py, edit_weights.py, evaluate.py, inference.py
|   |-- inject_defect*.py, lesion_calib*.py, matrix_screen/confirm.py
|   |-- run_phase1.py, run_p3_exp{1,2}.py, run_p4_agent*.py
|   |-- run_p4_control_{C,E}.py, run_exp3b_b2{a,b}.py
|   |-- eval_ifeval*.py, eval_arc.py, split_ifeval_p4.py
|   `-- test_weight_access.py, test_b2_tools.py   # unit tests
|-- agent/                  # JSON tool protocol (no arbitrary code)
|   |-- self_improvement.py, edit_protocol.py, diagnosis_tools.py
|   |-- weight_access_tools.py, b2_tools.py, p4_tools.py
|-- benchmarks/             # ds120/, ifeval_p4/, optimization/, validation/, test/
|-- experiments/            # per-phase records (prereg + reports + sealed)
|   |-- iteration_0..3/     # Phases 0-3 (defect injection, diagnosis, repair)
|   `-- phase4/             # real-task autonomous editing (5 arms)
|-- results/                # evaluation outputs (lightweight JSON kept)
`-- logs/                   # run logs (large logs gitignored)
```

## Experiment Phases (all complete)

- **Phase 0** - environment, model load, param index, baseline eval, V0
  checkpoint. Report: `experiments/iteration_0/PHASE0_REPORT.md`.
- **Phase 1** - controlled reversible defects + self-diagnosis/repair, with
  random-edit and oracle control groups. Key findings: behavioral degradation
  statistically insignificant (McNemar p >= 0.125); z-score-only detection
  0/8; norm is not importance (2.3% overlap; counterfactual logprob +1.02 for
  top-norm vs -3.41 for top-activity).
  Reports: `experiments/iteration_1/`.
- **Phase 2** - genuine self-editing with budget discipline; teacher-forced
  logprob diverges from free-generation behavior; aggregate-projection and
  budget-allocation studies (B2a/B2b). Reports: `experiments/iteration_2/`.
- **Phase 3** - raw weight inspection (read-behavior measurement; observation
  hallucination: 3/3 identical reports with zero tool calls) and blind
  diagnosis across 18 defect cells (proposal cliff).
  Reports: `experiments/iteration_3/`.
- **Phase 4** - autonomous weight self-improvement on a real downstream task
  (IFEval, 181/180/180 stratified split). Five arms: A baseline / B agent
  RSI (dual seed) / C random edit / D no-op / E random-search reference.
  Result: agent edits counterproductive (both real edits -3.3pp val,
  rolled back), random edit +2.8pp (n.s.), search space itself narrow at
  0.05 rel-L2 budget. Report: `experiments/phase4/PHASE4_REPORT.md`.

## Final Conclusion (Phases 0-4)

The tool-mediated weight self-improvement chain fails at **every link** for a
4B model, each with a quantified mechanism:

1. **Defect visibility**: sub-behavioral defects are statistically invisible
   (z-score AUC 0.08-0.44; 8-layer lesions within noise).
2. **Localization**: norm-based signals are anti-correlated with function;
   raw-number reading covers ~0.1% of the parameter space per session.
3. **Repair**: agent-proposed edits systematically hurt (both seeds, same
   L33-attention edit, -3.3pp each) while random edits slightly help -
   the diagnostic signal is worse than chance.
4. **Capability gain**: at 0.05 rel-L2 budget the improvement space itself is
   narrow (8-direction random search: mean -0.4pp, max +1.7pp/60q).

All experiments were preregistered before execution; negative results are
reported in full. See `PROGRESS_REPORT.md` for the consolidated narrative and
`experiments/*/` for per-phase evidence.

## Environment

- Windows 11, Python 3.12 (venv at `.venv/`), PyTorch + CUDA.
- Target model: **Qwen/Qwen3-4B** (BF16 safetensors, no quantization).
- Hardware: NVIDIA RTX 4060 Ti 8GB, 32GB RAM - model loaded via
  `device_map="auto"` with CPU offload.

Setup:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv\Scripts\python.exe transformers accelerate safetensors datasets
```

## Reproduction Entry Points

```powershell
# Phase 0: environment + baseline
.venv\Scripts\python.exe scripts\check_environment.py
.venv\Scripts\python.exe scripts\evaluate.py --splits optimization,validation,test

# Phase 4: split + baseline + agent loop + controls
.venv\Scripts\python.exe scripts\split_ifeval_p4.py
.venv\Scripts\python.exe scripts\eval_ifeval_p4.py --model-dir checkpoints\V0 --split test --output experiments\phase4\baseline\test_V0.json
.venv\Scripts\python.exe scripts\run_p4_agent_v2.py --seed 0 --max-rounds 6
.venv\Scripts\python.exe scripts\run_p4_control_C.py

# Unit tests (tool layer)
.venv\Scripts\python.exe scripts\test_weight_access.py
.venv\Scripts\python.exe scripts\test_b2_tools.py
```

Evaluation splits are strictly isolated (`benchmarks/manifest.json` records
provenance and per-example checksums): optimization/validation splits are
agent-visible at different levels; the test split is **never** agent-visible.
Decoding is greedy (deterministic) for reproducibility.
