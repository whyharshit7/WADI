# WADI: Wavefront Adaptive-Depth Inference

> A self-speculative decoding algorithm for LLM inference — no draft model needed.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![NumPy](https://img.shields.io/badge/built%20with-NumPy-013243.svg)](https://numpy.org/)

---

## TL;DR

WADI uses a transformer's own intermediate layers as draft models for speculative decoding. Lightweight exit heads at layers L/4, L/2, 3L/4 predict whether a token is "easy" (low entropy → exit early) or "hard" (high entropy → continue deeper). Draft tokens are verified via rejection sampling, making the output **mathematically identical** to the full model.

**Result:** Up to **2.86x latency speedup** on this PoC, with projected **3.9–4.4x** at 70B scale — without training or serving a separate draft model.

## How It Works

```
Token → Embed → [Layers 1–4] → Exit Head₁ → entropy < τ₁? ──YES──→ DRAFT TOKEN
                     │                                                    
                     NO                                                  
                     ↓                                                   
                [Layers 5–8] → Exit Head₂ → entropy < τ₂? ──YES──→ DRAFT TOKEN
                     │                                                    
                     NO                                                  
                     ↓                                                   
                [Layers 9–12] → Exit Head₃ → entropy < τ₃? ──YES──→ DRAFT TOKEN
                     │                                                    
                     NO                                                  
                     ↓                                                   
                [Layers 13–16] → Final Head ──────────────────────→ DRAFT TOKEN

        ┌──────────────────────────────────────────────────┐
        │  VERIFY: Full forward pass on all K draft tokens │
        │  Accept each with P = min(1, p_full / p_draft)   │
        │  → Lossless: output distribution = full model    │
        └──────────────────────────────────────────────────┘
```

### The Algorithm (4 Phases)

1. **Draft** — Generate K tokens using early exit. Each token stops at the shallowest layer where entropy drops below a threshold. Tokens using 25% of layers cost 25% of FLOPs.

2. **Verify** — Run all K drafts through the full model. Accept each token with probability `min(1, p_full / p_draft)`. On rejection, resample from the adjusted distribution and discard remaining drafts.

3. **Adapt** — Tune entropy thresholds based on acceptance rate. Low acceptance → tighten thresholds → deeper exits → better drafts. High acceptance → loosen → shallower exits → more savings.

4. **Repeat** — The accepted tokens become context for the next draft round.

## Results

### PoC Benchmark (16-layer model, pure NumPy)

```
Method              Seq Passes   Latency Speedup   Draft Model?
───────────────────────────────────────────────────────────────
Standard AR               60          1.00x        N/A
Speculative Decoding      22          2.67x        Yes (separate 4L model)
WADI                      21          2.86x        No (self-draft)
```

### Projected Speedups (Trained Models)

| Model Scale | Layers | Avg Exit Depth | Accept Rate | Projected Speedup |
|-------------|--------|---------------|-------------|-------------------|
| 125M        | 16     | L10           | 85%         | 3.70x             |
| 7B          | 32     | L16           | 85%         | 4.07x             |
| 30B         | 64     | L27           | 85%         | 4.28x             |
| 70B         | 96     | L38           | 85%         | 4.36x             |

### WADI Threshold Sensitivity

```
Threshold   FLOPs        Accept   Avg Depth   Speedup   Exit Distribution
─────────────────────────────────────────────────────────────────────────
P10         368,836,608    81%      13.5       2.73x     {L4:12, L8:3, L12:1, L16:52}
P30         407,371,776    65%      13.7       2.31x     {L4:14, L8:2, L16:65}
P50         384,565,248    72%      13.3       2.52x     {L4:15, L8:3, L16:57}
P70         357,040,128    85%      12.6       2.86x     {L4:18, L8:2, L16:48}  ← best
P90         397,148,160    65%      13.2       2.31x     {L4:17, L8:2, L16:61}
```

## Why WADI Over Existing Methods

| Property               | Standard AR | Speculative Decoding | Medusa  | LayerSkip | **WADI**           |
|------------------------|:-----------:|:-------------------:|:-------:|:---------:|:------------------:|
| Separate draft model   | —           | Required            | No      | No        | **No**             |
| Memory overhead        | 1x          | ~2x                 | ~1.1x   | ~1x       | **~1.01x**         |
| Lossless output        | Yes         | Yes                 | ~No     | Yes       | **Yes**            |
| Per-token adaptivity   | No          | No                  | No      | Partial   | **Yes**            |
| Self-tuning            | —           | No                  | No      | No        | **Yes (adaptive)** |
| Scales with depth      | —           | Moderate            | Weak    | Moderate  | **Strong**         |

## Quick Start

### Requirements

- Python 3.8+
- NumPy
- SciPy (for benchmarks only)
- **PyTorch (optional)** — required only for the GPU / `torch` backend

```bash
pip install numpy scipy            # reference implementation
pip install torch                  # optional: enables the Torch backend
```

### Run the Benchmark

```bash
python benchmark.py                # NumPy reference implementation
python benchmark_torch.py          # Torch backend (auto-uses CUDA if available)
```

`benchmark.py` runs all three inference strategies (Standard AR, Speculative Decoding, WADI) on a 16-layer transformer and prints a full comparison with latency analysis. `benchmark_torch.py` repeats the Torch-side sanity run with weights transferred from the NumPy model, so results are directly comparable.

### Use WADI in Your Code (NumPy backend)

```python
from model import Transformer, TransformerConfig
from wadi import WADIEngine, WADIConfig
import numpy as np

# Create model
config = TransformerConfig(
    vocab_size=32000, n_layers=32, n_heads=16,
    d_model=512, d_ff=2048, max_seq_len=1024,
    exit_layers=[8, 16, 24, 32],
)
model = Transformer(config)

# Train exit heads (in real usage, use distillation)
model.simulate_trained_exits(n_calibration=500)

# Configure WADI
wadi_config = WADIConfig(
    initial_thresholds={8: 2.0, 16: 3.0, 24: 4.5, 32: float('inf')},
    max_draft_len=6,
    target_acceptance_rate=0.80,
)

# Generate
engine = WADIEngine(model, wadi_config)
prompt = np.array([1, 50, 200, 15])  # your token IDs
output = engine.generate(prompt, max_new_tokens=100)

# Check stats
stats = engine.get_stats()
print(stats.summary())
```

### Use WADI on GPU (Torch backend)

```python
import torch
from model import Transformer, TransformerConfig               # NumPy, for distillation
from model_torch import TorchTransformer, TorchTransformerConfig
from wadi_torch import WADIEngineTorch, WADITorchConfig

# 1) Train / distill exits on the NumPy reference (fast, deterministic).
np_cfg = TransformerConfig(
    vocab_size=32000, n_layers=32, n_heads=16,
    d_model=512, d_ff=2048, max_seq_len=1024,
    exit_layers=[8, 16, 24, 32],
)
np_model = Transformer(np_cfg)
np_model.simulate_trained_exits(n_calibration=500)

# 2) Instantiate the Torch transformer with matching architecture, transfer
#    weights, and move to GPU.
t_cfg = TorchTransformerConfig(**{k: getattr(np_cfg, k)
                                   for k in TorchTransformerConfig.__dataclass_fields__})
t_model = TorchTransformer(t_cfg)
t_model.load_from_numpy(np_model)
t_model = t_model.to("cuda")

# 3) Generate on GPU.
engine = WADIEngineTorch(t_model, WADITorchConfig(max_draft_len=6))
output = engine.generate([1, 50, 200, 15], max_new_tokens=100)
```

Want to train the Torch model from scratch or fine-tune real LLM weights into it? Skip step 1 and call `t_model.train()` directly — `TorchTransformer` is a standard `nn.Module`.

## Project Structure

```
wadi/
├── model.py              # Pure NumPy transformer with KV-cache & exit heads
├── wadi.py               # WADI engine + AR and speculative decoding baselines (NumPy)
├── benchmark.py          # Full NumPy comparison benchmark
├── model_torch.py        # PyTorch transformer with KV-cache & exit heads
├── wadi_torch.py         # WADI engine using the Torch backend
├── benchmark_torch.py    # Torch sanity benchmark (auto-uses CUDA)
├── requirements.txt
├── LICENSE
└── README.md
```

### `model.py` — Transformer Implementation

- Multi-head self-attention with causal masking
- Pre-norm architecture (LayerNorm → Attention → LayerNorm → FFN)
- KV-cache with per-layer truncation (critical for early exit)
- Exit heads at configurable intermediate layers
- Simulated distillation via least-squares fitting on calibration data

### `wadi.py` — Inference Engines

- **`WADIEngine`** — Full WADI implementation: early exit drafting, rejection sampling verification, adaptive threshold tuning
- **`StandardInference`** — Vanilla autoregressive baseline
- **`SpeculativeDecodingBaseline`** — Standard speculative decoding with separate draft model

### `benchmark.py` — Analysis

- Entropy profiling across exit layers
- Threshold sensitivity sweep (P10–P90)
- Latency-oriented sequential pass counting
- Scaling projections to 70B-class models

### `model_torch.py` / `wadi_torch.py` — PyTorch Backend

A drop-in PyTorch port of the reference implementation:

- Standard `nn.Module` so the whole model is trainable, jittable, and moves to GPU with `.to("cuda")`.
- **Fused Q/K/V projection** via a single `nn.Linear(d, 3·d)`.
- Uses `torch.nn.functional.scaled_dot_product_attention` — gets Flash-style fused attention on modern GPUs automatically.
- Preallocated `TorchKVCache` with per-layer truncation (matches the NumPy version's API).
- `TorchTransformer.load_from_numpy(np_model)` — byte-for-byte weight transfer so you can distill exits with the NumPy model (cheap) and run inference with Torch (fast on GPU).
- Same `full_forward_single`, `full_forward_batch`, `forward_with_exits`, `forward_to_layer` surface as the NumPy version.

## Performance Notes

The reference implementation has been tuned for a few easy wins that keep the algorithm identical but drop wall-clock:

- **Batched verification** — a single K-token forward replaces K sequential single-token forwards in both the WADI engine and the speculative decoding baseline. Same total FLOPs, much better BLAS utilization.
- **Fused Q/K/V** — three einsums collapsed into one `(S, D) @ (D, 3·D)` matmul per layer.
- **Vectorized distillation** — `simulate_trained_exits` now runs all calibration samples through the stack in parallel (they're independent), rather than one-at-a-time with a fresh KV cache each. In the default 200-sample setup this is ~25× faster.

All optimizations preserve the public API (`full_forward_single`, `WADIEngine.generate`, etc.) and the algorithm (entropy-guided draft, rejection-sampling verify, adaptive thresholds).

## Limitations

This is a proof-of-concept demonstrating the algorithm, not a production implementation:

- **Random weights** — The model isn't trained on real data, so acceptance rates are lower than a production deployment would achieve
- **No GPU pipelining** — The wavefront parallelism (multiple tokens at staggered depths) is theoretical in this sequential NumPy implementation
- **Simulated distillation** — Exit heads are fit via least-squares on random data rather than proper knowledge distillation
- **No batching** — Real implementations would batch the verification pass across all draft tokens

## Extending to Production

To implement WADI on a real model (e.g., Llama):

1. **Add exit heads** at layers L/4, L/2, 3L/4 — each is just a single `nn.Linear(d_model, vocab_size)` with a LayerNorm
2. **Distill** — Freeze the backbone, train exit heads to match the final layer's output distribution using KL divergence on a small calibration set (~10K examples, <1 hour on a single GPU)
3. **Implement the draft loop** — Modify the forward pass to check entropy at each exit point and break early
4. **Add verification** — Standard rejection sampling (identical to existing speculative decoding implementations)
5. **Enable wavefront pipelining** — Use CUDA streams to overlap draft token computation across layer groups

## Citation

If you use WADI in your research:

```bibtex
@software{wadi2026,
  title={WADI: Wavefront Adaptive-Depth Inference},
  author={Harshit Saxena},
  year={2026},
  url={https://github.com/whyharshit7/wadi}
}
```

## License

MIT — see [LICENSE](LICENSE) for details.
