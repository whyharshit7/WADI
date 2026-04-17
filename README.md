# WADI: Wavefront Adaptive-Depth Inference

> A self-speculative decoding algorithm for LLM inference — no draft model needed.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## What it is

WADI uses a transformer's own intermediate layers as its draft model. Lightweight exit heads at layers L/4, L/2, 3L/4 let easy tokens exit early while hard tokens run deeper. Drafts are verified with standard rejection sampling, so the output distribution is **identical to the full model**.

**Result:** up to **2.86× latency speedup** on the PoC, projected **3.9–4.4×** at 70B scale — with no separate draft model and ~1% memory overhead.

## Algorithm

1. **Draft** — generate K tokens with early exit. Each token stops at the shallowest layer where entropy falls below its threshold.
2. **Verify** — run all K drafts through the full model in one pass. Accept each with probability `min(1, p_full / p_draft)`.
3. **Adapt** — on rejection, resample from `max(0, p_full − p_draft)` (renormalized) and discard remaining drafts. Nudge thresholds toward a target acceptance rate.
4. **Repeat.**

## Results (16-layer PoC)

| Method                | Seq Passes | Speedup | Draft Model  |
|-----------------------|:----------:|:-------:|:------------:|
| Standard AR           |     60     |  1.00×  | —            |
| Speculative Decoding  |     22     |  2.67×  | Separate 4L  |
| **WADI**              |   **21**   | **2.86×** | Self-draft |

Projected at 85% acceptance with trained exits: ~3.7× (125M), ~4.1× (7B), ~4.3× (30B), ~4.4× (70B).

## Quick start

```bash
pip install numpy scipy          # reference implementation
pip install torch                # optional: Torch/GPU backend
```

```bash
python benchmark.py              # NumPy reference
python benchmark_torch.py        # Torch backend (auto-uses CUDA)
```

### NumPy

```python
from model import Transformer, TransformerConfig
from wadi import WADIEngine, WADIConfig
import numpy as np

model = Transformer(TransformerConfig(
    vocab_size=32000, n_layers=32, n_heads=16,
    d_model=512, d_ff=2048, max_seq_len=1024,
    exit_layers=[8, 16, 24, 32],
))
model.simulate_trained_exits(n_calibration=500)

engine = WADIEngine(model, WADIConfig(
    initial_thresholds={8: 2.0, 16: 3.0, 24: 4.5, 32: float('inf')},
    max_draft_len=6,
))
output = engine.generate(np.array([1, 50, 200, 15]), max_new_tokens=100)
print(engine.get_stats().summary())
```

### Torch (GPU)

```python
from model_torch import TorchTransformer, TorchTransformerConfig
from wadi_torch import WADIEngineTorch, WADITorchConfig

t_model = TorchTransformer(TorchTransformerConfig(...)).to("cuda")
t_model.load_from_numpy(np_model)           # optional: reuse distilled NumPy exits
engine = WADIEngineTorch(t_model, WADITorchConfig(max_draft_len=6))
output = engine.generate([1, 50, 200, 15], max_new_tokens=100)
```

## Project layout

```
model.py           NumPy transformer with KV-cache & exit heads
wadi.py            WADI engine + AR and speculative baselines
benchmark.py       NumPy benchmark (entropy profile, threshold sweep, scaling projection)
model_torch.py     PyTorch port (nn.Module, SDPA, preallocated KV-cache, load_from_numpy)
wadi_torch.py      WADI engine, Torch backend
benchmark_torch.py Torch sanity benchmark
```

## Comparison

| Property            | Std AR | Spec. Dec. | Medusa | LayerSkip | **WADI**      |
|---------------------|:------:|:----------:|:------:|:---------:|:-------------:|
| Separate draft      |   —    |    yes     |   no   |    no     |    **no**     |
| Memory overhead     |   1×   |    ~2×     |  ~1.1× |    ~1×    |  **~1.01×**   |
| Lossless            |  yes   |    yes     |  ~no   |    yes    |    **yes**    |
| Per-token adaptive  |   no   |     no     |   no   |  partial  |    **yes**    |
| Self-tuning         |   —    |     no     |   no   |    no     | **yes**       |

## Limitations

This is a proof of concept. The model uses random weights and simulated distillation, so acceptance rates are lower than a trained deployment. Wavefront pipelining is algorithmic only — realizing it as true parallel latency requires CUDA streams (or similar) that overlap draft-token computation across layer groups.

## Extending to a real model

1. Add `nn.Linear(d_model, vocab_size)` exit heads (with LayerNorm) at L/4, L/2, 3L/4.
2. Distill: freeze the backbone and train exit heads against the final layer's distribution (KL divergence, ~10K examples).
3. In the forward pass, check entropy at each exit and break early.
4. Verify drafts with the standard rejection-sampling step — same math as existing speculative-decoding implementations.
5. For maximum latency gains, overlap draft-token layer groups with CUDA streams.

## Citation

```bibtex
@software{wadi2026,
  title  = {WADI: Wavefront Adaptive-Depth Inference},
  author = {Harshit Saxena},
  year   = {2026},
  url    = {https://github.com/whyharshit7/wadi}
}
```

## License

MIT — see [LICENSE](LICENSE).
