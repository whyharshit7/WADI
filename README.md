# WADI: Wavefront Adaptive-Depth Inference

> Self-speculative decoding for LLMs — draft with the model's own intermediate layers, no separate draft model.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

WADI attaches lightweight exit heads at layers `L/4`, `L/2`, `3L/4`. During decoding, easy tokens exit at a shallow layer; hard tokens run deeper. K drafts are verified in a single batched full-depth pass with rejection sampling, so the output distribution is **provably identical to the full model**.

Three backends:
- **NumPy reference** (`model.py`, `wadi.py`) — algorithm in ~500 lines, easy to read and modify
- **PyTorch port** (`model_torch.py`, `wadi_torch.py`) — fused QKV, SDPA attention, preallocated KV cache
- **HuggingFace runner** (`wadi_hf.py`) — end-to-end on any Llama/Qwen/Mistral-family model

## Algorithm

1. **Draft.** Generate K tokens with early exit. Each stops at the shallowest layer where the exit-head's entropy falls below that layer's threshold.
2. **Verify.** Run all K drafts through the full model in one batched pass. Accept each with probability `min(1, p_full / p_draft)`.
3. **Adapt.** On rejection, resample from `max(0, p_full − p_draft)` (renormalized) and discard remaining drafts. Nudge thresholds toward a target acceptance rate.
4. **Repeat.**

## Results

**16-layer controlled PoC (NumPy, simulated-trained exits):**

| Method                | Seq Passes | Speedup   | Draft Model  |
|-----------------------|:----------:|:---------:|:------------:|
| Standard AR           |     60     |  1.00×    | —            |
| Speculative Decoding  |     22     |  2.67×    | Separate 4L  |
| **WADI**              |   **21**   | **2.86×** | Self-draft   |

This validates the algorithm on a setting where exit heads are assumed well-distilled. Projected at 85% acceptance with trained exits: ~3.7× (125M), ~4.1× (7B), ~4.3× (30B), ~4.4× (70B).

**Qwen2.5-7B with light distillation (HF runner, 3K steps / 3M tokens):**

- Lossless output — rejection sampling preserves the base distribution (verified numerically).
- 26% overall acceptance, 33% at L21, degrading for shallower exits.
- Wall-clock below AR in this regime: shallow heads are undertrained, so most drafts hit full depth.

Full experimental results, exit distributions, and analysis in [RESULTS.md](RESULTS.md).

The toy-to-real gap is the distillation budget — all early-exit methods (Medusa, LayerSkip, CALM) require 100M–1B tokens and hours of training to land their published speedups. This repo demonstrates the algorithm and implementation; production-grade head training is future work.

## Quick start

```bash
pip install -r requirements.txt                     # numpy, scipy
pip install torch                                   # optional: GPU backend
pip install transformers datasets accelerate        # optional: HF runner
```

**NumPy reference benchmark:**

```bash
python benchmark.py
```

**PyTorch benchmark (auto-uses CUDA):**

```bash
python benchmark_torch.py
```

**End-to-end on a real model:**

```bash
python wadi_hf.py --model Qwen/Qwen2.5-7B \
                  --heads-ckpt heads.pt \
                  --distill-steps 3000 \
                  --exit-layers 14,21
```

See `python wadi_hf.py --help` for all options.

## Library usage

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
    initial_thresholds={8: 2.0, 16: 3.0, 24: 4.5, 32: float("inf")},
    max_draft_len=6,
))
output = engine.generate(np.array([1, 50, 200, 15]), max_new_tokens=100)
print(engine.get_stats().summary())
```

The Torch backend (`wadi_torch.WADIEngineTorch`) has the same API. `TorchTransformer.load_from_numpy(np_model)` copies weights and distilled exit heads from the NumPy reference for apples-to-apples comparison.

## Project layout

```
model.py            NumPy transformer with KV cache and exit heads
wadi.py             WADI engine + AR and speculative-decoding baselines
benchmark.py        NumPy benchmark (entropy profile, threshold sweep, scaling projection)
model_torch.py      PyTorch port (nn.Module, SDPA, preallocated KV cache, load_from_numpy)
wadi_torch.py       WADI engine, Torch backend
benchmark_torch.py  Torch sanity benchmark
wadi_hf.py          End-to-end runner on HuggingFace models (distillation + benchmark)
RESULTS.md          Experimental results and analysis
```

## Comparison

| Property            | Std AR | Spec. Dec. | Medusa | LayerSkip | **WADI**      |
|---------------------|:------:|:----------:|:------:|:---------:|:-------------:|
| Separate draft      |   —    |    yes     |   no   |    no     |    **no**     |
| Memory overhead     |   1×   |    ~2×     |  ~1.1× |    ~1×    |  **~1.01×**   |
| Lossless            |  yes   |    yes     |  ~no   |    yes    |    **yes**    |
| Per-token adaptive  |   no   |     no     |   no   |  partial  |    **yes**    |
| Self-tuning         |   —    |     no     |   no   |    no     | **yes**       |

## Status and limitations

This is a research-grade implementation. The algorithm is provably lossless (verified). Speedup on production-scale models depends on the quality of the distilled exit heads, which is the well-known training bottleneck of all early-exit methods. The `wadi_hf.py` runner documents its own distillation recipe so anyone can scale up training if desired.

Wavefront pipelining — overlapping draft-token computation across layer groups via CUDA streams — is algorithmic only in this repo. A true parallel-latency implementation is a natural extension.

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
