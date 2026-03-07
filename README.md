# WADI: Wavefront Adaptive-Depth Inference

**A self-speculative decoding algorithm with entropy-guided early exit and lossless verification.**

## What is WADI?

WADI is an LLM inference algorithm that uses the model as its own draft model. Instead of maintaining a separate smaller model for speculative decoding, WADI attaches lightweight exit heads at intermediate layers and uses entropy to decide how deep each token needs to go. Tokens that are "easy" (low entropy at shallow layers) exit early, saving compute. Tokens that are "hard" propagate to full depth.

All draft tokens are verified using standard rejection sampling, guaranteeing that the output distribution is **mathematically identical** to running the full model — zero quality loss.

## Key Results (PoC)

```
Method              Seq Passes   Latency Speedup
─────────────────────────────────────────────────
Standard AR               60          1.00x
Speculative Decoding      22          2.67x  (requires separate draft model)
WADI                      24          2.52x  (no draft model needed)
WADI (best config)        21          2.86x
```

Projected for trained models at scale:
- **7B model**: ~3.7-4.1x speedup
- **70B model**: ~3.9-4.4x speedup

## Architecture

```
                    ┌─────────┐
                    │  Token  │
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │ Embed   │
                    └────┬────┘
                         │
              ┌──────────▼──────────┐
              │   Layers 1-4        │──→ Exit Head (L4) ──→ H < τ₄? → EXIT
              └──────────┬──────────┘
                         │ (continue if H ≥ τ₄)
              ┌──────────▼──────────┐
              │   Layers 5-8        │──→ Exit Head (L8) ──→ H < τ₈? → EXIT
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   Layers 9-12       │──→ Exit Head (L12) ──→ H < τ₁₂? → EXIT
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   Layers 13-16      │──→ Final Head (L16) ──→ ALWAYS EXIT
              └─────────────────────┘
```

## Algorithm Overview

1. **Draft Phase**: Generate K speculative tokens using early exit. Each token propagates through layers until entropy drops below a threshold.

2. **Verify Phase**: Run all K draft tokens through the full model in one pass. Accept/reject each using rejection sampling: `accept with P = min(1, p_full / p_draft)`.

3. **Adapt Phase**: If acceptance rate is too low, tighten thresholds (push tokens deeper). If too high, loosen them (allow earlier exit).

## Files

- `model.py` — Pure NumPy transformer with KV-cache, exit heads, and simulated distillation
- `wadi.py` — WADI engine + standard AR and speculative decoding baselines
- `benchmark.py` — Comprehensive comparison with latency analysis

## Running

```bash
python3 benchmark.py
```

No dependencies beyond NumPy and SciPy.

## Why This Matters

| Property | Standard AR | Spec. Decoding | WADI |
|---|---|---|---|
| Draft model needed | No | Yes (separate) | **No (self-draft)** |
| Memory overhead | Baseline | 2x (two models) | **~1.01x (exit heads)** |
| Lossless | Yes | Yes | **Yes** |
| Per-token adaptivity | No | No | **Yes (entropy routing)** |
| Scales with depth | N/A | Moderate | **Strong** |

## Limitations of this PoC

- Uses random weights (not a trained model), so acceptance rates are lower than a real deployment
- No GPU pipelining (the wavefront overlap is theoretical in this implementation)
- Exit head distillation is simulated via least-squares fit on calibration data
- Sequential NumPy execution doesn't demonstrate the parallelism advantages
