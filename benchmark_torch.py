"""
WADI Torch-backend benchmark.

Runs the same three strategies (Standard AR, Speculative Decoding, WADI) as
``benchmark.py`` but using ``model_torch`` / ``wadi_torch``. Automatically
uses CUDA if available.

Exit heads are distilled on the NumPy reference implementation (fast, and
keeps the two backends comparable) and then copied over to the Torch model
via ``TorchTransformer.load_from_numpy``.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch

from model import Transformer, TransformerConfig
from model_torch import TorchTransformer, TorchTransformerConfig
from wadi_torch import WADIEngineTorch, WADITorchConfig


def header(t: str) -> None:
    print(f"\n{'=' * 64}\n  {t}\n{'=' * 64}")


def section(t: str) -> None:
    print(f"\n--- {t} ---")


def _standard_ar(model: TorchTransformer, prompt, max_new_tokens: int):
    """Baseline: plain autoregressive decoding on the Torch model."""
    model.reset_flops()
    gen = torch.Generator(device=model.device).manual_seed(456)
    kv = model.create_kv_cache()
    for i, tok in enumerate(prompt[:-1]):
        model.full_forward_single(tok, kv, i)

    current = int(prompt[-1])
    pos = len(prompt) - 1
    out = list(prompt)
    for _ in range(max_new_tokens):
        probs, _ = model.full_forward_single(current, kv, pos)
        idx = int(torch.multinomial(probs, num_samples=1, generator=gen).item())
        out.append(idx)
        current = idx
        pos += 1
    return out, model.get_flops()


def run() -> None:
    header("WADI (Torch backend): performance sanity run")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Torch:  {torch.__version__}")

    cfg_np = TransformerConfig(
        vocab_size=512, n_layers=16, n_heads=4,
        d_model=128, d_ff=512, max_seq_len=256,
        exit_layers=[4, 8, 12, 16], init_scale=0.08,
    )
    cfg_t = TorchTransformerConfig(
        vocab_size=cfg_np.vocab_size,
        n_layers=cfg_np.n_layers,
        n_heads=cfg_np.n_heads,
        d_model=cfg_np.d_model,
        d_ff=cfg_np.d_ff,
        max_seq_len=cfg_np.max_seq_len,
        exit_layers=list(cfg_np.exit_layers),
        init_scale=cfg_np.init_scale,
    )

    section("1. Distill exits on NumPy backend, transfer to Torch")
    np_model = Transformer(cfg_np, seed=42)
    np_model.simulate_trained_exits(n_calibration=200, seed=1042)

    t_model = TorchTransformer(cfg_t, seed=42).to(device)
    t_model.load_from_numpy(np_model)
    print("  Weights transferred:", sum(p.numel() for p in t_model.parameters()), "params")

    # Auto-threshold from the NumPy entropy profile.
    from benchmark import measure_entropy

    prompt = np.random.default_rng(7).integers(0, cfg_np.vocab_size, size=16).tolist()
    gen_len = 60
    ent = measure_entropy(np_model, np.array(prompt))
    auto_th = {}
    L = cfg_np.n_layers
    for e in sorted(ent):
        auto_th[e] = float(np.percentile(np.array(ent[e]), 50)) if e != L else float("inf")
    print(f"  Auto thresholds: {auto_th}")

    section("2. Standard AR (Torch)")
    t0 = time.perf_counter()
    _, ar_flops = _standard_ar(t_model, prompt, gen_len)
    ar_wall = time.perf_counter() - t0
    print(f"  Wall time: {ar_wall * 1000:>8.1f} ms   FLOPs: {ar_flops:,}")

    section("3. WADI (Torch)")
    wcfg = WADITorchConfig(
        initial_thresholds=auto_th,
        max_draft_len=6,
        target_acceptance_rate=0.75,
        threshold_lr=0.03,
    )
    wadi = WADIEngineTorch(t_model, wcfg)
    t0 = time.perf_counter()
    wadi.generate(prompt, gen_len)
    wadi_wall = time.perf_counter() - t0
    stats = wadi.get_stats()
    print(f"  Wall time: {wadi_wall * 1000:>8.1f} ms")
    print(stats.summary())

    section("4. Comparison")
    speedup = ar_wall / max(wadi_wall, 1e-9)
    print(f"  WADI wall-clock speedup vs AR: {speedup:.2f}x")
    print(
        f"  WADI FLOPs vs AR: {(stats.flops_draft + stats.flops_verify) / ar_flops:.2f}x"
    )
    print(
        "  (FLOPs can be slightly higher than AR on untrained weights; "
        "latency win comes from batched verify + shallower drafts.)"
    )


if __name__ == "__main__":
    run()
