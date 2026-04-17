"""
wadi_hf.py — WADI on a real HuggingFace model, end-to-end.
===========================================================

Loads a pretrained causal-LM (default: Qwen2.5-1.5B — Apache 2.0, ungated),
attaches exit heads at L/4, L/2, 3L/4, distills them against the final head
on WikiText-2, calibrates entropy thresholds, and benchmarks standard AR
vs WADI with batched rejection-sampling verification.

Tested with:
  transformers >= 4.44
  torch        >= 2.2
  datasets     >= 2.18

Quick start on an A100:

    pip install -U transformers datasets accelerate
    python wadi_hf.py                                   # Qwen2.5-1.5B, 100 new tokens
    python wadi_hf.py --model meta-llama/Llama-3.2-1B   # other HF models
    python wadi_hf.py --heads-ckpt heads.pt             # cache distilled heads

The draft loop uses forward hooks on the exit layers — an `EarlyExit`
exception short-circuits the HF forward pass at the shallowest layer whose
entropy clears its threshold, so deeper layers are genuinely skipped
(real FLOP savings, not just reported ones).
"""

from __future__ import annotations

import argparse
import copy
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


# ---------------------------------------------------------------------------
# Exit head
# ---------------------------------------------------------------------------


class _RMSNormFallback(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(v + self.eps)).to(x.dtype) * self.weight


def _rms_norm(d: int, eps: float = 1e-5) -> nn.Module:
    return getattr(nn, "RMSNorm", _RMSNormFallback)(d, eps=eps)


class ExitHead(nn.Module):
    """RMSNorm + Linear to vocab — same shape as a typical LM head."""

    def __init__(self, d: int, vocab_size: int, rms_eps: float = 1e-5):
        super().__init__()
        self.norm = _rms_norm(d, rms_eps)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(h))


def attach_exit_heads(
    model, exit_points: List[int], device, dtype
) -> nn.ModuleDict:
    cfg = model.config
    d, V = cfg.hidden_size, cfg.vocab_size
    rms_eps = getattr(cfg, "rms_norm_eps", 1e-5)

    heads = nn.ModuleDict()
    for ell in exit_points:
        h = ExitHead(d, V, rms_eps=rms_eps).to(device=device, dtype=dtype)
        # Warm-start from lm_head + final norm — trains ~3-5x faster.
        with torch.no_grad():
            h.lm_head.weight.copy_(model.lm_head.weight)
            if hasattr(model.model, "norm") and hasattr(model.model.norm, "weight"):
                h.norm.weight.copy_(model.model.norm.weight)
        heads[str(ell)] = h
    return heads


# ---------------------------------------------------------------------------
# Distillation
# ---------------------------------------------------------------------------


def distill(
    model,
    heads: nn.ModuleDict,
    tokenizer,
    num_tokens: int = 200_000,
    seq_len: int = 1024,
    lr: float = 1e-4,
    steps: int | None = None,
    log_every: int = 5,
) -> None:
    """Train exit heads to match the final-layer distribution via forward KL.

    Freezes the backbone. Uses WikiText-2 by default (small, fast download).
    """
    device = next(model.parameters()).device

    print("  Loading WikiText-2 calibration data...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(s for s in ds["text"] if len(s.strip()) > 80)
    ids = tokenizer(text, return_tensors="pt").input_ids[0][:num_tokens].to(device)

    N = ids.numel() // seq_len
    if N == 0:
        raise RuntimeError(
            f"Not enough tokens ({ids.numel()}) for seq_len={seq_len}."
        )
    ids = ids[: N * seq_len].reshape(N, seq_len)

    if steps is None:
        steps = N

    head_params = [p for h in heads.values() for p in h.parameters()]
    opt = torch.optim.AdamW(head_params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)

    model.eval()
    heads.train()

    print(f"  Distilling {len(heads)} heads on {N * seq_len:,} tokens ({steps} steps)...")
    losses = {ell: 0.0 for ell in heads}
    t0 = time.perf_counter()
    for step in range(steps):
        batch = ids[step % N].unsqueeze(0)

        with torch.no_grad():
            out = model(
                input_ids=batch, output_hidden_states=True, use_cache=False
            )
            teacher_lp = F.log_softmax(out.logits.float(), dim=-1)

        loss_total = 0.0
        step_losses: Dict[str, float] = {}
        for e_str, head in heads.items():
            ell = int(e_str)
            # hidden_states is [embedding, layer_1_out, ..., layer_N_out]; so index `ell`
            # gives the output of the `ell`-th decoder layer.
            h = out.hidden_states[ell]
            student_lp = F.log_softmax(head(h).float(), dim=-1)
            l = F.kl_div(student_lp, teacher_lp, reduction="batchmean", log_target=True)
            loss_total = loss_total + l
            step_losses[e_str] = float(l.item())

        opt.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(head_params, 1.0)
        opt.step()

        for e, v in step_losses.items():
            losses[e] = 0.9 * losses[e] + 0.1 * v if step > 0 else v

        if (step + 1) % log_every == 0 or step == steps - 1:
            elapsed = time.perf_counter() - t0
            per_step = elapsed / (step + 1)
            loss_str = "  ".join(f"L{e}={v:.3f}" for e, v in losses.items())
            print(f"  step {step+1:>4}/{steps}   {loss_str}   ({per_step*1000:.0f} ms/step)")

    heads.eval()


# ---------------------------------------------------------------------------
# Entropy calibration
# ---------------------------------------------------------------------------


@torch.no_grad()
def calibrate_thresholds(
    model,
    heads: nn.ModuleDict,
    tokenizer,
    prompt_text: str,
    percentile: float = 50.0,
    extra_tokens: int = 64,
) -> Dict[int, float]:
    """Set each layer's entropy threshold to a percentile of observed entropies.

    Runs one autoregressive-style pass: full forward on prompt, then a short
    greedy completion, gathering entropies at each exit head.
    """
    device = next(model.parameters()).device
    L = model.config.num_hidden_layers

    ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    out = model(ids, output_hidden_states=True, use_cache=False)
    ent_by_layer: Dict[int, List[float]] = {int(e): [] for e in heads}

    for e_str, head in heads.items():
        ell = int(e_str)
        h = out.hidden_states[ell]
        probs = F.softmax(head(h).float(), dim=-1)
        ent = -(probs * probs.clamp_min(1e-10).log()).sum(-1)
        ent_by_layer[ell].extend(ent.flatten().cpu().tolist())

    # A few greedy steps to capture generation-time entropy too.
    cache = DynamicCache()
    _ = model(ids[:, :-1], past_key_values=cache, use_cache=True)
    cur = ids[:, -1:]
    for _ in range(extra_tokens):
        o = model(cur, past_key_values=cache, use_cache=True, output_hidden_states=True)
        for e_str, head in heads.items():
            ell = int(e_str)
            h = o.hidden_states[ell][:, -1]
            probs = F.softmax(head(h).float(), dim=-1)
            ent = -(probs * probs.clamp_min(1e-10).log()).sum().item()
            ent_by_layer[ell].append(ent)
        cur = o.logits[:, -1:].argmax(-1)

    # Use per-layer percentiles so each exit gets a threshold tuned to
    # *its own* entropy distribution. Shallow heads are undertrained and
    # have much higher entropy than deep ones; coupling them (e.g. via a
    # monotonicity constraint) either starves shallow exits or drags deep
    # ones down. Target lower percentiles for shallow layers: only exit
    # when the shallow head is *very* confident.
    exits_sorted = sorted(ell for ell in ent_by_layer if ell != L)
    th: Dict[int, float] = {}
    n_exits = max(1, len(exits_sorted))
    for i, ell in enumerate(exits_sorted):
        # Interpolate per-layer target percentile: shallowest uses
        # percentile/3, deepest (non-final) uses percentile. That keeps
        # shallow exits selective and deep ones lenient.
        frac = (i + 1) / n_exits  # 1/N..1
        layer_pct = percentile * (1.0 / 3 + (2.0 / 3) * frac)
        vals = sorted(ent_by_layer[ell])
        idx = min(int(len(vals) * layer_pct / 100), len(vals) - 1)
        th[ell] = float(vals[idx])
    th[L] = float("inf")  # final layer always exits
    return th


# ---------------------------------------------------------------------------
# Draft / Verify
# ---------------------------------------------------------------------------


@dataclass
class Draft:
    token_id: int
    exit_layer: int
    probs: torch.Tensor  # (vocab,) — float32


class _EarlyExit(Exception):
    __slots__ = ("probs", "exit_layer")

    def __init__(self, probs: torch.Tensor, exit_layer: int):
        super().__init__()
        self.probs = probs
        self.exit_layer = exit_layer


def _make_exit_hook(ell: int, head: ExitHead, threshold: float):
    def hook(module, _inputs, outputs):
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
        logits = head(hidden[:, -1]).float()
        probs = F.softmax(logits, dim=-1).squeeze(0)
        ent = -(probs * probs.clamp_min(1e-10).log()).sum().item()
        if ent < threshold:
            raise _EarlyExit(probs, ell)
    return hook


@torch.no_grad()
def draft_one(
    model,
    heads: nn.ModuleDict,
    thresholds: Dict[int, float],
    input_ids: torch.Tensor,
    cache: DynamicCache,
) -> Draft:
    """One draft token via early-exit.

    A forward hook on each exit layer raises ``_EarlyExit`` the moment an
    exit-head's entropy clears its threshold, so deeper layers aren't run.
    """
    layers = model.model.layers
    handles = []
    for e_str, head in heads.items():
        ell = int(e_str)
        handles.append(
            layers[ell - 1].register_forward_hook(
                _make_exit_hook(ell, head, thresholds[ell])
            )
        )
    try:
        try:
            out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        except _EarlyExit as ee:
            tok = int(torch.multinomial(ee.probs, 1).item())
            return Draft(tok, ee.exit_layer, ee.probs)
        probs = F.softmax(out.logits[:, -1].float(), dim=-1).squeeze(0)
        tok = int(torch.multinomial(probs, 1).item())
        return Draft(tok, len(layers), probs)
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def verify_and_accept(
    model,
    drafts: List[Draft],
    context_token: int,
    cache: DynamicCache,
) -> Tuple[List[int], int]:
    """Single batched forward over [context, d_0, ..., d_{K-2}], then
    rejection-sample each draft. Returns (accepted_tokens, n_drafts_kept).

    ``cache`` is cropped to match len(accepted) after the call — so it stays
    in sync with ``generated[:-0]`` for the next round.
    """
    device = next(model.parameters()).device
    K = len(drafts)
    ids_list = [context_token] + [d.token_id for d in drafts[:-1]]
    input_ids = torch.tensor([ids_list], device=device)

    cache_before = cache.get_seq_length()
    out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    full_probs = F.softmax(out.logits.float(), dim=-1)[0]  # (K, vocab)

    accepted: List[int] = []
    n_drafts_kept = 0
    per_exit_acc: Dict[int, int] = {}
    per_exit_rej: Dict[int, int] = {}
    for i, d in enumerate(drafts):
        p_full = float(full_probs[i, d.token_id].item())
        p_draft = float(d.probs[d.token_id].item())
        r = min(1.0, p_full / (p_draft + 1e-10))
        if torch.rand(()).item() < r:
            accepted.append(d.token_id)
            n_drafts_kept += 1
            per_exit_acc[d.exit_layer] = per_exit_acc.get(d.exit_layer, 0) + 1
        else:
            per_exit_rej[d.exit_layer] = per_exit_rej.get(d.exit_layer, 0) + 1
            residual = torch.clamp(full_probs[i] - d.probs.to(full_probs.device), min=0)
            z = residual.sum()
            dist = residual / z if z > 1e-10 else full_probs[i]
            accepted.append(int(torch.multinomial(dist, 1).item()))
            break

    cache.crop(cache_before + len(accepted))
    return accepted, n_drafts_kept, per_exit_acc, per_exit_rej


# ---------------------------------------------------------------------------
# Top-level generate
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_ar(model, tokenizer, prompt_text: str, max_new_tokens: int) -> List[int]:
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

    cache = DynamicCache()
    if input_ids.shape[1] > 1:
        _ = model(input_ids=input_ids[:, :-1], past_key_values=cache, use_cache=True)

    current = input_ids[:, -1:]
    generated: List[int] = input_ids[0].tolist()
    for _ in range(max_new_tokens):
        out = model(input_ids=current, past_key_values=cache, use_cache=True)
        probs = F.softmax(out.logits[:, -1].float(), dim=-1).squeeze(0)
        tok = int(torch.multinomial(probs, 1).item())
        generated.append(tok)
        current = torch.tensor([[tok]], device=device)
    return generated


@torch.no_grad()
def generate_wadi(
    model,
    tokenizer,
    heads: nn.ModuleDict,
    thresholds: Dict[int, float],
    prompt_text: str,
    max_new_tokens: int,
    max_draft_len: int = 6,
):
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

    cache = DynamicCache()
    if input_ids.shape[1] > 1:
        _ = model(input_ids=input_ids[:, :-1], past_key_values=cache, use_cache=True)

    current_token = int(input_ids[0, -1].item())
    generated: List[int] = input_ids[0].tolist()

    stats = {
        "drafts": 0, "drafts_accepted": 0, "rounds": 0, "rejections": 0,
        "exit_hist": {}, "exit_accepted": {}, "exit_rejected": {},
    }

    tokens_left = max_new_tokens
    while tokens_left > 0:
        draft_cache = copy.deepcopy(cache)
        drafts: List[Draft] = []
        draft_tok = current_token
        K = min(max_draft_len, tokens_left)
        for _ in range(K):
            dtok_ids = torch.tensor([[draft_tok]], device=device)
            d = draft_one(model, heads, thresholds, dtok_ids, draft_cache)
            drafts.append(d)
            stats["drafts"] += 1
            stats["exit_hist"][d.exit_layer] = stats["exit_hist"].get(d.exit_layer, 0) + 1
            draft_tok = d.token_id

        accepted, n_kept, per_exit_acc, per_exit_rej = verify_and_accept(
            model, drafts, current_token, cache
        )
        generated.extend(accepted)
        stats["rounds"] += 1
        stats["drafts_accepted"] += n_kept
        if n_kept < len(drafts):
            stats["rejections"] += 1
        for ell, c in per_exit_acc.items():
            stats["exit_accepted"][ell] = stats["exit_accepted"].get(ell, 0) + c
        for ell, c in per_exit_rej.items():
            stats["exit_rejected"][ell] = stats["exit_rejected"].get(ell, 0) + c
        tokens_left -= len(accepted)
        current_token = accepted[-1]

    return generated, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument(
        "--prompt",
        default="The quick brown fox jumps over the lazy dog. The result of the experiment was",
    )
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--max-draft-len", type=int, default=6)

    p.add_argument("--calibration-tokens", type=int, default=200_000)
    p.add_argument("--distill-seq-len", type=int, default=1024)
    p.add_argument("--distill-steps", type=int, default=None,
                   help="Default: one pass over the calibration data.")
    p.add_argument("--distill-lr", type=float, default=1e-4)
    p.add_argument("--skip-distill", action="store_true",
                   help="Use lm_head-initialized heads without any distillation.")
    p.add_argument("--heads-ckpt", default=None,
                   help="Path to save/load distilled exit heads.")

    p.add_argument("--threshold-percentile", type=float, default=50.0)
    p.add_argument("--exit-layers", default=None,
                   help="Comma-separated exit layer indices (1-based into the "
                        "transformer blocks). Default: L/4, L/2, 3L/4. "
                        "Example: --exit-layers 14,21 drops the shallowest exit, "
                        "which often helps on small/undertrained heads.")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    print(f"Loading {args.model} on {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model = model.to(device).eval()

    for param in model.parameters():
        param.requires_grad = False

    L = model.config.num_hidden_layers
    if args.exit_layers:
        exit_points = sorted({int(x) for x in args.exit_layers.split(",") if x.strip()})
        for e in exit_points:
            if not (1 <= e < L):
                raise ValueError(f"Exit layer {e} out of range [1, {L - 1}] for this model")
    else:
        exit_points = [max(1, L // 4), L // 2, (3 * L) // 4]
    print(f"Model: {L} layers, {model.config.hidden_size} hidden.")
    print(f"Exit points: {exit_points}")

    heads = attach_exit_heads(model, exit_points, device, dtype)

    if args.heads_ckpt and os.path.exists(args.heads_ckpt):
        print(f"Loading distilled heads from {args.heads_ckpt}")
        heads.load_state_dict(torch.load(args.heads_ckpt, map_location=device))
    elif not args.skip_distill:
        print("\n--- Distillation ---")
        distill(
            model, heads, tokenizer,
            num_tokens=args.calibration_tokens,
            seq_len=args.distill_seq_len,
            lr=args.distill_lr,
            steps=args.distill_steps,
        )
        if args.heads_ckpt:
            torch.save(heads.state_dict(), args.heads_ckpt)
            print(f"Saved distilled heads to {args.heads_ckpt}")

    print("\n--- Calibrating entropy thresholds ---")
    thresholds = calibrate_thresholds(
        model, heads, tokenizer, args.prompt,
        percentile=args.threshold_percentile,
    )
    print("  " + ", ".join(
        f"L{ell}={v:.2f}" if v != float("inf") else f"L{ell}=inf"
        for ell, v in sorted(thresholds.items())
    ))

    # Warmup CUDA kernels — first forward is always slower.
    print("\n--- Warmup ---")
    _ = generate_ar(model, tokenizer, args.prompt, 8)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print("\n--- Standard AR ---")
    t0 = time.perf_counter()
    ar_tokens = generate_ar(model, tokenizer, args.prompt, args.max_new_tokens)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ar_time = time.perf_counter() - t0
    print(f"  {ar_time:.2f}s   {args.max_new_tokens / ar_time:.1f} tok/s")

    print("\n--- WADI ---")
    t0 = time.perf_counter()
    wadi_tokens, stats = generate_wadi(
        model, tokenizer, heads, thresholds, args.prompt,
        args.max_new_tokens, max_draft_len=args.max_draft_len,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wadi_time = time.perf_counter() - t0
    print(f"  {wadi_time:.2f}s   {args.max_new_tokens / wadi_time:.1f} tok/s")
    print(f"  Speedup:           {ar_time / wadi_time:.2f}x")
    print(f"  Acceptance rate:   "
          f"{stats['drafts_accepted'] / max(1, stats['drafts']):.1%}")
    print(f"  Avg tok / round:   "
          f"{(args.max_new_tokens) / max(1, stats['rounds']):.1f}")
    print(f"  Exit distribution: "
          f"{dict(sorted(stats['exit_hist'].items()))}")
    print(f"  Per-exit acceptance:")
    for ell in sorted(stats["exit_hist"]):
        acc = stats["exit_accepted"].get(ell, 0)
        rej = stats["exit_rejected"].get(ell, 0)
        tot = acc + rej
        rate = (acc / tot) if tot else 0.0
        print(f"    L{ell}: {acc}/{tot} accepted ({rate:.1%})")

    print("\n--- Output ---")
    print(tokenizer.decode(wadi_tokens, skip_special_tokens=True))


if __name__ == "__main__":
    main()
