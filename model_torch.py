"""
PyTorch backend for WADI.

Mirrors the pure-NumPy reference implementation in ``model.py`` but uses
``torch.nn`` modules so the model can run on GPU, be fine-tuned, and use
``torch.nn.functional.scaled_dot_product_attention`` for fused / flash
attention on modern hardware.

The public surface is intentionally parallel to the NumPy version:

* ``TorchTransformerConfig`` - same fields as ``TransformerConfig``
* ``TorchTransformer.full_forward_single(token_id, kv_cache, pos)``
* ``TorchTransformer.full_forward_batch(token_ids, kv_cache, start_pos)``
* ``TorchTransformer.forward_with_exits(token_id, kv_cache, pos)``
* ``TorchKVCache`` with ``update``/``get``/``truncate``/``clone``

This lets ``wadi_torch.WADIEngine`` reuse the same algorithm as the NumPy
version with a drop-in model swap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TorchTransformerConfig:
    vocab_size: int = 512
    n_layers: int = 16
    n_heads: int = 4
    d_model: int = 128
    d_ff: int = 512
    max_seq_len: int = 256
    exit_layers: list = field(default_factory=lambda: [4, 8, 12, 16])
    init_scale: float = 0.02
    # The backend picks a reasonable dtype/device default; override as needed.
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        self.d_head = self.d_model // self.n_heads


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------


class TorchKVCache:
    """Per-layer K/V buffer with preallocation and per-layer truncation."""

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        d_head: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        shape = (n_layers, n_heads, max_seq_len, d_head)
        self.keys = torch.empty(shape, device=device, dtype=dtype)
        self.values = torch.empty(shape, device=device, dtype=dtype)
        self.lengths = torch.zeros(n_layers, dtype=torch.int64, device=device)

    def update(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor) -> None:
        cur_len = int(self.lengths[layer_idx].item())
        n_new = new_k.shape[1]
        end = cur_len + n_new
        if end > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow at layer {layer_idx}: {end} > {self.max_seq_len}"
            )
        self.keys[layer_idx, :, cur_len:end, :] = new_k
        self.values[layer_idx, :, cur_len:end, :] = new_v
        self.lengths[layer_idx] = end

    def get(self, layer_idx: int):
        cur_len = int(self.lengths[layer_idx].item())
        if cur_len == 0:
            return None, None
        return (
            self.keys[layer_idx, :, :cur_len, :],
            self.values[layer_idx, :, :cur_len, :],
        )

    def get_seq_len(self, layer_idx: int = 0) -> int:
        return int(self.lengths[layer_idx].item())

    def truncate(self, layer_idx: int, length: int) -> None:
        if length < int(self.lengths[layer_idx].item()):
            self.lengths[layer_idx] = length

    def truncate_all(self, length: int) -> None:
        self.lengths.clamp_(max=length)

    def clone(self) -> "TorchKVCache":
        new = TorchKVCache(
            self.n_layers,
            self.n_heads,
            self.d_head,
            self.max_seq_len,
            self.device,
            self.dtype,
        )
        new.keys.copy_(self.keys)
        new.values.copy_(self.values)
        new.lengths.copy_(self.lengths)
        return new


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


class TorchTransformerLayer(nn.Module):
    def __init__(self, config: TorchTransformerConfig, layer_idx: int):
        super().__init__()
        d = config.d_model
        ff = config.d_ff

        # Fused QKV projection — one GEMM for all three.
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

        self.ff1 = nn.Linear(d, ff, bias=True)
        self.ff2 = nn.Linear(ff, d, bias=True)

        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)

        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.layer_idx = layer_idx

        # Match the NumPy reference's small-initialization scheme so numeric
        # behavior is comparable when weights are transferred.
        with torch.no_grad():
            for w in (self.qkv.weight, self.proj.weight, self.ff1.weight, self.ff2.weight):
                w.normal_(mean=0.0, std=config.init_scale)
            self.ff1.bias.zero_()
            self.ff2.bias.zero_()

    def _attention(self, x: torch.Tensor, kv_cache: Optional[TorchKVCache]) -> torch.Tensor:
        # x: (S, D)
        S, D = x.shape
        h, dh = self.n_heads, self.d_head

        qkv = self.qkv(x)  # (S, 3D)
        q, k, v = qkv.view(S, 3, h, dh).unbind(dim=1)  # each (S, h, dh)
        # Reshape to (h, S, dh) for attention.
        q = q.transpose(0, 1).contiguous()
        k = k.transpose(0, 1).contiguous()
        v = v.transpose(0, 1).contiguous()

        if kv_cache is not None:
            kv_cache.update(self.layer_idx, k, v)
            k, v = kv_cache.get(self.layer_idx)

        # SDPA expects (..., S, D); here "..." is the head axis.
        # For prefill/batch verify (S > 1) use is_causal so each query only
        # attends to keys up to its own kv-position. For single-token decode
        # the query attends to all cached keys (no mask needed).
        is_causal = S > 1 and k.shape[1] == S
        attn = F.scaled_dot_product_attention(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            is_causal=is_causal,
        ).squeeze(0)
        # (h, S, dh) -> (S, h*dh)
        attn = attn.transpose(0, 1).contiguous().view(S, h * dh)
        return self.proj(attn)

    def forward(self, x: torch.Tensor, kv_cache: Optional[TorchKVCache] = None) -> torch.Tensor:
        x = x + self._attention(self.ln1(x), kv_cache)
        x = x + self.ff2(F.gelu(self.ff1(self.ln2(x))))
        return x


class TorchExitHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, init_scale: float = 0.02):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=True)
        with torch.no_grad():
            self.head.weight.normal_(mean=0.0, std=init_scale)
            self.head.bias.zero_()

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.ln(x))


class TorchTransformer(nn.Module):
    """Transformer with exit heads. Runs on CPU or GPU via standard torch."""

    def __init__(self, config: TorchTransformerConfig, seed: int = 42):
        super().__init__()
        self.config = config

        if seed is not None:
            torch.manual_seed(seed)

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        with torch.no_grad():
            self.tok_emb.weight.normal_(0, 0.02)
            self.pos_emb.weight.normal_(0, 0.02)

        self.layers = nn.ModuleList(
            [TorchTransformerLayer(config, i) for i in range(config.n_layers)]
        )
        self.final_ln = nn.LayerNorm(config.d_model)

        # nn.ModuleDict keys must be strings — canonicalize on lookup helpers.
        self.exit_heads = nn.ModuleDict(
            {str(ell): TorchExitHead(config.d_model, config.vocab_size) for ell in config.exit_layers}
        )

        self._flop_counter = 0
        self.to(dtype=config.dtype)

    # ---- KV cache ---------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self.tok_emb.weight.device

    def create_kv_cache(self) -> TorchKVCache:
        return TorchKVCache(
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
            d_head=self.config.d_head,
            max_seq_len=self.config.max_seq_len,
            device=self.device,
            dtype=self.tok_emb.weight.dtype,
        )

    # ---- FLOP counting (matches NumPy accounting) -------------------------

    def reset_flops(self) -> None:
        self._flop_counter = 0

    def get_flops(self) -> int:
        return self._flop_counter

    def _charge_flops(self, seq_len: int, start_layer: int, end_layer: int) -> None:
        d = self.config.d_model
        ff = self.config.d_ff
        n = end_layer - start_layer
        self._flop_counter += int(seq_len * n * (4 * d * d + 2 * d * ff))

    # ---- Forward paths ----------------------------------------------------

    def _embed(self, token_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        seq_len = token_ids.shape[0]
        positions = torch.arange(start_pos, start_pos + seq_len, device=self.device)
        return self.tok_emb(token_ids) + self.pos_emb(positions)

    def _forward_layers(
        self,
        x: torch.Tensor,
        start_layer: int,
        end_layer: int,
        kv_cache: Optional[TorchKVCache],
    ) -> torch.Tensor:
        for i in range(start_layer, end_layer):
            x = self.layers[i](x, kv_cache)
        self._charge_flops(x.shape[0], start_layer, end_layer)
        return x

    @torch.no_grad()
    def full_forward_single(
        self, token_id: int, kv_cache: TorchKVCache, pos: int
    ):
        ids = torch.as_tensor([int(token_id)], dtype=torch.long, device=self.device)
        x = self._embed(ids, pos)
        x = self._forward_layers(x, 0, self.config.n_layers, kv_cache)
        x = self.final_ln(x)
        logits = self.exit_heads[str(self.config.n_layers)].head(x[-1:])
        probs = F.softmax(logits, dim=-1).squeeze(0)
        entropy = -(probs * torch.log(probs.clamp_min(1e-10))).sum()
        return probs, float(entropy.item())

    @torch.no_grad()
    def full_forward_batch(
        self,
        token_ids: Sequence[int],
        kv_cache: TorchKVCache,
        start_pos: int,
    ):
        """Run K tokens through the full model in a single forward pass.

        Returns (probs, entropies) of shape (K, vocab) and (K,).
        """
        ids = torch.as_tensor(list(token_ids), dtype=torch.long, device=self.device)
        x = self._embed(ids, start_pos)
        x = self._forward_layers(x, 0, self.config.n_layers, kv_cache)
        x = self.final_ln(x)
        logits = self.exit_heads[str(self.config.n_layers)].head(x)
        probs = F.softmax(logits, dim=-1)
        entropies = -(probs * torch.log(probs.clamp_min(1e-10))).sum(dim=-1)
        return probs, entropies

    @torch.no_grad()
    def forward_with_exits(
        self, token_id: int, kv_cache: TorchKVCache, pos: int
    ) -> List[Dict]:
        """Yield (hidden, probs, entropy, exit_layer) at each exit point."""
        ids = torch.as_tensor([int(token_id)], dtype=torch.long, device=self.device)
        x = self._embed(ids, pos)
        prev = 0
        results = []
        for ell in self.config.exit_layers:
            x = self._forward_layers(x, prev, ell, kv_cache)
            head = self.exit_heads[str(ell)]
            logits = head.logits(x[-1:])
            probs = F.softmax(logits, dim=-1).squeeze(0)
            entropy = -(probs * torch.log(probs.clamp_min(1e-10))).sum()
            results.append({
                "hidden": x.clone(),
                "probs": probs,
                "entropy": float(entropy.item()),
                "exit_layer": ell,
            })
            prev = ell
        return results

    @torch.no_grad()
    def forward_to_layer(
        self,
        token_id: int,
        kv_cache: TorchKVCache,
        pos: int,
        target_layer: int,
    ):
        """Forward pass up to a given exit layer. Returns (hidden, probs, entropy)."""
        ids = torch.as_tensor([int(token_id)], dtype=torch.long, device=self.device)
        x = self._embed(ids, pos)
        prev = 0
        for ell in sorted(self.config.exit_layers):
            if ell > target_layer:
                break
            x = self._forward_layers(x, prev, ell, kv_cache)
            prev = ell
        head = self.exit_heads[str(target_layer)]
        logits = head.logits(x[-1:])
        probs = F.softmax(logits, dim=-1).squeeze(0)
        entropy = -(probs * torch.log(probs.clamp_min(1e-10))).sum()
        return x, probs, float(entropy.item())

    # ---- Weight transfer from the NumPy reference -------------------------

    @torch.no_grad()
    def load_from_numpy(self, numpy_model) -> None:
        """Copy weights from a NumPy ``Transformer`` into this module.

        Useful for apples-to-apples comparison between the two backends,
        or for using NumPy-distilled exit heads from a Torch inference path.
        Layer counts, head dims, etc. must match.
        """
        import numpy as np

        np_cfg = numpy_model.config
        cfg = self.config
        assert np_cfg.n_layers == cfg.n_layers
        assert np_cfg.d_model == cfg.d_model
        assert np_cfg.n_heads == cfg.n_heads
        assert np_cfg.vocab_size == cfg.vocab_size

        self.tok_emb.weight.copy_(torch.from_numpy(numpy_model.tok_emb))
        self.pos_emb.weight.copy_(torch.from_numpy(numpy_model.pos_emb))

        self.final_ln.weight.copy_(torch.from_numpy(numpy_model.final_ln_g))
        self.final_ln.bias.copy_(torch.from_numpy(numpy_model.final_ln_b))

        for i, (t_layer, np_layer) in enumerate(zip(self.layers, numpy_model.layers)):
            d = cfg.d_model
            # NumPy stores W_q/W_k/W_v as (d, h, dh). Flatten to (d, d) and
            # stack to (3d, d) in row-major to match nn.Linear convention
            # (Linear.weight is (out, in), so W @ x^T needs transpose).
            W_q = np_layer.W_q.reshape(d, d)
            W_k = np_layer.W_k.reshape(d, d)
            W_v = np_layer.W_v.reshape(d, d)
            W_qkv = np.concatenate([W_q, W_k, W_v], axis=1).T  # (3d, d)
            t_layer.qkv.weight.copy_(torch.from_numpy(W_qkv.astype(np.float32)))

            # NumPy W_o is (h, dh, d); flatten to (h*dh, d) then transpose
            # to (d, h*dh) for Linear.weight storage.
            W_o = np_layer.W_o.reshape(-1, d).T
            t_layer.proj.weight.copy_(torch.from_numpy(W_o.astype(np.float32)))

            t_layer.ff1.weight.copy_(torch.from_numpy(np_layer.W1.T.astype(np.float32)))
            t_layer.ff1.bias.copy_(torch.from_numpy(np_layer.b1))
            t_layer.ff2.weight.copy_(torch.from_numpy(np_layer.W2.T.astype(np.float32)))
            t_layer.ff2.bias.copy_(torch.from_numpy(np_layer.b2))

            t_layer.ln1.weight.copy_(torch.from_numpy(np_layer.ln1_g))
            t_layer.ln1.bias.copy_(torch.from_numpy(np_layer.ln1_b))
            t_layer.ln2.weight.copy_(torch.from_numpy(np_layer.ln2_g))
            t_layer.ln2.bias.copy_(torch.from_numpy(np_layer.ln2_b))

        for ell, head in numpy_model.exit_heads.items():
            t_head = self.exit_heads[str(ell)]
            t_head.ln.weight.copy_(torch.from_numpy(head.ln_g))
            t_head.ln.bias.copy_(torch.from_numpy(head.ln_b))
            t_head.head.weight.copy_(torch.from_numpy(head.W.T.astype("float32")))
            t_head.head.bias.copy_(torch.from_numpy(head.b))
