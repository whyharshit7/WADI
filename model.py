"""
Minimal GPT-like Transformer in pure NumPy.
Includes exit heads at configurable layer depths for WADI inference.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TransformerConfig:
    vocab_size: int = 512
    n_layers: int = 16
    n_heads: int = 4
    d_model: int = 128
    d_ff: int = 512
    max_seq_len: int = 256
    exit_layers: list = field(default_factory=lambda: [4, 8, 12, 16])
    init_scale: float = 0.02  # Weight init scale; smaller = more stable residual

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        self.d_head = self.d_model // self.n_heads


def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


class KVCache:
    """Key-Value cache for efficient autoregressive inference."""

    def __init__(self, n_layers, n_heads=0, d_head=0, max_seq_len=0, dtype=np.float32):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.keys = [None] * n_layers
        self.values = [None] * n_layers
        self.lengths = np.zeros(n_layers, dtype=np.int32)

    def _supports_preallocation(self):
        return self.n_heads > 0 and self.d_head > 0 and self.max_seq_len > 0

    def _ensure_layer_buffers(self, layer_idx):
        if self.keys[layer_idx] is not None or not self._supports_preallocation():
            return
        shape = (self.n_heads, self.max_seq_len, self.d_head)
        self.keys[layer_idx] = np.empty(shape, dtype=self.dtype)
        self.values[layer_idx] = np.empty(shape, dtype=self.dtype)

    def update(self, layer_idx, new_k, new_v):
        """Append new K,V to cache. new_k/new_v: (n_heads, n_new, d_head)"""
        cur_len = int(self.lengths[layer_idx])
        n_new = new_k.shape[1]

        if self.keys[layer_idx] is None:
            self._ensure_layer_buffers(layer_idx)

        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = new_k.copy()
            self.values[layer_idx] = new_v.copy()
            self.lengths[layer_idx] = n_new
            return

        if self._supports_preallocation() and self.keys[layer_idx].shape[1] == self.max_seq_len:
            end = cur_len + n_new
            if end > self.max_seq_len:
                raise ValueError(f"KV cache overflow at layer {layer_idx}: {end} > {self.max_seq_len}")
            self.keys[layer_idx][:, cur_len:end, :] = new_k
            self.values[layer_idx][:, cur_len:end, :] = new_v
            self.lengths[layer_idx] = end
            return

        active_k = self.keys[layer_idx][:, :cur_len, :]
        active_v = self.values[layer_idx][:, :cur_len, :]
        self.keys[layer_idx] = np.concatenate([active_k, new_k], axis=1)
        self.values[layer_idx] = np.concatenate([active_v, new_v], axis=1)
        self.lengths[layer_idx] = cur_len + n_new

    def get(self, layer_idx):
        if self.keys[layer_idx] is None:
            return None, None
        cur_len = int(self.lengths[layer_idx])
        return self.keys[layer_idx][:, :cur_len, :], self.values[layer_idx][:, :cur_len, :]

    def get_seq_len(self, layer_idx=0):
        return int(self.lengths[layer_idx])

    def truncate(self, layer_idx, length):
        """Truncate cache for a layer to a given length."""
        cur_len = int(self.lengths[layer_idx])
        if self.keys[layer_idx] is None or length >= cur_len:
            return

        self.lengths[layer_idx] = length
        if not (self._supports_preallocation() and self.keys[layer_idx].shape[1] == self.max_seq_len):
            if length == 0:
                self.keys[layer_idx] = None
                self.values[layer_idx] = None
            else:
                self.keys[layer_idx] = self.keys[layer_idx][:, :length, :]
                self.values[layer_idx] = self.values[layer_idx][:, :length, :]

    def truncate_all(self, length):
        """Truncate all layer caches to a given length."""
        for i in range(self.n_layers):
            self.truncate(i, length)

    def clone(self):
        """Deep copy."""
        new_cache = KVCache(
            self.n_layers,
            n_heads=self.n_heads,
            d_head=self.d_head,
            max_seq_len=self.max_seq_len,
            dtype=self.dtype,
        )
        for i in range(self.n_layers):
            cur_len = int(self.lengths[i])
            if cur_len == 0:
                continue

            if new_cache._supports_preallocation():
                new_cache._ensure_layer_buffers(i)
                new_cache.keys[i][:, :cur_len, :] = self.keys[i][:, :cur_len, :]
                new_cache.values[i][:, :cur_len, :] = self.values[i][:, :cur_len, :]
            else:
                new_cache.keys[i] = self.keys[i][:, :cur_len, :].copy()
                new_cache.values[i] = self.values[i][:, :cur_len, :].copy()
            new_cache.lengths[i] = cur_len
        return new_cache


class TransformerLayer:
    """Single transformer block: Pre-Norm MHA + Pre-Norm FFN.

    Uses a fused Q/K/V projection (single matmul) for speed. The original
    (W_q, W_k, W_v) weights are kept as separate views so downstream code
    that introspects them still works.
    """

    def __init__(self, config: TransformerConfig, layer_idx: int, rng: np.random.Generator):
        d = config.d_model
        h = config.n_heads
        dh = config.d_head
        ff = config.d_ff
        scale = config.init_scale

        self.W_q = rng.normal(0, scale, (d, h, dh)).astype(np.float32)
        self.W_k = rng.normal(0, scale, (d, h, dh)).astype(np.float32)
        self.W_v = rng.normal(0, scale, (d, h, dh)).astype(np.float32)
        self.W_o = rng.normal(0, scale, (h, dh, d)).astype(np.float32)

        self.W1 = rng.normal(0, scale, (d, ff)).astype(np.float32)
        self.b1 = np.zeros(ff, dtype=np.float32)
        self.W2 = rng.normal(0, scale, (ff, d)).astype(np.float32)
        self.b2 = np.zeros(d, dtype=np.float32)

        self.ln1_g = np.ones(d, dtype=np.float32)
        self.ln1_b = np.zeros(d, dtype=np.float32)
        self.ln2_g = np.ones(d, dtype=np.float32)
        self.ln2_b = np.zeros(d, dtype=np.float32)

        self.layer_idx = layer_idx
        self.n_heads = h
        self.d_head = dh
        self.d_model = d
        self.attn_scale = 1.0 / np.sqrt(dh)

        # Fused (d, 3 * h * dh) weight for a single Q/K/V matmul.
        # Flat 2D shape hits BLAS GEMM directly without einsum overhead.
        self._rebuild_fused_qkv()

        # Flat (h*dh, d) output projection for a single matmul.
        self._W_o_flat = self.W_o.reshape(h * dh, d)

    def _rebuild_fused_qkv(self):
        h, dh, d = self.n_heads, self.d_head, self.d_model
        self._W_qkv = np.concatenate(
            [
                self.W_q.reshape(d, h * dh),
                self.W_k.reshape(d, h * dh),
                self.W_v.reshape(d, h * dh),
            ],
            axis=1,
        ).astype(np.float32)

    def attention(self, x, kv_cache: Optional[KVCache] = None):
        seq_len, d = x.shape
        h, dh = self.n_heads, self.d_head

        # One fused GEMM produces Q, K, V concatenated.
        qkv = x @ self._W_qkv  # (s, 3*h*dh)
        qkv = qkv.reshape(seq_len, 3, h, dh)
        # (h, s, dh) for each — contiguous for the matmuls that follow.
        Q = np.ascontiguousarray(qkv[:, 0].transpose(1, 0, 2))
        K = np.ascontiguousarray(qkv[:, 1].transpose(1, 0, 2))
        V = np.ascontiguousarray(qkv[:, 2].transpose(1, 0, 2))

        if kv_cache is not None:
            kv_cache.update(self.layer_idx, K, V)
            K, V = kv_cache.get(self.layer_idx)

        # (h, s, dh) @ (h, dh, kv) -> (h, s, kv)
        scores = np.matmul(Q, K.transpose(0, 2, 1)) * self.attn_scale

        kv_len = K.shape[1]
        if seq_len > 1:
            # Causal mask for multi-token input: each query position attends
            # only to keys up to and including itself.
            q_positions = np.arange(kv_len - seq_len, kv_len)
            k_positions = np.arange(kv_len)
            mask = q_positions[:, None] < k_positions[None, :]
            scores[:, mask] = -1e9

        attn = softmax(scores, axis=-1)
        # (h, s, kv) @ (h, kv, dh) -> (h, s, dh) -> (s, h*dh)
        out = np.matmul(attn, V).transpose(1, 0, 2).reshape(seq_len, h * dh)
        return out @ self._W_o_flat

    def ffn(self, x):
        h = gelu(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def forward(self, x, kv_cache=None):
        normed = layer_norm(x, self.ln1_g, self.ln1_b)
        x = x + self.attention(normed, kv_cache)
        normed = layer_norm(x, self.ln2_g, self.ln2_b)
        x = x + self.ffn(normed)
        return x


class ExitHead:
    """Lightweight exit head: LayerNorm -> Linear -> Softmax."""

    def __init__(self, d_model, vocab_size, rng: np.random.Generator):
        self.ln_g = np.ones(d_model, dtype=np.float32)
        self.ln_b = np.zeros(d_model, dtype=np.float32)
        self.W = rng.normal(0, 0.02, (d_model, vocab_size)).astype(np.float32)
        self.b = np.zeros(vocab_size, dtype=np.float32)

    def logits(self, x):
        normed = layer_norm(x, self.ln_g, self.ln_b)
        return normed @ self.W + self.b

    def get_probs(self, hidden_last):
        """hidden_last: (d_model,) -> probs: (vocab_size,), entropy: float"""
        lgts = self.logits(hidden_last[np.newaxis, :]).squeeze(0)
        probs = softmax(lgts)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return probs, entropy


class Transformer:
    """Full transformer with exit heads for WADI."""

    def __init__(self, config: TransformerConfig, seed=42):
        self.config = config
        rng = np.random.default_rng(seed)

        self.tok_emb = rng.normal(0, 0.02, (config.vocab_size, config.d_model)).astype(np.float32)
        self.pos_emb = rng.normal(0, 0.02, (config.max_seq_len, config.d_model)).astype(np.float32)

        self.layers = [TransformerLayer(config, i, rng) for i in range(config.n_layers)]

        self.final_ln_g = np.ones(config.d_model, dtype=np.float32)
        self.final_ln_b = np.zeros(config.d_model, dtype=np.float32)

        # Exit heads at specified layers (1-indexed: layer 4 means after self.layers[3])
        self.exit_heads = {}
        for ell in config.exit_layers:
            self.exit_heads[ell] = ExitHead(config.d_model, config.vocab_size, rng)

        self.flop_counter = 0

    def create_kv_cache(self):
        return KVCache(
            self.config.n_layers,
            n_heads=self.config.n_heads,
            d_head=self.config.d_head,
            max_seq_len=self.config.max_seq_len,
            dtype=self.tok_emb.dtype,
        )

    def embed(self, token_ids, start_pos=0):
        seq_len = len(token_ids)
        positions = np.arange(start_pos, start_pos + seq_len)
        return self.tok_emb[token_ids] + self.pos_emb[positions]

    def forward_layer_range(self, x, start_layer, end_layer, kv_cache=None):
        """Run layers [start_layer, end_layer) on x. Tracks FLOPs."""
        d = self.config.d_model
        ff = self.config.d_ff
        seq_len = x.shape[0]

        for i in range(start_layer, end_layer):
            x = self.layers[i].forward(x, kv_cache)
            self.flop_counter += seq_len * (4 * d * d + 2 * d * ff)

        return x

    def full_forward_single(self, token_id, kv_cache, pos):
        """
        Full forward pass for a single token through all layers.
        Returns (probs, entropy) from the final exit head.
        """
        x = self.embed(np.array([token_id]), start_pos=pos)
        x = self.forward_layer_range(x, 0, self.config.n_layers, kv_cache)
        x_normed = layer_norm(x, self.final_ln_g, self.final_ln_b)
        return self.exit_heads[self.config.n_layers].get_probs(x_normed[-1])

    def full_forward_batch(self, token_ids, kv_cache, start_pos):
        """
        Full forward pass over a sequence of K tokens in a single call.

        This is the fast verification path for speculative / self-speculative
        decoding: instead of K sequential single-token forwards, we run one
        K-token pass with a causal mask. FLOP accounting is unchanged
        (total work is the same), but wall-clock drops because BLAS GEMMs
        are fed much larger matrices.

        Returns (probs, entropies) of shape (K, vocab) and (K,).
        """
        token_ids = np.asarray(token_ids, dtype=np.int64)
        K = int(token_ids.shape[0])
        x = self.embed(token_ids, start_pos=start_pos)
        x = self.forward_layer_range(x, 0, self.config.n_layers, kv_cache)
        x_normed = layer_norm(x, self.final_ln_g, self.final_ln_b)
        head = self.exit_heads[self.config.n_layers]
        logits = head.logits(x_normed)  # (K, vocab)
        probs = softmax(logits, axis=-1)
        entropies = -np.sum(probs * np.log(probs + 1e-10), axis=-1)
        return probs, entropies

    def forward_with_exits(self, token_id, kv_cache, pos):
        """
        Forward pass that yields (hidden_state, probs, entropy, exit_layer)
        at each exit point. Used by WADI for early exit decisions.
        """
        x = self.embed(np.array([token_id]), start_pos=pos)
        prev_layer = 0
        results = []

        for ell in self.config.exit_layers:
            x = self.forward_layer_range(x, prev_layer, ell, kv_cache)
            probs, entropy = self.exit_heads[ell].get_probs(x[-1])
            results.append({
                'hidden': x.copy(),
                'probs': probs,
                'entropy': entropy,
                'exit_layer': ell,
            })
            prev_layer = ell

        return results

    def forward_to_layer(self, token_id, kv_cache, pos, target_layer):
        """Forward pass up to a specific layer. Returns (hidden, probs, entropy)."""
        x = self.embed(np.array([token_id]), start_pos=pos)
        exit_layers_sorted = sorted(self.config.exit_layers)

        prev = 0
        for ell in exit_layers_sorted:
            if ell > target_layer:
                break
            x = self.forward_layer_range(x, prev, ell, kv_cache)
            prev = ell

        probs, entropy = self.exit_heads[target_layer].get_probs(x[-1])
        return x, probs, entropy

    def verify_sequence(self, token_ids, kv_cache, start_pos):
        """
        Full-depth forward pass on a sequence of tokens.
        Returns list of (probs, entropy) for each token position.
        Used for verification in speculative decoding.
        """
        results = []
        for i, tid in enumerate(token_ids):
            probs, entropy = self.full_forward_single(tid, kv_cache, start_pos + i)
            results.append((probs, entropy))
        return results

    def _independent_layer_forward(self, x, layer_idx):
        """
        Apply a single transformer layer to a stack of N independent
        single-token samples in parallel.

        x: (N, d_model) — each row is a separate "sequence of length 1"
        returns: (N, d_model)

        When each sample has only one token at position 0, self-attention
        softmax is over a single score and always equals 1.0, so the
        attention output reduces to V projected through W_o. This lets us
        run the whole batch as a few flat GEMMs — orders of magnitude faster
        than looping N times with a fresh KV cache.
        """
        layer = self.layers[layer_idx]
        h, dh, d = layer.n_heads, layer.d_head, layer.d_model

        normed = layer_norm(x, layer.ln1_g, layer.ln1_b)
        qkv = normed @ layer._W_qkv  # (N, 3*h*dh)
        # We only need V — Q/K don't affect the output for a 1-token attention.
        V = qkv[:, 2 * h * dh :]  # (N, h*dh)
        attn_out = V @ layer._W_o_flat  # (N, d)
        x = x + attn_out

        normed = layer_norm(x, layer.ln2_g, layer.ln2_b)
        x = x + layer.ffn(normed)
        return x

    def simulate_trained_exits(self, n_calibration=200, seed=42):
        """
        Simulate distillation training by fitting exit heads to match the
        full model's predictions on calibration data.

        For each exit layer, we:
        1. Collect hidden states and full-model target logits
        2. Fit the exit head via ridge-regularized least squares on logits
        3. Add depth-proportional noise (shallower = more noise) to mimic
           the reality that shallower heads can't perfectly match the full
           model.

        The calibration tokens are independent single-token samples with no
        shared context, so we can process all of them as one batched forward
        through the layers — this is ~50-100x faster than the original
        one-token-at-a-time loop.
        """
        rng = np.random.default_rng(seed)
        config = self.config
        final_layer = config.n_layers
        final_head = self.exit_heads[final_layer]

        # Generate calibration tokens.
        cal_tokens = rng.integers(0, config.vocab_size, size=n_calibration)

        # Batched forward: treat calibration samples as an independent axis.
        # All samples share position 0, so pos_emb[0] is the same row everywhere.
        x = self.tok_emb[cal_tokens] + self.pos_emb[0][None, :]  # (N, d_model)

        hidden_by_layer = {}  # ell -> (N, d_model) hidden state at that exit
        exit_layers = sorted(self.exit_heads.keys())

        li = 0
        for ell in exit_layers:
            while li < ell:
                x = self._independent_layer_forward(x, li)
                li += 1
            hidden_by_layer[ell] = x.copy()

        # Final hidden state is at `final_layer`.
        h_full = hidden_by_layer[final_layer]
        T = final_head.logits(h_full)  # (N, vocab)

        for ell in exit_layers:
            if ell == final_layer:
                continue

            H = hidden_by_layer[ell]  # (N, d_model)
            head = self.exit_heads[ell]

            # Apply this head's LayerNorm to all samples at once.
            H_normed = layer_norm(H, head.ln_g, head.ln_b)

            # Ridge-regularized least squares: H_aug @ W_fit ≈ T
            H_aug = np.concatenate(
                [H_normed, np.ones((H_normed.shape[0], 1), dtype=H_normed.dtype)],
                axis=1,
            )
            ridge = 0.01 * np.eye(H_aug.shape[1], dtype=H_aug.dtype)
            W_fit = np.linalg.solve(H_aug.T @ H_aug + ridge, H_aug.T @ T)

            # Depth-proportional noise (shallower = noisier head).
            depth_frac = ell / final_layer
            noise = rng.normal(0, 0.1 * (1 - depth_frac), W_fit.shape).astype(np.float32)

            head.W = (W_fit[:-1] + noise[:-1]).astype(np.float32)
            head.b = (W_fit[-1] + noise[-1]).astype(np.float32)

    def reset_flops(self):
        self.flop_counter = 0

    def get_flops(self):
        return self.flop_counter
