"""
WADI: Wavefront Adaptive-Depth Inference
=========================================

Core inference algorithm implementing:
1. Entropy-guided early exit with confidence cascading
2. Self-speculative draft generation (model is its own draft)
3. Verification via rejection sampling (lossless guarantee)
4. Adaptive threshold tuning based on acceptance rate
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from model import Transformer, TransformerConfig, KVCache, softmax


@dataclass
class WADIConfig:
    # Initial entropy thresholds per exit layer (lower = more aggressive early exit)
    initial_thresholds: Dict[int, float] = field(default_factory=dict)
    # Max number of speculative draft tokens before verification
    max_draft_len: int = 6
    # Target acceptance rate for adaptive thresholds
    target_acceptance_rate: float = 0.80
    # Learning rate for threshold adaptation
    threshold_lr: float = 0.05
    # Minimum / maximum entropy thresholds
    min_threshold: float = 0.5
    max_threshold: float = 8.0
    # Temperature for sampling
    temperature: float = 1.0

    def __post_init__(self):
        if not self.initial_thresholds:
            # Default: shallower layers get tighter thresholds
            # (only very confident predictions exit early)
            self.initial_thresholds = {
                4: 1.5,
                8: 2.5,
                12: 4.0,
                16: float('inf'),  # final layer always "exits"
            }


@dataclass
class DraftToken:
    """A speculatively generated token with its draft metadata."""
    token_id: int
    exit_layer: int
    draft_probs: np.ndarray
    entropy: float


@dataclass
class WADIStats:
    """Statistics for analyzing WADI performance."""
    tokens_generated: int = 0
    total_draft_tokens: int = 0
    accepted_draft_tokens: int = 0
    rejected_draft_tokens: int = 0
    verification_passes: int = 0
    exit_layer_counts: Dict[int, int] = field(default_factory=dict)
    threshold_history: List[Dict[int, float]] = field(default_factory=list)
    flops_draft: int = 0
    flops_verify: int = 0

    @property
    def acceptance_rate(self):
        if self.total_draft_tokens == 0:
            return 0.0
        return self.accepted_draft_tokens / self.total_draft_tokens

    @property
    def avg_exit_depth(self):
        if not self.exit_layer_counts:
            return 0
        total = sum(self.exit_layer_counts.values())
        weighted = sum(k * v for k, v in self.exit_layer_counts.items())
        return weighted / total if total > 0 else 0

    def summary(self):
        lines = [
            "=== WADI Inference Statistics ===",
            f"  Tokens generated:        {self.tokens_generated}",
            f"  Total draft tokens:      {self.total_draft_tokens}",
            f"  Accepted drafts:         {self.accepted_draft_tokens}",
            f"  Rejected drafts:         {self.rejected_draft_tokens}",
            f"  Acceptance rate:         {self.acceptance_rate:.1%}",
            f"  Verification passes:     {self.verification_passes}",
            f"  Avg tokens/verify:       {self.total_draft_tokens / max(1, self.verification_passes):.1f}",
            f"  Avg exit depth:          {self.avg_exit_depth:.1f}",
            f"  Exit layer distribution: {dict(sorted(self.exit_layer_counts.items()))}",
            f"  Draft FLOPs:             {self.flops_draft:,}",
            f"  Verify FLOPs:            {self.flops_verify:,}",
            f"  Total FLOPs:             {self.flops_draft + self.flops_verify:,}",
        ]
        return "\n".join(lines)


class WADIEngine:
    """
    Main WADI inference engine.

    Generates tokens using entropy-guided early exit for drafting,
    followed by full-depth verification with rejection sampling.
    """

    def __init__(self, model: Transformer, config: WADIConfig = None):
        self.model = model
        self.config = config or WADIConfig()
        self.thresholds = dict(self.config.initial_thresholds)
        self.stats = WADIStats()
        self.rng = np.random.default_rng(123)

        # Ensure thresholds exist for all exit layers
        for ell in model.config.exit_layers:
            if ell not in self.thresholds:
                if ell == model.config.n_layers:
                    self.thresholds[ell] = float('inf')
                else:
                    self.thresholds[ell] = 3.0

    def _sample_token(self, probs: np.ndarray) -> int:
        """Sample a token from probability distribution."""
        if self.config.temperature != 1.0:
            logits = np.log(probs + 1e-10) / self.config.temperature
            probs = softmax(logits)
        return self.rng.choice(len(probs), p=probs)

    def _draft_one_token(self, token_id: int, kv_cache: KVCache, pos: int) -> DraftToken:
        """
        Run a single token through layers with TRUE early exit.
        Processes layer groups incrementally. Stops as soon as entropy
        drops below the threshold — remaining layers are NOT computed.
        """
        model = self.model
        config = model.config
        exit_layers = sorted(config.exit_layers)

        # Embed
        x = model.embed(np.array([token_id]), start_pos=pos)
        prev_layer = 0

        for ell in exit_layers:
            # Process this layer group
            x = model.forward_layer_range(x, prev_layer, ell, kv_cache)
            prev_layer = ell

            # Check entropy at this exit point
            probs, entropy = model.exit_heads[ell].get_probs(x[-1])
            threshold = self.thresholds.get(ell, float('inf'))

            if entropy < threshold or ell == config.n_layers:
                # EXIT HERE — do not process deeper layers
                # For layers we didn't process, we need to ensure the KV cache
                # doesn't have stale entries. The cache only has entries up to
                # layer `ell` because forward_layer_range only wrote up to there.
                self.stats.exit_layer_counts[ell] = self.stats.exit_layer_counts.get(ell, 0) + 1

                sampled = self._sample_token(probs)
                return DraftToken(
                    token_id=sampled,
                    exit_layer=ell,
                    draft_probs=probs,
                    entropy=entropy,
                )

        # Shouldn't reach here
        raise RuntimeError("No exit point found")

    def _generate_drafts(self, context_token: int, kv_cache: KVCache,
                         start_pos: int) -> List[DraftToken]:
        """
        Generate a sequence of draft tokens using early exit.
        Each draft uses the previous draft's output as input.
        """
        drafts = []
        current_token = context_token
        current_pos = start_pos

        # Save the KV cache state before drafting so we can restore for verification
        for _ in range(self.config.max_draft_len):
            draft = self._draft_one_token(current_token, kv_cache, current_pos)
            drafts.append(draft)
            current_token = draft.token_id
            current_pos += 1

        return drafts

    def _verify_and_accept(self, drafts: List[DraftToken],
                           verify_cache: KVCache,
                           verify_start_pos: int,
                           context_token: int) -> Tuple[List[int], int]:
        """
        Verify draft tokens using full-depth forward passes.
        Uses rejection sampling for lossless generation.

        Returns (accepted_tokens, n_accepted).
        """
        self.stats.verification_passes += 1
        accepted_tokens = []

        # We need to verify: given context, is each draft token acceptable?
        # Run the full model on the context token first to get p_full for draft[0]
        current_token = context_token

        for i, draft in enumerate(drafts):
            self.model.reset_flops()

            # Full-depth forward pass
            full_probs, full_entropy = self.model.full_forward_single(
                current_token, verify_cache, verify_start_pos + i
            )
            self.stats.flops_verify += self.model.get_flops()

            # Rejection sampling criterion
            draft_prob = draft.draft_probs[draft.token_id]
            full_prob = full_probs[draft.token_id]

            # Accept with probability min(1, p_full / p_draft)
            accept_ratio = min(1.0, full_prob / (draft_prob + 1e-10))

            if self.rng.random() < accept_ratio:
                # Accept this draft token
                accepted_tokens.append(draft.token_id)
                current_token = draft.token_id
                self.stats.accepted_draft_tokens += 1
            else:
                # Reject: resample from adjusted distribution
                # p_adjusted(x) = max(0, p_full(x) - p_draft(x)) / Z
                adjusted = np.maximum(0, full_probs - draft.draft_probs)
                z = np.sum(adjusted)
                if z > 1e-10:
                    adjusted /= z
                    resampled = self._sample_token(adjusted)
                else:
                    resampled = self._sample_token(full_probs)
                accepted_tokens.append(resampled)
                self.stats.rejected_draft_tokens += 1
                break  # Discard remaining drafts

        return accepted_tokens, len(accepted_tokens)

    def _adapt_thresholds(self):
        """Adjust entropy thresholds based on recent acceptance rate."""
        alpha = self.stats.acceptance_rate
        alpha_target = self.config.target_acceptance_rate
        delta = self.config.threshold_lr * (alpha - alpha_target)

        for ell in self.thresholds:
            if ell == self.model.config.n_layers:
                continue  # Final layer always exits
            self.thresholds[ell] = np.clip(
                self.thresholds[ell] + delta,
                self.config.min_threshold,
                self.config.max_threshold,
            )

        self.stats.threshold_history.append(dict(self.thresholds))

    def generate(self, prompt_tokens: np.ndarray, max_new_tokens: int,
                 verbose: bool = False) -> np.ndarray:
        """
        Generate tokens using WADI.

        Args:
            prompt_tokens: (seq_len,) array of token IDs
            max_new_tokens: number of tokens to generate
            verbose: print step-by-step info

        Returns:
            (prompt_len + generated_len,) array of all token IDs
        """
        self.stats = WADIStats()  # Reset stats
        self.model.reset_flops()

        config = self.model.config
        generated = list(prompt_tokens)
        if not generated:
            raise ValueError("prompt_tokens must contain at least one token")

        # --- Prefill: process prompt through full model ---
        prefill_cache = KVCache(config.n_layers)
        for i, tok in enumerate(prompt_tokens[:-1]):
            self.model.full_forward_single(tok, prefill_cache, i)

        # Main generation loop
        tokens_left = max_new_tokens
        current_token = int(prompt_tokens[-1])
        current_pos = len(prompt_tokens) - 1

        while tokens_left > 0:
            # Phase 1: Draft generation with early exit
            draft_cache = prefill_cache.clone()
            self.model.reset_flops()

            n_drafts = min(self.config.max_draft_len, tokens_left)
            drafts = []
            draft_token = current_token
            draft_pos = current_pos

            for _ in range(n_drafts):
                draft = self._draft_one_token(draft_token, draft_cache, draft_pos)
                drafts.append(draft)
                draft_token = draft.token_id
                draft_pos += 1
                self.stats.total_draft_tokens += 1

            self.stats.flops_draft += self.model.get_flops()

            if verbose:
                exit_depths = [d.exit_layer for d in drafts]
                draft_ids = [d.token_id for d in drafts]
                print(f"  Drafted {len(drafts)} tokens: {draft_ids} at depths {exit_depths}")

            # Phase 2: Verification using full model
            verify_cache = prefill_cache  # Reuse the clean cache
            accepted, n_accepted = self._verify_and_accept(
                drafts, verify_cache, current_pos, current_token
            )

            if verbose:
                print(f"  Accepted {n_accepted}/{len(drafts)}: {accepted}")

            # Phase 3: Commit accepted tokens
            generated.extend(accepted)
            current_pos += n_accepted
            tokens_left -= n_accepted
            self.stats.tokens_generated += n_accepted
            current_token = accepted[-1]

            # The verify_cache (which is prefill_cache) has been updated
            # through the verification forward passes up to the accepted tokens.
            # We need to truncate it to remove any extra entries.
            target_len = current_pos
            for layer_idx in range(config.n_layers):
                cache_len = verify_cache.get_seq_len(layer_idx)
                if cache_len > target_len:
                    verify_cache.truncate(layer_idx, target_len)

            # Phase 4: Adaptive threshold tuning
            self._adapt_thresholds()

        return np.array(generated[:len(prompt_tokens) + max_new_tokens])

    def get_stats(self) -> WADIStats:
        return self.stats


class StandardInference:
    """Baseline: standard autoregressive inference (no speculation, no early exit)."""

    def __init__(self, model: Transformer, temperature: float = 1.0):
        self.model = model
        self.temperature = temperature
        self.rng = np.random.default_rng(456)
        self.total_flops = 0

    def generate(self, prompt_tokens: np.ndarray, max_new_tokens: int,
                 verbose: bool = False) -> np.ndarray:
        config = self.model.config
        self.model.reset_flops()
        generated = list(prompt_tokens)
        if not generated:
            raise ValueError("prompt_tokens must contain at least one token")

        kv_cache = KVCache(config.n_layers)

        # Prefill
        for i, tok in enumerate(prompt_tokens[:-1]):
            self.model.full_forward_single(tok, kv_cache, i)

        current_token = int(prompt_tokens[-1])
        current_pos = len(prompt_tokens) - 1

        # Autoregressive generation
        for step in range(max_new_tokens):
            probs, entropy = self.model.full_forward_single(current_token, kv_cache, current_pos)

            if self.temperature != 1.0:
                logits = np.log(probs + 1e-10) / self.temperature
                probs = softmax(logits)

            next_token = self.rng.choice(len(probs), p=probs)
            generated.append(next_token)
            current_token = next_token
            current_pos += 1

            if verbose and step < 10:
                print(f"  Step {step}: token={next_token}, entropy={entropy:.2f}")

        self.total_flops = self.model.get_flops()
        return np.array(generated)


class SpeculativeDecodingBaseline:
    """Baseline: standard speculative decoding with a separate small draft model."""

    def __init__(self, target_model: Transformer, draft_model: Transformer,
                 draft_len: int = 5, temperature: float = 1.0):
        self.target = target_model
        self.draft = draft_model
        self.draft_len = draft_len
        self.temperature = temperature
        self.rng = np.random.default_rng(789)
        self.total_flops_target = 0
        self.total_flops_draft = 0
        self.accepted = 0
        self.total = 0

    def generate(self, prompt_tokens: np.ndarray, max_new_tokens: int,
                 verbose: bool = False) -> np.ndarray:
        self.target.reset_flops()
        self.draft.reset_flops()
        config_target = self.target.config
        config_draft = self.draft.config
        generated = list(prompt_tokens)
        if not generated:
            raise ValueError("prompt_tokens must contain at least one token")

        target_cache = KVCache(config_target.n_layers)
        draft_cache = KVCache(config_draft.n_layers)

        # Prefill both models
        for i, tok in enumerate(prompt_tokens[:-1]):
            self.target.full_forward_single(tok, target_cache, i)
            self.draft.full_forward_single(tok, draft_cache, i)

        tokens_left = max_new_tokens
        current_token = int(prompt_tokens[-1])
        current_pos = len(prompt_tokens) - 1

        while tokens_left > 0:
            # Draft phase
            draft_tokens = []
            draft_probs_list = []
            draft_cache_copy = draft_cache.clone()
            draft_current = current_token
            draft_pos = current_pos

            n = min(self.draft_len, tokens_left)
            for _ in range(n):
                probs, _ = self.draft.full_forward_single(draft_current, draft_cache_copy, draft_pos)
                tok = self.rng.choice(len(probs), p=probs)
                draft_tokens.append(tok)
                draft_probs_list.append(probs)
                draft_current = tok
                draft_pos += 1

            # Verify phase
            accepted_tokens = []
            verify_current = current_token

            for i, dtok in enumerate(draft_tokens):
                pos = current_pos + i
                target_probs, _ = self.target.full_forward_single(
                    verify_current, target_cache, pos
                )
                self.draft.full_forward_single(verify_current, draft_cache, pos)

                tp = target_probs[dtok]
                dp = draft_probs_list[i][dtok]
                accept_ratio = min(1.0, tp / (dp + 1e-10))

                self.total += 1
                if self.rng.random() < accept_ratio:
                    accepted_tokens.append(dtok)
                    verify_current = dtok
                    self.accepted += 1
                else:
                    adjusted = np.maximum(0, target_probs - draft_probs_list[i])
                    z = np.sum(adjusted)
                    if z > 1e-10:
                        adjusted /= z
                        resampled = self.rng.choice(len(adjusted), p=adjusted)
                    else:
                        resampled = self.rng.choice(len(target_probs), p=target_probs)
                    accepted_tokens.append(resampled)
                    break

            generated.extend(accepted_tokens)
            tokens_left -= len(accepted_tokens)
            current_token = accepted_tokens[-1]
            current_pos += len(accepted_tokens)

            if verbose and len(generated) < len(prompt_tokens) + 20:
                print(f"  Spec: drafted {len(draft_tokens)}, accepted {len(accepted_tokens)}")

        self.total_flops_target = self.target.get_flops()
        self.total_flops_draft = self.draft.get_flops()
        return np.array(generated[:len(prompt_tokens) + max_new_tokens])
