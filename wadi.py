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
from typing import List, Tuple, Dict
from model import Transformer, KVCache, softmax


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
    exit_layer_accepted: Dict[int, int] = field(default_factory=dict)
    exit_layer_rejected: Dict[int, int] = field(default_factory=dict)
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
        layer_acceptance = {}
        for ell in sorted(set(self.exit_layer_accepted) | set(self.exit_layer_rejected)):
            accepted = self.exit_layer_accepted.get(ell, 0)
            rejected = self.exit_layer_rejected.get(ell, 0)
            attempts = accepted + rejected
            if attempts > 0:
                layer_acceptance[ell] = accepted / attempts

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
            f"  Exit acceptance:         {dict(sorted((k, round(v, 3)) for k, v in layer_acceptance.items()))}",
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
                           context_token: int) -> Tuple[List[int], int, Dict[int, int], Dict[int, int]]:
        """
        Verify draft tokens using one full-depth batched forward pass over
        [context_token, draft[0], ..., draft[K-2]].

        The output at position i is the full-model distribution for draft[i],
        which is exactly what we need for the rejection-sampling check.
        Using a single K-token forward instead of K sequential 1-token
        forwards yields a large wall-clock speedup (same total FLOPs).

        Uses rejection sampling for lossless generation.
        """
        self.stats.verification_passes += 1
        accepted_tokens: List[int] = []
        accepted_by_exit: Dict[int, int] = {}
        rejected_by_exit: Dict[int, int] = {}

        # Build the K-token verification input. Position 0 is context_token,
        # positions 1..K-1 are draft[0..K-2]. We don't include draft[K-1]
        # because we never need the distribution "after" it (we'd only need
        # that to sample a bonus token, which this algorithm skips).
        K = len(drafts)
        verify_input = np.empty(K, dtype=np.int64)
        verify_input[0] = context_token
        if K > 1:
            verify_input[1:] = [d.token_id for d in drafts[:-1]]

        self.model.reset_flops()
        full_probs_batch, _ = self.model.full_forward_batch(
            verify_input, verify_cache, verify_start_pos
        )
        self.stats.flops_verify += self.model.get_flops()

        # Walk the K draft tokens against the K returned distributions,
        # stopping at the first rejection. The KV cache was updated for all
        # K positions by the batched forward; the outer generate() loop
        # truncates it back to the accepted prefix.
        for i, draft in enumerate(drafts):
            ell = draft.exit_layer
            full_probs = full_probs_batch[i]

            draft_prob = draft.draft_probs[draft.token_id]
            full_prob = full_probs[draft.token_id]

            # Accept with probability min(1, p_full / p_draft).
            accept_ratio = min(1.0, full_prob / (draft_prob + 1e-10))

            if self.rng.random() < accept_ratio:
                accepted_tokens.append(int(draft.token_id))
                self.stats.accepted_draft_tokens += 1
                self.stats.exit_layer_accepted[ell] = self.stats.exit_layer_accepted.get(ell, 0) + 1
                accepted_by_exit[ell] = accepted_by_exit.get(ell, 0) + 1
            else:
                # Reject: resample from max(0, p_full - p_draft), renormalized.
                adjusted = np.maximum(0, full_probs - draft.draft_probs)
                z = np.sum(adjusted)
                if z > 1e-10:
                    adjusted /= z
                    resampled = self._sample_token(adjusted)
                else:
                    resampled = self._sample_token(full_probs)
                accepted_tokens.append(int(resampled))
                self.stats.rejected_draft_tokens += 1
                self.stats.exit_layer_rejected[ell] = self.stats.exit_layer_rejected.get(ell, 0) + 1
                rejected_by_exit[ell] = rejected_by_exit.get(ell, 0) + 1
                break  # Discard remaining drafts.

        return accepted_tokens, len(accepted_tokens), accepted_by_exit, rejected_by_exit

    def _adapt_thresholds(self, accepted_by_exit: Dict[int, int], rejected_by_exit: Dict[int, int]):
        """Adjust entropy thresholds using each exit layer's local acceptance signal."""
        alpha_target = self.config.target_acceptance_rate

        for ell in self.thresholds:
            if ell == self.model.config.n_layers:
                continue  # Final layer always exits
            accepted = accepted_by_exit.get(ell, 0)
            rejected = rejected_by_exit.get(ell, 0)
            attempts = accepted + rejected
            if attempts == 0:
                continue

            alpha = accepted / attempts
            delta = self.config.threshold_lr * (alpha - alpha_target)
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
        prefill_cache = self.model.create_kv_cache()
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
            accepted, n_accepted, accepted_by_exit, rejected_by_exit = self._verify_and_accept(
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
            self._adapt_thresholds(accepted_by_exit, rejected_by_exit)

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

        kv_cache = self.model.create_kv_cache()

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

        target_cache = self.target.create_kv_cache()
        draft_cache = self.draft.create_kv_cache()

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

            # Verify phase — single batched forward on the target model.
            # Also keep the draft cache in sync (one batched forward there too).
            K = len(draft_tokens)
            verify_input = np.empty(K, dtype=np.int64)
            verify_input[0] = current_token
            if K > 1:
                verify_input[1:] = draft_tokens[:-1]

            target_probs_batch, _ = self.target.full_forward_batch(
                verify_input, target_cache, current_pos
            )
            self.draft.full_forward_batch(verify_input, draft_cache, current_pos)

            accepted_tokens = []
            for i, dtok in enumerate(draft_tokens):
                target_probs = target_probs_batch[i]
                tp = target_probs[dtok]
                dp = draft_probs_list[i][dtok]
                accept_ratio = min(1.0, tp / (dp + 1e-10))

                self.total += 1
                if self.rng.random() < accept_ratio:
                    accepted_tokens.append(int(dtok))
                    self.accepted += 1
                else:
                    adjusted = np.maximum(0, target_probs - draft_probs_list[i])
                    z = np.sum(adjusted)
                    if z > 1e-10:
                        adjusted /= z
                        resampled = self.rng.choice(len(adjusted), p=adjusted)
                    else:
                        resampled = self.rng.choice(len(target_probs), p=target_probs)
                    accepted_tokens.append(int(resampled))
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
