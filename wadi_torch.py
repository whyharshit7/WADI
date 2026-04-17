"""
WADI engine, PyTorch backend.

Mirrors ``wadi.WADIEngine`` but runs on top of ``model_torch.TorchTransformer``.
The algorithm is identical — entropy-guided early exit, batched full-depth
verification with rejection sampling, adaptive threshold tuning. The only
changes are:

* Probabilities and draft state are ``torch.Tensor`` rather than ``np.ndarray``.
* Sampling uses ``torch.multinomial`` (so the same RNG seed on GPU gives
  reproducible streams).
* One batched ``full_forward_batch`` call per round replaces K sequential
  single-token verifies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch

from model_torch import TorchKVCache, TorchTransformer


@dataclass
class WADITorchConfig:
    initial_thresholds: Dict[int, float] = field(default_factory=dict)
    max_draft_len: int = 6
    target_acceptance_rate: float = 0.80
    threshold_lr: float = 0.05
    min_threshold: float = 0.5
    max_threshold: float = 8.0
    temperature: float = 1.0

    def __post_init__(self):
        if not self.initial_thresholds:
            self.initial_thresholds = {4: 1.5, 8: 2.5, 12: 4.0, 16: float("inf")}


@dataclass
class DraftToken:
    token_id: int
    exit_layer: int
    draft_probs: torch.Tensor  # (vocab,)
    entropy: float


@dataclass
class WADIStats:
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
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.accepted_draft_tokens / self.total_draft_tokens

    @property
    def avg_exit_depth(self) -> float:
        if not self.exit_layer_counts:
            return 0.0
        total = sum(self.exit_layer_counts.values())
        weighted = sum(k * v for k, v in self.exit_layer_counts.items())
        return weighted / total if total > 0 else 0.0

    def summary(self) -> str:
        lines = [
            "=== WADI Inference Statistics (Torch) ===",
            f"  Tokens generated:        {self.tokens_generated}",
            f"  Total draft tokens:      {self.total_draft_tokens}",
            f"  Accepted drafts:         {self.accepted_draft_tokens}",
            f"  Rejected drafts:         {self.rejected_draft_tokens}",
            f"  Acceptance rate:         {self.acceptance_rate:.1%}",
            f"  Verification passes:     {self.verification_passes}",
            f"  Avg exit depth:          {self.avg_exit_depth:.1f}",
            f"  Exit distribution:       {dict(sorted(self.exit_layer_counts.items()))}",
            f"  Draft FLOPs:             {self.flops_draft:,}",
            f"  Verify FLOPs:            {self.flops_verify:,}",
            f"  Total FLOPs:             {self.flops_draft + self.flops_verify:,}",
        ]
        return "\n".join(lines)


class WADIEngineTorch:
    """PyTorch-backed WADI inference engine.

    Supports CPU and CUDA. For GPU runs, pass a model already moved to the
    target device; the engine allocates all intermediate tensors on the same
    device as the model.
    """

    def __init__(self, model: TorchTransformer, config: WADITorchConfig | None = None):
        self.model = model
        self.config = config or WADITorchConfig()
        self.thresholds: Dict[int, float] = dict(self.config.initial_thresholds)
        self.stats = WADIStats()
        # torch.Generator keeps sampling reproducible per engine instance.
        self._gen = torch.Generator(device=model.device).manual_seed(123)

        for ell in model.config.exit_layers:
            if ell not in self.thresholds:
                self.thresholds[ell] = (
                    float("inf") if ell == model.config.n_layers else 3.0
                )

    # ---- Sampling ---------------------------------------------------------

    def _sample(self, probs: torch.Tensor) -> int:
        if self.config.temperature != 1.0:
            logits = torch.log(probs.clamp_min(1e-10)) / self.config.temperature
            probs = torch.softmax(logits, dim=-1)
        idx = torch.multinomial(probs, num_samples=1, generator=self._gen)
        return int(idx.item())

    def _random_uniform(self) -> float:
        return float(torch.rand((), generator=self._gen, device=self.model.device).item())

    # ---- Draft phase ------------------------------------------------------

    @torch.no_grad()
    def _draft_one_token(
        self, token_id: int, kv_cache: TorchKVCache, pos: int
    ) -> DraftToken:
        model = self.model
        config = model.config
        device = model.device
        exit_layers = sorted(config.exit_layers)

        ids = torch.as_tensor([int(token_id)], dtype=torch.long, device=device)
        x = model.tok_emb(ids) + model.pos_emb(
            torch.as_tensor([pos], dtype=torch.long, device=device)
        )
        prev_layer = 0

        for ell in exit_layers:
            # Process layer group [prev_layer, ell).
            for li in range(prev_layer, ell):
                x = model.layers[li](x, kv_cache)
            model._charge_flops(x.shape[0], prev_layer, ell)
            prev_layer = ell

            head = model.exit_heads[str(ell)]
            logits = head.logits(x[-1:])
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            entropy = float(-(probs * torch.log(probs.clamp_min(1e-10))).sum().item())

            threshold = self.thresholds.get(ell, float("inf"))
            if entropy < threshold or ell == config.n_layers:
                self.stats.exit_layer_counts[ell] = self.stats.exit_layer_counts.get(ell, 0) + 1
                sampled = self._sample(probs)
                return DraftToken(
                    token_id=sampled,
                    exit_layer=ell,
                    draft_probs=probs,
                    entropy=entropy,
                )

        raise RuntimeError("No exit point found")

    # ---- Verify phase -----------------------------------------------------

    @torch.no_grad()
    def _verify_and_accept(
        self,
        drafts: List[DraftToken],
        verify_cache: TorchKVCache,
        verify_start_pos: int,
        context_token: int,
    ) -> Tuple[List[int], int, Dict[int, int], Dict[int, int]]:
        self.stats.verification_passes += 1
        accepted_tokens: List[int] = []
        accepted_by_exit: Dict[int, int] = {}
        rejected_by_exit: Dict[int, int] = {}

        # Batched verify: one forward over [ctx, d0, ..., d_{K-2}].
        K = len(drafts)
        verify_input = [context_token] + [d.token_id for d in drafts[:-1]]

        self.model.reset_flops()
        full_probs_batch, _ = self.model.full_forward_batch(
            verify_input, verify_cache, verify_start_pos
        )
        self.stats.flops_verify += self.model.get_flops()

        for i, draft in enumerate(drafts):
            ell = draft.exit_layer
            full_probs = full_probs_batch[i]

            draft_prob = draft.draft_probs[draft.token_id]
            full_prob = full_probs[draft.token_id]
            accept_ratio = min(1.0, float(full_prob / (draft_prob + 1e-10)))

            if self._random_uniform() < accept_ratio:
                accepted_tokens.append(int(draft.token_id))
                self.stats.accepted_draft_tokens += 1
                self.stats.exit_layer_accepted[ell] = self.stats.exit_layer_accepted.get(ell, 0) + 1
                accepted_by_exit[ell] = accepted_by_exit.get(ell, 0) + 1
            else:
                adjusted = torch.clamp(full_probs - draft.draft_probs, min=0)
                z = adjusted.sum()
                if z > 1e-10:
                    adjusted = adjusted / z
                    resampled = self._sample(adjusted)
                else:
                    resampled = self._sample(full_probs)
                accepted_tokens.append(int(resampled))
                self.stats.rejected_draft_tokens += 1
                self.stats.exit_layer_rejected[ell] = self.stats.exit_layer_rejected.get(ell, 0) + 1
                rejected_by_exit[ell] = rejected_by_exit.get(ell, 0) + 1
                break

        return accepted_tokens, len(accepted_tokens), accepted_by_exit, rejected_by_exit

    # ---- Adaptive thresholds ---------------------------------------------

    def _adapt_thresholds(
        self,
        accepted_by_exit: Dict[int, int],
        rejected_by_exit: Dict[int, int],
    ) -> None:
        alpha_target = self.config.target_acceptance_rate
        for ell in self.thresholds:
            if ell == self.model.config.n_layers:
                continue
            accepted = accepted_by_exit.get(ell, 0)
            rejected = rejected_by_exit.get(ell, 0)
            attempts = accepted + rejected
            if attempts == 0:
                continue
            alpha = accepted / attempts
            delta = self.config.threshold_lr * (alpha - alpha_target)
            new_th = self.thresholds[ell] + delta
            self.thresholds[ell] = float(
                max(self.config.min_threshold, min(self.config.max_threshold, new_th))
            )
        self.stats.threshold_history.append(dict(self.thresholds))

    # ---- Top-level generate loop -----------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens,
        max_new_tokens: int,
        verbose: bool = False,
    ):
        self.stats = WADIStats()
        self.model.reset_flops()
        if isinstance(prompt_tokens, torch.Tensor):
            prompt_list = prompt_tokens.tolist()
        else:
            prompt_list = list(prompt_tokens)
        if not prompt_list:
            raise ValueError("prompt_tokens must contain at least one token")

        config = self.model.config
        generated = list(prompt_list)

        prefill_cache = self.model.create_kv_cache()
        for i, tok in enumerate(prompt_list[:-1]):
            self.model.full_forward_single(tok, prefill_cache, i)

        tokens_left = max_new_tokens
        current_token = int(prompt_list[-1])
        current_pos = len(prompt_list) - 1

        while tokens_left > 0:
            draft_cache = prefill_cache.clone()
            self.model.reset_flops()

            n_drafts = min(self.config.max_draft_len, tokens_left)
            drafts: List[DraftToken] = []
            draft_token = current_token
            draft_pos = current_pos

            for _ in range(n_drafts):
                draft = self._draft_one_token(draft_token, draft_cache, draft_pos)
                drafts.append(draft)
                draft_token = draft.token_id
                draft_pos += 1
                self.stats.total_draft_tokens += 1

            self.stats.flops_draft += self.model.get_flops()

            verify_cache = prefill_cache
            accepted, n_accepted, accepted_by_exit, rejected_by_exit = self._verify_and_accept(
                drafts, verify_cache, current_pos, current_token
            )

            if verbose:
                print(f"  Accepted {n_accepted}/{len(drafts)}: {accepted}")

            generated.extend(accepted)
            current_pos += n_accepted
            tokens_left -= n_accepted
            self.stats.tokens_generated += n_accepted
            current_token = accepted[-1]

            target_len = current_pos
            for layer_idx in range(config.n_layers):
                if verify_cache.get_seq_len(layer_idx) > target_len:
                    verify_cache.truncate(layer_idx, target_len)

            self._adapt_thresholds(accepted_by_exit, rejected_by_exit)

        return generated[: len(prompt_list) + max_new_tokens]

    def get_stats(self) -> WADIStats:
        return self.stats
