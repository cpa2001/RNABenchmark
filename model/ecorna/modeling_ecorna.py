"""
EcoRNA model wrappers for BEACON benchmark downstream tasks.
Wraps the base EcoRNA model for sequence classification and token-level tasks.

IMPORTANT: EcoRNA checkpoints trained with Liger Kernel use different parameter names
for MLP layers (gate_proj/up_proj/down_proj vs gate/up/down). This module ensures
correct loading and provides options for inference precision.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
import logging

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecorna import EcoRNAConfig, EcoRNAModel, EcoRNAPreTrainedModel
from ecorna.modeling_ecorna import LIGER_KERNEL_AVAILABLE, FLASH_ATTN_AVAILABLE

logger = logging.getLogger(__name__)

# Backward-compatible strategy aliases
_STRATEGY_ALIASES = {
    "loop_layer_scalar_mix_content": "layer_weighted",
    "mix": "layer_weighted",
    "weighted_layer_average": "weighted_layer_content",
    "weighted_cell_average": "weighted_cell_content",
}
_VALID_STRATEGIES = {
    "layer_weighted",
    "loop_layer_attn_content",
    "weighted_layer_content",
    "weighted_cell_content",
    "mean",
    "cls",
    "cls_tanh",
    "cls_mean_concat",
    "loop_mean_cls",
    "content_mean",
    "cls_ln",
    "loop_mean_content",
    "fixed_cell_content",
    "fixed_cell_cls",
}


def _resolve_strategy(strategy: str) -> str:
    resolved = _STRATEGY_ALIASES.get(strategy, strategy)
    if resolved not in _VALID_STRATEGIES:
        raise ValueError(
            f"Unknown pooling_strategy: {strategy!r}. "
            f"Valid: {_VALID_STRATEGIES}, aliases: {list(_STRATEGY_ALIASES)}"
        )
    if resolved != strategy:
        logger.info("Pooling strategy %r aliased to %r", strategy, resolved)
    return resolved


def _collect_special_token_ids(config: EcoRNAConfig) -> Set[int]:
    token_ids: Set[int] = set()
    for attr in ("cls_token_id", "sep_token_id", "eos_token_id", "pad_token_id"):
        token_id = getattr(config, attr, None)
        if token_id is not None:
            token_ids.add(token_id)
    return token_ids


def _build_sequence_mask(
    input_ids: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    special_token_ids: Set[int],
    exclude_special_tokens: bool,
) -> Tuple[Optional[torch.Tensor], int, int]:
    if attention_mask is not None:
        base_mask = attention_mask.bool()
    elif input_ids is not None:
        base_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        return None, 0, 0

    if not exclude_special_tokens or input_ids is None:
        return base_mask, 0, 0

    mask = base_mask.clone()
    for token_id in special_token_ids:
        mask &= input_ids.ne(token_id)

    empty = ~mask.any(dim=1)
    fallback_examples = int(empty.sum().item())
    fallback_batches = int(fallback_examples > 0)
    if fallback_examples:
        mask[empty] = base_mask[empty]

    return mask, fallback_batches, fallback_examples


def _masked_mean(hidden_states: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return hidden_states.mean(dim=1)
    mask_float = mask.unsqueeze(-1).to(hidden_states.dtype)
    denom = mask_float.sum(dim=1).clamp(min=1e-6)
    return (hidden_states * mask_float).sum(dim=1) / denom


def _parse_pooling_cells(pooling_cells: Optional[Union[str, Sequence[str]]]) -> Tuple[Tuple[int, int], ...]:
    if pooling_cells is None:
        return ()

    if isinstance(pooling_cells, str):
        raw_cells = [item.strip() for item in pooling_cells.split(",") if item.strip()]
    else:
        raw_cells = [str(item).strip() for item in pooling_cells if str(item).strip()]

    parsed: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()
    for raw_cell in raw_cells:
        if ":" not in raw_cell:
            raise ValueError(
                f"Invalid pooling cell spec {raw_cell!r}. Expected 'loop-<n>:layer-<m>'."
            )
        loop_part, layer_part = raw_cell.split(":", 1)
        if not loop_part.startswith("loop-") or not layer_part.startswith("layer-"):
            raise ValueError(
                f"Invalid pooling cell spec {raw_cell!r}. Expected 'loop-<n>:layer-<m>'."
            )
        try:
            loop_idx = int(loop_part.replace("loop-", "", 1))
            layer_idx = int(layer_part.replace("layer-", "", 1))
        except ValueError as exc:
            raise ValueError(
                f"Invalid pooling cell spec {raw_cell!r}. Expected integer loop/layer indices."
            ) from exc
        if loop_idx < 1 or layer_idx < 1:
            raise ValueError(
                f"Invalid pooling cell spec {raw_cell!r}. Loop and layer indices must be >= 1."
            )
        cell = (loop_idx, layer_idx)
        if cell in seen:
            continue
        seen.add(cell)
        parsed.append(cell)
    return tuple(parsed)


def _format_pooling_cells(cells: Sequence[Tuple[int, int]]) -> List[str]:
    return [f"loop-{loop_idx}:layer-{layer_idx}" for loop_idx, layer_idx in cells]


# ---------------------------------------------------------------------------
# Poolers
# ---------------------------------------------------------------------------

class EcoRNAClsTanhPooler(nn.Module):
    def __init__(self, config: EcoRNAConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.activation(self.dense(hidden_states[:, 0]))

class EcoRNALayerWeightedPooler(nn.Module):
    """
    Trainable pooler that learns softmax weights over all (loop, layer)
    hidden states and produces a single [B, hidden_size] representation.

    For each (loop, layer) combination:
      1. Apply final_norm to the raw hidden state
      2. Masked mean-pool over token positions (optionally excluding special tokens)
      3. Multiply by the learned weight

    Sum all weighted contributions to get the final pooled output.
    """

    def __init__(
        self,
        config: EcoRNAConfig,
        final_norm: nn.Module,
        exclude_special_tokens: bool = True,
    ):
        super().__init__()
        self.num_loops = config.num_loops
        self.num_layers = config.num_hidden_layers
        # Keep a plain Python reference so the shared backbone norm is not
        # re-registered under pooler.* and later saved as a duplicated tensor.
        self.__dict__["_final_norm_ref"] = final_norm
        self.exclude_special_tokens = exclude_special_tokens
        self.content_mean_fallback_batches = 0
        self.content_mean_fallback_examples = 0

        # Collect special token IDs from config
        self.special_token_ids = _collect_special_token_ids(config)

        # Uniform initialization so the step-0 behavior is an equal average over
        # all active loop-layer cells.
        self.loop_layer_mix_logits = nn.Parameter(
            torch.zeros(config.num_loops, config.num_hidden_layers)
        )

    def _get_active_logits(self, num_loops: int) -> torch.Tensor:
        """Slice or extend logits to match actual num_loops at runtime."""
        if num_loops <= self.num_loops:
            return self.loop_layer_mix_logits[:num_loops]
        extra = self.loop_layer_mix_logits[-1:].expand(num_loops - self.num_loops, -1)
        return torch.cat([self.loop_layer_mix_logits, extra], dim=0)

    def _build_content_mask(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], int, int]:
        return _build_sequence_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            special_token_ids=self.special_token_ids,
            exclude_special_tokens=self.exclude_special_tokens,
        )

    def forward(
        self,
        loop_layer_hidden_states: Tuple[Tuple[torch.Tensor, ...], ...],
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            loop_layer_hidden_states: (num_loops, num_layers) nested tuple of [B, T, H]
            input_ids: [B, T] for building content mask
            attention_mask: [B, T]
        Returns:
            [B, H] pooled representation
        """
        num_active_loops = len(loop_layer_hidden_states)
        logits = self._get_active_logits(num_active_loops)
        flat_weights = torch.softmax(logits.reshape(-1).float(), dim=0)
        weights = flat_weights.reshape(num_active_loops, self.num_layers)

        content_mask, fallback_batches, fallback_examples = self._build_content_mask(
            input_ids,
            attention_mask,
        )
        self.content_mean_fallback_batches += fallback_batches
        self.content_mean_fallback_examples += fallback_examples

        pooled = torch.zeros_like(loop_layer_hidden_states[0][0][:, 0])
        for loop_idx, per_layer in enumerate(loop_layer_hidden_states):
            for layer_idx, h in enumerate(per_layer):
                normed = self._final_norm_ref(h)
                mean = _masked_mean(normed, content_mask)
                w = weights[loop_idx, layer_idx].to(dtype=mean.dtype, device=mean.device)
                pooled = pooled + mean * w

        return pooled

    def get_mix_stats(self, num_loops: Optional[int] = None) -> Dict[str, object]:
        """Return current mixing weights for diagnostics."""
        n = num_loops if num_loops and num_loops > 0 else self.num_loops
        logits = self._get_active_logits(n)
        flat_weights = torch.softmax(logits.reshape(-1).float(), dim=0)
        weights = flat_weights.reshape(logits.shape)
        entropy = -(flat_weights * flat_weights.clamp_min(1e-12).log()).sum().item()
        return {
            "loop_layer_mix_active_loops": n,
            "loop_layer_mix_weights": weights.detach().cpu().tolist(),
            "loop_layer_mix_logits": logits.detach().cpu().tolist(),
            "loop_layer_mix_max_abs_logit": float(logits.detach().abs().max().item()),
            "loop_layer_mix_entropy": float(entropy),
            "content_mean_fallback_batches": int(self.content_mean_fallback_batches),
            "content_mean_fallback_examples": int(self.content_mean_fallback_examples),
        }


class EcoRNALoopLayerAttnContentPooler(nn.Module):
    """
    Sample-dependent content-only pooler over all active (loop, layer) cells.

    Each active cell contributes:
      1. final_norm(hidden_state)
      2. content-token masked mean (special tokens excluded)
      3. sample-dependent score via a shared linear scorer + loop/layer bias

    Weights are stabilized with a temperature-scaled softmax and a uniform floor:
        a = softmax(score / temperature)
        w = (1 - uniform_floor) * a + uniform_floor / num_cells
    """

    def __init__(
        self,
        config: EcoRNAConfig,
        final_norm: nn.Module,
        temperature: float = 2.0,
        uniform_floor: float = 0.10,
        exclude_special_tokens: bool = True,
    ):
        super().__init__()
        self.num_loops = config.num_loops
        self.num_layers = config.num_hidden_layers
        self.temperature = float(temperature)
        self.uniform_floor = float(uniform_floor)
        self.exclude_special_tokens = exclude_special_tokens
        self.special_token_ids = _collect_special_token_ids(config)
        self.__dict__["_final_norm_ref"] = final_norm

        self.scorer = nn.Linear(config.hidden_size, 1, bias=False)
        self.loop_bias = nn.Parameter(torch.zeros(config.num_loops))
        self.layer_bias = nn.Parameter(torch.zeros(config.num_hidden_layers))

        self.content_mean_fallback_batches = 0
        self.content_mean_fallback_examples = 0
        self.reset_stats()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.scorer.weight.zero_()
            self.loop_bias.zero_()
            self.layer_bias.zero_()

    def reset_stats(self) -> None:
        self._stats_num_examples = 0
        self._stats_active_loops = 0
        self._stats_sum_weights: Optional[torch.Tensor] = None
        self._stats_sum_sq_weights: Optional[torch.Tensor] = None
        self._stats_sum_entropy = 0.0
        self._stats_sum_max_weight = 0.0
        self.content_mean_fallback_batches = 0
        self.content_mean_fallback_examples = 0

    def _get_active_loop_bias(self, num_loops: int) -> torch.Tensor:
        if num_loops <= self.num_loops:
            return self.loop_bias[:num_loops]
        extra = self.loop_bias[-1:].expand(num_loops - self.num_loops)
        return torch.cat([self.loop_bias, extra], dim=0)

    def _build_content_mask(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], int, int]:
        return _build_sequence_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            special_token_ids=self.special_token_ids,
            exclude_special_tokens=self.exclude_special_tokens,
        )

    def _record_stats(self, weights: torch.Tensor, num_loops: int) -> None:
        weights_cpu = weights.detach().float().cpu()
        batch_size, num_cells = weights_cpu.shape
        if self._stats_sum_weights is None:
            self._stats_sum_weights = torch.zeros(num_cells, dtype=torch.float64)
            self._stats_sum_sq_weights = torch.zeros(num_cells, dtype=torch.float64)
            self._stats_active_loops = num_loops
        elif self._stats_sum_weights.numel() != num_cells:
            raise ValueError(
                f"Observed inconsistent active cell counts in loop-layer attention stats: "
                f"{self._stats_sum_weights.numel()} vs {num_cells}"
            )

        assert self._stats_sum_sq_weights is not None
        self._stats_sum_weights += weights_cpu.sum(dim=0, dtype=torch.float64)
        self._stats_sum_sq_weights += (weights_cpu ** 2).sum(dim=0, dtype=torch.float64)
        entropy = -(weights_cpu * weights_cpu.clamp_min(1e-12).log()).sum(dim=-1)
        self._stats_sum_entropy += float(entropy.sum().item())
        self._stats_sum_max_weight += float(weights_cpu.max(dim=-1).values.sum().item())
        self._stats_num_examples += int(batch_size)

    def forward(
        self,
        loop_layer_hidden_states: Tuple[Tuple[torch.Tensor, ...], ...],
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_active_loops = len(loop_layer_hidden_states)
        content_mask, fallback_batches, fallback_examples = self._build_content_mask(
            input_ids,
            attention_mask,
        )
        self.content_mean_fallback_batches += fallback_batches
        self.content_mean_fallback_examples += fallback_examples

        cell_vectors = []
        loop_indices = []
        layer_indices = []
        for loop_idx, per_layer in enumerate(loop_layer_hidden_states):
            for layer_idx, hidden_state in enumerate(per_layer):
                normed = self._final_norm_ref(hidden_state)
                cell_vectors.append(_masked_mean(normed, content_mask))
                loop_indices.append(loop_idx)
                layer_indices.append(layer_idx)

        cells = torch.stack(cell_vectors, dim=1)  # [B, N, H]
        scores = self.scorer(cells).squeeze(-1).float()
        loop_bias = self._get_active_loop_bias(num_active_loops)
        cell_loop_bias = loop_bias[torch.tensor(loop_indices, device=loop_bias.device)]
        cell_layer_bias = self.layer_bias[torch.tensor(layer_indices, device=self.layer_bias.device)]
        scores = scores + cell_loop_bias.to(scores.dtype) + cell_layer_bias.to(scores.dtype)
        attn = torch.softmax(scores / self.temperature, dim=-1)
        num_cells = attn.shape[-1]
        weights = (1.0 - self.uniform_floor) * attn + (self.uniform_floor / float(num_cells))
        self._record_stats(weights, num_active_loops)
        pooled = (cells * weights.to(dtype=cells.dtype).unsqueeze(-1)).sum(dim=1)
        return pooled

    def get_runtime_stats(self, num_loops: Optional[int] = None) -> Dict[str, object]:
        active_loops = int(num_loops) if num_loops else self.num_loops
        num_cells = active_loops * self.num_layers
        if self._stats_num_examples == 0:
            uniform = torch.full(
                (active_loops, self.num_layers),
                fill_value=1.0 / float(num_cells),
                dtype=torch.float32,
            )
            entropy = float(-(uniform.reshape(-1) * uniform.reshape(-1).log()).sum().item())
            return {
                "pooler_active_loops": active_loops,
                "pooler_mean_weights": uniform.tolist(),
                "pooler_weight_std": torch.zeros_like(uniform).tolist(),
                "pooler_mean_max_weight": float(1.0 / float(num_cells)),
                "pooler_effective_cells": float(num_cells),
                "pooler_mean_entropy": entropy,
                "pooler_temperature": self.temperature,
                "pooler_uniform_floor": self.uniform_floor,
                "content_mean_fallback_batches": int(self.content_mean_fallback_batches),
                "content_mean_fallback_examples": int(self.content_mean_fallback_examples),
            }

        assert self._stats_sum_weights is not None
        assert self._stats_sum_sq_weights is not None
        mean_weights = (self._stats_sum_weights / self._stats_num_examples).reshape(
            self._stats_active_loops, self.num_layers
        )
        mean_sq_weights = (self._stats_sum_sq_weights / self._stats_num_examples).reshape(
            self._stats_active_loops, self.num_layers
        )
        weight_std = (mean_sq_weights - mean_weights ** 2).clamp(min=0.0).sqrt()
        mean_entropy = self._stats_sum_entropy / float(self._stats_num_examples)
        return {
            "pooler_active_loops": int(self._stats_active_loops),
            "pooler_mean_weights": mean_weights.float().tolist(),
            "pooler_weight_std": weight_std.float().tolist(),
            "pooler_mean_max_weight": float(self._stats_sum_max_weight / float(self._stats_num_examples)),
            "pooler_effective_cells": float(torch.exp(torch.tensor(mean_entropy)).item()),
            "pooler_mean_entropy": float(mean_entropy),
            "pooler_temperature": self.temperature,
            "pooler_uniform_floor": self.uniform_floor,
            "content_mean_fallback_batches": int(self.content_mean_fallback_batches),
            "content_mean_fallback_examples": int(self.content_mean_fallback_examples),
        }


class EcoRNAGlobalWeightedContentPooler(nn.Module):
    """
    Globally shared content-only weighted average over all active (loop, layer) cells.

    This pooler does not use sample-dependent attention. Instead, it learns a single
    set of shared weights, either factorized across loop/layer or directly per cell.
    The logits are centered and bounded before softmax to keep the distribution
    interpretable and numerically stable.
    """

    def __init__(
        self,
        config: EcoRNAConfig,
        final_norm: nn.Module,
        mode: str,
        temperature: float = 2.0,
        uniform_floor: float = 0.05,
        logit_cap: float = 4.0,
        exclude_special_tokens: bool = True,
    ):
        super().__init__()
        if mode not in {"factorized", "cell"}:
            raise ValueError(f"Unsupported weighted content pooler mode: {mode}")
        self.mode = mode
        self.num_loops = config.num_loops
        self.num_layers = config.num_hidden_layers
        self.temperature = float(temperature)
        self.uniform_floor = float(uniform_floor)
        self.logit_cap = float(logit_cap)
        self.exclude_special_tokens = exclude_special_tokens
        self.special_token_ids = _collect_special_token_ids(config)
        self.__dict__["_final_norm_ref"] = final_norm

        if self.mode == "factorized":
            self.loop_logits = nn.Parameter(torch.zeros(config.num_loops))
            self.layer_logits = nn.Parameter(torch.zeros(config.num_hidden_layers))
        else:
            self.cell_logits = nn.Parameter(torch.zeros(config.num_loops, config.num_hidden_layers))

        self.content_mean_fallback_batches = 0
        self.content_mean_fallback_examples = 0
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.mode == "factorized":
                self.loop_logits.zero_()
                self.layer_logits.zero_()
            else:
                self.cell_logits.zero_()

    def reset_stats(self) -> None:
        self.content_mean_fallback_batches = 0
        self.content_mean_fallback_examples = 0

    def _extend_active_loop_logits(self, logits: torch.Tensor, num_loops: int) -> torch.Tensor:
        if num_loops <= logits.shape[0]:
            return logits[:num_loops]
        extra = logits[-1:].expand(num_loops - logits.shape[0], *logits.shape[1:])
        return torch.cat([logits, extra], dim=0)

    def _build_content_mask(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], int, int]:
        return _build_sequence_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            special_token_ids=self.special_token_ids,
            exclude_special_tokens=self.exclude_special_tokens,
        )

    def _compute_logits_and_weights(
        self,
        num_active_loops: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mode == "factorized":
            active_loop_logits = self._extend_active_loop_logits(self.loop_logits, num_active_loops)
            centered_loop_logits = active_loop_logits - active_loop_logits.mean()
            centered_layer_logits = self.layer_logits - self.layer_logits.mean()
            raw_logits = centered_loop_logits[:, None] + centered_layer_logits[None, :]
        else:
            active_cell_logits = self._extend_active_loop_logits(self.cell_logits, num_active_loops)
            raw_logits = active_cell_logits - active_cell_logits.mean()

        bounded_logits = self.logit_cap * torch.tanh(raw_logits)
        flat_attn = torch.softmax(bounded_logits.reshape(-1).float() / self.temperature, dim=0)
        num_cells = flat_attn.numel()
        flat_weights = (1.0 - self.uniform_floor) * flat_attn + (self.uniform_floor / float(num_cells))
        weights = flat_weights.reshape(num_active_loops, self.num_layers)
        return raw_logits, bounded_logits, weights

    def forward(
        self,
        loop_layer_hidden_states: Tuple[Tuple[torch.Tensor, ...], ...],
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_active_loops = len(loop_layer_hidden_states)
        _, _, weights = self._compute_logits_and_weights(num_active_loops)
        content_mask, fallback_batches, fallback_examples = self._build_content_mask(
            input_ids,
            attention_mask,
        )
        self.content_mean_fallback_batches += fallback_batches
        self.content_mean_fallback_examples += fallback_examples

        pooled = torch.zeros_like(loop_layer_hidden_states[0][0][:, 0])
        for loop_idx, per_layer in enumerate(loop_layer_hidden_states):
            for layer_idx, hidden_state in enumerate(per_layer):
                normed = self._final_norm_ref(hidden_state)
                mean = _masked_mean(normed, content_mask)
                weight = weights[loop_idx, layer_idx].to(dtype=mean.dtype, device=mean.device)
                pooled = pooled + mean * weight

        return pooled

    def get_runtime_stats(self, num_loops: Optional[int] = None) -> Dict[str, object]:
        active_loops = int(num_loops) if num_loops else self.num_loops
        raw_logits, bounded_logits, weights = self._compute_logits_and_weights(active_loops)
        flat_weights = weights.reshape(-1).float()
        entropy = float(-(flat_weights * flat_weights.clamp_min(1e-12).log()).sum().item())
        return {
            "pooler_mode": self.mode,
            "pooler_active_loops": active_loops,
            "pooler_mean_weights": weights.detach().cpu().tolist(),
            "pooler_weight_std": torch.zeros_like(weights).detach().cpu().tolist(),
            "pooler_mean_max_weight": float(flat_weights.max().item()),
            "pooler_effective_cells": float(torch.exp(torch.tensor(entropy)).item()),
            "pooler_mean_entropy": entropy,
            "pooler_temperature": self.temperature,
            "pooler_uniform_floor": self.uniform_floor,
            "pooler_logit_cap": self.logit_cap,
            "pooler_max_abs_raw_logit": float(raw_logits.detach().abs().max().item()),
            "pooler_max_abs_bounded_logit": float(bounded_logits.detach().abs().max().item()),
            "pooler_raw_logits": raw_logits.detach().cpu().tolist(),
            "pooler_bounded_logits": bounded_logits.detach().cpu().tolist(),
            "content_mean_fallback_batches": int(self.content_mean_fallback_batches),
            "content_mean_fallback_examples": int(self.content_mean_fallback_examples),
        }


# ---------------------------------------------------------------------------
# Sequence Classification
# ---------------------------------------------------------------------------

def _validate_liger(config: EcoRNAConfig, context: str = ""):
    if config.use_liger_kernel and not LIGER_KERNEL_AVAILABLE:
        raise RuntimeError(
            f"CRITICAL{' (' + context + ')' if context else ''}: "
            "Checkpoint trained with use_liger_kernel=True but liger_kernel not installed. "
            "MLP weights will be randomly initialized. Fix: pip install liger-kernel"
        )


def _remap_legacy_state_dict(state_dict, prefix, *args, **kwargs):
    """Remap old checkpoint keys to new structure."""
    old = prefix + "loop_layer_mix_logits"
    new = prefix + "pooler.loop_layer_mix_logits"
    if old in state_dict and new not in state_dict:
        state_dict[new] = state_dict.pop(old)


class EcoRNAForSequenceClassification(EcoRNAPreTrainedModel):
    """
    EcoRNA with a sequence classification head.
    Supports pooling strategies:
    cls, cls_tanh, mean, cls_mean_concat, loop_mean_cls,
    content_mean, cls_ln, loop_mean_content, layer_weighted,
    loop_layer_attn_content, weighted_layer_content, weighted_cell_content,
    fixed_cell_content, and fixed_cell_cls.
    """

    _keys_to_ignore_on_load_unexpected = [
        "pooler.dense.weight", "pooler.dense.bias", "pooler.activation",
        "pooler.final_norm.weight",
    ]

    def __init__(self, config: EcoRNAConfig, **kwargs):
        num_labels = kwargs.pop("num_labels", getattr(config, "num_labels", 2))
        problem_type = kwargs.pop("problem_type", getattr(config, "problem_type", None))
        pooling_strategy = kwargs.pop(
            "pooling_strategy", getattr(config, "pooling_strategy", "cls_tanh")
        )
        pooling_cells = kwargs.pop("pooling_cells", getattr(config, "pooling_cells", ""))
        num_loops = kwargs.pop("num_loops", getattr(config, "num_loops", None))
        exclude_special_tokens = kwargs.pop("exclude_special_tokens", True)
        kwargs.pop("token_type", None)  # accepted but unused

        _validate_liger(config)
        super().__init__(config)

        self.num_labels = num_labels
        self.config.num_labels = num_labels
        self.config.problem_type = problem_type
        self.config.use_fused_cross_entropy = False

        self.pooling_strategy = _resolve_strategy(pooling_strategy)
        self.pooling_cells = _parse_pooling_cells(pooling_cells)
        self.infer_num_loops = num_loops
        self.exclude_special_tokens = exclude_special_tokens
        self.special_token_ids = _collect_special_token_ids(config)
        self._content_mean_fallback_batches = 0
        self._content_mean_fallback_examples = 0

        if self.pooling_strategy in {"fixed_cell_content", "fixed_cell_cls"} and not self.pooling_cells:
            raise ValueError(
                f"Pooling strategy {self.pooling_strategy!r} requires non-empty pooling_cells."
            )

        self.ecorna = EcoRNAModel(config)

        classifier_hidden_size = config.hidden_size
        self.cls_norm = None

        if self.pooling_strategy == "layer_weighted":
            self.pooler = EcoRNALayerWeightedPooler(
                config,
                final_norm=self.ecorna.final_norm,
                exclude_special_tokens=exclude_special_tokens,
            )
        elif self.pooling_strategy == "weighted_layer_content":
            self.pooler = EcoRNAGlobalWeightedContentPooler(
                config,
                final_norm=self.ecorna.final_norm,
                mode="factorized",
                exclude_special_tokens=exclude_special_tokens,
            )
        elif self.pooling_strategy == "weighted_cell_content":
            self.pooler = EcoRNAGlobalWeightedContentPooler(
                config,
                final_norm=self.ecorna.final_norm,
                mode="cell",
                exclude_special_tokens=exclude_special_tokens,
            )
        elif self.pooling_strategy == "loop_layer_attn_content":
            self.pooler = EcoRNALoopLayerAttnContentPooler(
                config,
                final_norm=self.ecorna.final_norm,
                exclude_special_tokens=exclude_special_tokens,
            )
        elif self.pooling_strategy == "cls_tanh":
            self.pooler = EcoRNAClsTanhPooler(config)
        else:
            self.pooler = None

        if self.pooling_strategy == "cls_mean_concat":
            classifier_hidden_size = config.hidden_size * 2
        elif self.pooling_strategy == "cls_ln":
            self.cls_norm = nn.LayerNorm(config.hidden_size)

        self.classifier = nn.Linear(classifier_hidden_size, num_labels)

        # State dict compatibility
        self._register_load_state_dict_pre_hook(_remap_legacy_state_dict)

        self.post_init()
        self._reset_pooling_parameters()

    @property
    def loop_layer_mix_logits(self) -> Optional[nn.Parameter]:
        if self.pooler is not None and hasattr(self.pooler, "loop_layer_mix_logits"):
            return self.pooler.loop_layer_mix_logits
        return None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        config = kwargs.get("config", None)
        if config is None:
            config = EcoRNAConfig.from_pretrained(pretrained_model_name_or_path)
        source_architectures = tuple(getattr(config, "architectures", []) or [])
        _validate_liger(config, pretrained_model_name_or_path)
        model = super().from_pretrained(
            pretrained_model_name_or_path, *args, config=config, **kwargs
        )
        if "EcoRNAForMaskedLM" in source_architectures:
            model._reset_downstream_readout_parameters()
        return model

    def get_loop_layer_mix_stats(self, num_loops: Optional[int] = None) -> Dict[str, object]:
        return self.get_runtime_diagnostics(num_loops)

    def _reset_pooling_parameters(self) -> None:
        if self.pooling_strategy == "layer_weighted" and self.pooler is not None:
            with torch.no_grad():
                self.pooler.loop_layer_mix_logits.zero_()
        elif self.pooling_strategy in {"weighted_layer_content", "weighted_cell_content"} and self.pooler is not None:
            self.pooler.reset_parameters()
        elif self.pooling_strategy == "loop_layer_attn_content" and self.pooler is not None:
            self.pooler.reset_parameters()
        elif self.pooling_strategy == "cls_tanh" and self.pooler is not None:
            with torch.no_grad():
                nn.init.normal_(self.pooler.dense.weight, mean=0.0, std=0.02)
                nn.init.zeros_(self.pooler.dense.bias)
        elif self.pooling_strategy == "cls_ln" and self.cls_norm is not None:
            with torch.no_grad():
                nn.init.ones_(self.cls_norm.weight)
                nn.init.zeros_(self.cls_norm.bias)

    def _reset_downstream_readout_parameters(self) -> None:
        self._reset_pooling_parameters()
        with torch.no_grad():
            nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
            if self.classifier.bias is not None:
                nn.init.zeros_(self.classifier.bias)

    def get_runtime_diagnostics(self, num_loops: Optional[int] = None) -> Dict[str, object]:
        stats = {
            "content_mean_fallback_batches": int(self._content_mean_fallback_batches),
            "content_mean_fallback_examples": int(self._content_mean_fallback_examples),
        }
        if self.pooling_strategy in {"fixed_cell_content", "fixed_cell_cls"}:
            stats.update(
                {
                    "fixed_cell_pooling_strategy": self.pooling_strategy,
                    "fixed_cell_pooling_cells": _format_pooling_cells(self.pooling_cells),
                }
            )
        if self.pooler is not None:
            n = num_loops if num_loops else self.infer_num_loops
            if hasattr(self.pooler, "get_runtime_stats"):
                stats.update(self.pooler.get_runtime_stats(n))
            elif hasattr(self.pooler, "get_mix_stats"):
                stats.update(self.pooler.get_mix_stats(n))
        return stats

    def reset_runtime_diagnostics(self) -> None:
        self._content_mean_fallback_batches = 0
        self._content_mean_fallback_examples = 0
        if self.pooler is not None and hasattr(self.pooler, "reset_stats"):
            self.pooler.reset_stats()

    def _build_content_mask(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        return self._content_mask(input_ids, attention_mask)

    def _masked_mean(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return _masked_mean(hidden_states, mask)

    def _content_mask(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        mask, fallback_batches, fallback_examples = _build_sequence_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            special_token_ids=self.special_token_ids,
            exclude_special_tokens=self.exclude_special_tokens,
        )
        self._content_mean_fallback_batches += fallback_batches
        self._content_mean_fallback_examples += fallback_examples
        return mask

    def _content_mean(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return _masked_mean(hidden_states, self._content_mask(input_ids, attention_mask))

    def _loop_mean_cls(self, hidden_states: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        normed_cls = [self.ecorna.final_norm(loop_state)[:, 0] for loop_state in hidden_states]
        return torch.stack(normed_cls, dim=0).mean(dim=0)

    def _loop_mean_content(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        content_mask = self._content_mask(input_ids, attention_mask)
        normed_means = [
            _masked_mean(self.ecorna.final_norm(loop_state), content_mask)
            for loop_state in hidden_states
        ]
        return torch.stack(normed_means, dim=0).mean(dim=0)

    def _pool_fixed_cells(
        self,
        loop_layer_hidden_states: Tuple[Tuple[torch.Tensor, ...], ...],
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        mode: str,
    ) -> torch.Tensor:
        num_active_loops = len(loop_layer_hidden_states)
        selected: List[torch.Tensor] = []
        content_mask = None
        if mode == "content":
            content_mask = self._content_mask(input_ids, attention_mask)

        for loop_idx, layer_idx in self.pooling_cells:
            if loop_idx > num_active_loops:
                continue
            per_layer = loop_layer_hidden_states[loop_idx - 1]
            if layer_idx > len(per_layer):
                continue
            hidden_state = self.ecorna.final_norm(per_layer[layer_idx - 1])
            if mode == "content":
                selected.append(_masked_mean(hidden_state, content_mask))
            elif mode == "cls":
                selected.append(hidden_state[:, 0])
            else:
                raise ValueError(f"Unsupported fixed-cell pooling mode: {mode}")

        if not selected:
            raise ValueError(
                f"All pooling cells { _format_pooling_cells(self.pooling_cells) } are outside "
                f"the active loop/layer range for num_active_loops={num_active_loops}."
            )

        return torch.stack(selected, dim=0).mean(dim=0)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        need_loop_states = self.pooling_strategy in {"loop_mean_cls", "loop_mean_content"}
        need_loop_layer = self.pooling_strategy in {
            "layer_weighted",
            "loop_layer_attn_content",
            "weighted_layer_content",
            "weighted_cell_content",
            "fixed_cell_content",
            "fixed_cell_cls",
        }

        outputs = self.ecorna(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_loops=self.infer_num_loops,
            output_hidden_states=need_loop_states or bool(output_hidden_states),
            output_loop_layer_hidden_states=need_loop_layer,
        )

        sequence_output = outputs.last_hidden_state

        # Pool
        if self.pooling_strategy == "layer_weighted":
            pooled = self.pooler(
                outputs.loop_layer_hidden_states,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        elif self.pooling_strategy == "weighted_layer_content":
            pooled = self.pooler(
                outputs.loop_layer_hidden_states,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        elif self.pooling_strategy == "weighted_cell_content":
            pooled = self.pooler(
                outputs.loop_layer_hidden_states,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        elif self.pooling_strategy == "loop_layer_attn_content":
            pooled = self.pooler(
                outputs.loop_layer_hidden_states,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        elif self.pooling_strategy == "fixed_cell_content":
            pooled = self._pool_fixed_cells(
                outputs.loop_layer_hidden_states,
                input_ids=input_ids,
                attention_mask=attention_mask,
                mode="content",
            )
        elif self.pooling_strategy == "fixed_cell_cls":
            pooled = self._pool_fixed_cells(
                outputs.loop_layer_hidden_states,
                input_ids=input_ids,
                attention_mask=attention_mask,
                mode="cls",
            )
        elif self.pooling_strategy == "cls":
            pooled = sequence_output[:, 0]
        elif self.pooling_strategy == "mean":
            pooled = _masked_mean(sequence_output, attention_mask.bool() if attention_mask is not None else None)
        elif self.pooling_strategy == "cls_tanh":
            pooled = self.pooler(sequence_output)
        elif self.pooling_strategy == "cls_mean_concat":
            pooled = torch.cat(
                [
                    sequence_output[:, 0],
                    _masked_mean(sequence_output, attention_mask.bool() if attention_mask is not None else None),
                ],
                dim=-1,
            )
        elif self.pooling_strategy == "content_mean":
            pooled = self._content_mean(sequence_output, input_ids, attention_mask)
        elif self.pooling_strategy == "cls_ln":
            pooled = self.cls_norm(sequence_output[:, 0])
        elif self.pooling_strategy == "loop_mean_cls":
            pooled = self._loop_mean_cls(outputs.hidden_states)
        elif self.pooling_strategy == "loop_mean_content":
            pooled = self._loop_mean_content(outputs.hidden_states, input_ids, attention_mask)
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling_strategy}")

        logits = self.classifier(pooled)

        # Loss
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and labels.dtype in (torch.long, torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss = MSELoss()(
                    logits.squeeze() if self.num_labels == 1 else logits,
                    labels.squeeze() if self.num_labels == 1 else labels,
                )
            elif self.config.problem_type == "single_label_classification":
                loss = CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = BCEWithLogitsLoss()(logits, labels)

        if not return_dict:
            out = (logits,)
            if output_hidden_states:
                out = out + (outputs.hidden_states,)
            return ((loss,) + out) if loss is not None else out

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=None,
        )


# ---------------------------------------------------------------------------
# Token-Level (Nucleotide) Prediction
# ---------------------------------------------------------------------------

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y) + self.eps)


class MCRMSELoss(nn.Module):
    def __init__(self, num_scored=3):
        super().__init__()
        self.rmse = RMSELoss()
        self.num_scored = num_scored

    def forward(self, yhat, y):
        score = 0
        for i in range(self.num_scored):
            score += self.rmse(yhat[:, :, i], y[:, :, i]) / self.num_scored
        return score


class EcoRNAForNucleotideLevel(EcoRNAPreTrainedModel):
    """
    EcoRNA with a token classification head for nucleotide-level prediction.
    Used for tasks like: Degradation, SpliceAI.
    """

    def __init__(self, config: EcoRNAConfig, tokenizer=None, **kwargs):
        num_labels = kwargs.pop("num_labels", getattr(config, "num_labels", 2))
        problem_type = kwargs.pop("problem_type", getattr(config, "problem_type", None))
        kwargs.pop("token_type", None)

        _validate_liger(config)
        super().__init__(config)

        self.num_labels = num_labels
        self.config.num_labels = num_labels
        self.config.problem_type = problem_type
        self.config.use_fused_cross_entropy = False
        self.tokenizer = tokenizer

        self.ecorna = EcoRNAModel(config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        config = kwargs.get("config", None)
        if config is None:
            config = EcoRNAConfig.from_pretrained(pretrained_model_name_or_path)
        source_architectures = tuple(getattr(config, "architectures", []) or [])
        _validate_liger(config, pretrained_model_name_or_path)
        model = super().from_pretrained(
            pretrained_model_name_or_path, *args, config=config, **kwargs
        )
        if "EcoRNAForMaskedLM" in source_architectures:
            model._reset_downstream_readout_parameters()
        return model

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        weight_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.ecorna(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs.last_hidden_state
        if weight_mask is not None:
            sequence_output = sequence_output * weight_mask.unsqueeze(-1)

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            logits = logits[:, 1:1 + labels.size(1), :]

            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and labels.dtype in (torch.long, torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss = MCRMSELoss(num_scored=self.num_labels)(
                    logits.squeeze() if self.num_labels == 1 else logits,
                    labels.squeeze() if self.num_labels == 1 else labels,
                )
            elif self.config.problem_type == "single_label_classification":
                loss = CrossEntropyLoss()(
                    logits.reshape(-1, self.num_labels), labels.reshape(-1).long()
                )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=None,
        )
