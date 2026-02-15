"""
EcoRNA model wrappers for BEACON benchmark downstream tasks.
Wraps the base EcoRNA model for sequence classification and token-level tasks.

IMPORTANT: EcoRNA checkpoints trained with Liger Kernel use different parameter names
for MLP layers (gate_proj/up_proj/down_proj vs gate/up/down). This module ensures
correct loading and provides options for inference precision.
"""

from __future__ import annotations
from typing import Optional, Tuple, Union
import logging

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput

import sys
from pathlib import Path

# Add ecorna to path
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecorna import EcoRNAConfig, EcoRNAModel, EcoRNAPreTrainedModel
from ecorna.modeling_ecorna import LIGER_KERNEL_AVAILABLE, FLASH_ATTN_AVAILABLE

logger = logging.getLogger(__name__)


def log_inference_config(config: EcoRNAConfig):
    """Log the inference configuration for debugging."""
    logger.info("=" * 50)
    logger.info("EcoRNA Inference Configuration:")
    logger.info(f"  use_liger_kernel: {config.use_liger_kernel}")
    logger.info(f"  use_liger_rms_norm: {config.use_liger_rms_norm}")
    logger.info(f"  LIGER_KERNEL_AVAILABLE: {LIGER_KERNEL_AVAILABLE}")
    logger.info(f"  FLASH_ATTN_AVAILABLE: {FLASH_ATTN_AVAILABLE}")
    logger.info(f"  hidden_size: {config.hidden_size}")
    logger.info(f"  num_hidden_layers: {config.num_hidden_layers}")
    logger.info(f"  num_loops: {config.num_loops}")
    logger.info("=" * 50)


class EcoRNAPooler(nn.Module):
    """Pooler that takes the [CLS] token representation."""

    def __init__(self, config: EcoRNAConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Take [CLS] token (first token)
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class EcoRNAForSequenceClassification(EcoRNAPreTrainedModel):
    """
    EcoRNA Model with a sequence classification head.
    Used for tasks like: NoncodingRNAFamily, Isoform, MeanRibosomeLoading,
    CRISPROnTarget, Modification, ProgrammableRNASwitches.

    IMPORTANT Notes on Precision:
    - Liger Kernel: If checkpoint was trained with Liger, inference MUST use Liger
      to ensure correct weight loading (different param names: gate_proj vs gate)
    - Flash Attention: Mathematically equivalent to standard attention, safe to use
    - For benchmark fairness, we use the same inference settings as training
    """

    def __init__(self, config: EcoRNAConfig, **kwargs):
        # Handle num_labels passed as kwarg
        num_labels = kwargs.pop("num_labels", getattr(config, "num_labels", 2))
        problem_type = kwargs.pop("problem_type", getattr(config, "problem_type", None))
        token_type = kwargs.pop("token_type", getattr(config, "token_type", "single"))
        pooling_strategy = kwargs.pop(
            "pooling_strategy", getattr(config, "pooling_strategy", "cls_tanh")
        )
        num_loops = kwargs.pop("num_loops", getattr(config, "num_loops", None))

        # Validate Liger Kernel availability
        if config.use_liger_kernel and not LIGER_KERNEL_AVAILABLE:
            raise RuntimeError(
                "CRITICAL: Model checkpoint was trained with use_liger_kernel=True, "
                "but liger_kernel is not installed. This will cause MLP weights to be "
                "randomly initialized due to parameter name mismatch "
                "(gate_proj/up_proj/down_proj vs gate/up/down).\n"
                "Solution: pip install liger-kernel"
            )

        super().__init__(config)

        self.num_labels = num_labels
        self.config.num_labels = num_labels
        self.config.problem_type = problem_type
        self.config.token_type = token_type
        self.config.pooling_strategy = pooling_strategy
        self.config.infer_num_loops = num_loops

        # Disable fused cross entropy for inference (not needed, only affects training)
        self.config.use_fused_cross_entropy = False

        self.ecorna = EcoRNAModel(config)
        self.pooler = EcoRNAPooler(config)
        self.pooling_strategy = pooling_strategy
        self.infer_num_loops = num_loops
        if pooling_strategy in ["cls_mean_concat"]:
            classifier_in = config.hidden_size * 2
        else:
            classifier_in = config.hidden_size
        self.classifier = nn.Linear(classifier_in, num_labels)
        self.loop_weights = None
        if pooling_strategy == "loop_weighted_cls":
            self.loop_weights = nn.Parameter(torch.zeros(config.num_loops))

        # Log configuration
        log_inference_config(config)

        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Override from_pretrained to ensure correct Liger Kernel handling.
        """
        # Load config first to check Liger settings
        config = kwargs.get("config", None)
        if config is None:
            config = EcoRNAConfig.from_pretrained(pretrained_model_name_or_path)

        # Validate before loading
        if config.use_liger_kernel and not LIGER_KERNEL_AVAILABLE:
            raise RuntimeError(
                f"CRITICAL: Checkpoint at {pretrained_model_name_or_path} was trained with "
                "use_liger_kernel=True, but liger_kernel is not installed.\n"
                "This will cause MLP weights to be randomly initialized!\n"
                "Solution: pip install liger-kernel"
            )

        logger.info(f"Loading EcoRNA from {pretrained_model_name_or_path}")
        logger.info(f"  Config use_liger_kernel={config.use_liger_kernel}, "
                   f"LIGER_KERNEL_AVAILABLE={LIGER_KERNEL_AVAILABLE}")

        return super().from_pretrained(pretrained_model_name_or_path, *args, config=config, **kwargs)

    def _masked_mean(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        hidden_sum = (hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return hidden_sum / denom

    def _pool_features(
        self,
        sequence_output: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        all_hidden_states: Optional[Tuple[torch.Tensor, ...]],
    ) -> torch.Tensor:
        strategy = self.pooling_strategy
        if strategy == "cls_tanh":
            return self.pooler(sequence_output)
        if strategy == "cls":
            return sequence_output[:, 0]
        if strategy == "mean":
            return self._masked_mean(sequence_output, attention_mask)
        if strategy == "cls_mean_concat":
            cls = sequence_output[:, 0]
            mean = self._masked_mean(sequence_output, attention_mask)
            return torch.cat([cls, mean], dim=-1)
        if strategy in ["loop_mean_cls", "loop_weighted_cls"]:
            if not all_hidden_states:
                raise ValueError(
                    f"Pooling strategy {strategy} requires output_hidden_states=True"
                )
            loop_cls = torch.stack([h[:, 0] for h in all_hidden_states], dim=1)  # (B, L, H)
            if strategy == "loop_mean_cls":
                return loop_cls.mean(dim=1)
            weights = torch.softmax(self.loop_weights[: loop_cls.shape[1]], dim=0)
            return (loop_cls * weights.view(1, -1, 1)).sum(dim=1)
        raise ValueError(f"Unknown pooling_strategy: {strategy}")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.ecorna(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_loops=self.infer_num_loops,
            output_hidden_states=(
                output_hidden_states
                or self.pooling_strategy in ["loop_mean_cls", "loop_weighted_cls"]
            ),
        )

        sequence_output = outputs.last_hidden_state
        pooled_output = self._pool_features(
            sequence_output=sequence_output,
            attention_mask=attention_mask,
            all_hidden_states=outputs.hidden_states,
        )
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class RMSELoss(nn.Module):
    """Root Mean Square Error Loss."""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class MCRMSELoss(nn.Module):
    """Multi-Column Root Mean Square Error Loss."""
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
    EcoRNA Model with a token classification head for nucleotide-level prediction.
    Used for tasks like: Degradation, SpliceAI.
    """

    def __init__(self, config: EcoRNAConfig, tokenizer=None, **kwargs):
        num_labels = kwargs.pop("num_labels", getattr(config, "num_labels", 2))
        problem_type = kwargs.pop("problem_type", getattr(config, "problem_type", None))
        token_type = kwargs.pop("token_type", getattr(config, "token_type", "single"))

        # Validate Liger Kernel availability
        if config.use_liger_kernel and not LIGER_KERNEL_AVAILABLE:
            raise RuntimeError(
                "CRITICAL: Model checkpoint was trained with use_liger_kernel=True, "
                "but liger_kernel is not installed. This will cause MLP weights to be "
                "randomly initialized due to parameter name mismatch.\n"
                "Solution: pip install liger-kernel"
            )

        super().__init__(config)

        self.num_labels = num_labels
        self.config.num_labels = num_labels
        self.config.problem_type = problem_type
        self.config.token_type = token_type
        self.config.use_fused_cross_entropy = False
        self.tokenizer = tokenizer

        self.ecorna = EcoRNAModel(config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        log_inference_config(config)
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Override from_pretrained to ensure correct Liger Kernel handling."""
        config = kwargs.get("config", None)
        if config is None:
            config = EcoRNAConfig.from_pretrained(pretrained_model_name_or_path)

        if config.use_liger_kernel and not LIGER_KERNEL_AVAILABLE:
            raise RuntimeError(
                f"CRITICAL: Checkpoint at {pretrained_model_name_or_path} was trained with "
                "use_liger_kernel=True, but liger_kernel is not installed.\n"
                "Solution: pip install liger-kernel"
            )

        return super().from_pretrained(pretrained_model_name_or_path, *args, config=config, **kwargs)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        weight_mask: Optional[torch.Tensor] = None,
        post_token_length: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, seq_len)` or `(batch_size, seq_len, num_labels)`, *optional*):
            Labels for computing the token classification/regression loss.
        weight_mask: Mask for valid positions
        post_token_length: Token lengths for BPE tokenizers (not used for single-char tokenizer)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.ecorna(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs.last_hidden_state

        # For single token type, use directly (EcoRNA uses single-char tokenization)
        if weight_mask is not None:
            sequence_output = sequence_output * weight_mask.unsqueeze(-1)

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # Exclude [CLS] and [SEP] tokens
            logits = logits[:, 1:1+labels.size(1), :]

            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MCRMSELoss(num_scored=self.num_labels)
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.reshape(-1, self.num_labels), labels.reshape(-1).long())

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=None,
        )
