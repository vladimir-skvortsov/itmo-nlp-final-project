"""Token classification model for NER.

Wraps HuggingFace ``AutoModelForTokenClassification`` to provide a consistent
``forward`` API that returns a plain dict with ``"logits"`` and optionally
``"loss"``.  Loading in ``torch.float32`` prevents the NaN issue observed with
DeBERTa-v3 in mixed-precision environments.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForTokenClassification


class NERModel(nn.Module):
    """BERT-style token classifier for named entity recognition.

    Args:
        model_name: HuggingFace model name or local path.
        num_labels: Number of BIO label classes (``2 * num_entity_types + 1``).
        dropout: Dropout probability applied by the encoder.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialise encoder and classification head."""
        super().__init__()
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
        # Load in float32 to prevent NaN from DeBERTa-v3 XSoftmax overflow.
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float32,
            ignore_mismatched_sizes=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run a forward pass.

        Args:
            input_ids: Token ID tensor of shape ``(batch, seq_len)``.
            attention_mask: Attention mask of shape ``(batch, seq_len)``.
            token_type_ids: Optional token type IDs (not used by DeBERTa).
            labels: Optional integer label tensor of shape ``(batch, seq_len)``.
                Positions with value ``-100`` are ignored in the loss.

        Returns:
            Dict with ``"logits"`` (shape ``(batch, seq_len, num_labels)``)
            and, when ``labels`` is provided, ``"loss"``.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        result: dict[str, torch.Tensor] = {'logits': outputs.logits}
        if outputs.loss is not None:
            result['loss'] = outputs.loss
        return result
