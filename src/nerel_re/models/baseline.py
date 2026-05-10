"""Baseline RE model: standard BERT encoder with plain entity markers.

Architecture:
    [CLS] … [E1] subj [/E1] … [E2] obj [/E2] … [SEP]

The [CLS] token representation is passed through a classification head.
This reproduces the OpenNRE-style approach used as the NEREL baseline
(Loukachevitch et al., RANLP 2021).
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class BaselineREModel(nn.Module):
    """BERT-style relation classifier using the [CLS] representation.

    Args:
        model_name: HuggingFace model name or path (e.g. ``"ai-forever/ruBert-base"``).
        num_labels: Number of relation type classes (including ``no_relation``).
        dropout: Dropout probability applied before the classification head.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        tokenizer_vocab_size: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        """Initialise encoder and classification head."""
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        if tokenizer_vocab_size is not None:
            self.encoder.resize_token_embeddings(tokenizer_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

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
            attention_mask: Attention mask tensor of shape ``(batch, seq_len)``.
            token_type_ids: Optional token type IDs (not used by RoBERTa-based models).
            labels: Optional ground-truth class indices for loss computation.

        Returns:
            Dict with ``"logits"`` and, if ``labels`` is provided, ``"loss"``.
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_repr = self.dropout(outputs.last_hidden_state[:, 0, :])
        logits = self.classifier(cls_repr)

        result: dict[str, torch.Tensor] = {'logits': logits}
        if labels is not None:
            result['loss'] = nn.CrossEntropyLoss()(logits, labels)
        return result
