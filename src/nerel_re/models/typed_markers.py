"""Main model: DeBERTa-v3 with typed entity markers.

Architecture (Zhou et al., 2022):
    [CLS] … [e1_PER] subj [/e1_PER] … [e2_ORG] obj [/e2_ORG] … [SEP]

Entity representations are taken from the positions of the opening typed
marker tokens ([e1_TYPE] and [e2_TYPE]), concatenated, and passed through
a classification head.  This outperforms [CLS]-only and standard [E1]/[E2]
representations because entity type information is encoded at the input level.

Reference:
    Zhou et al. (2022) "An Improved Baseline for Sentence-level Relation
    Extraction." AACL-IJCNLP Short Papers.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from nerel_re.data.tokenizer_utils import OBJ_START_TMPL, SUBJ_START_TMPL


def _find_marker_position(
    input_ids: torch.Tensor,
    marker_id: int,
) -> torch.Tensor:
    """Return the position (index) of a special marker token in each sequence.

    If the token is not found (e.g. truncated away), falls back to position 0.

    Args:
        input_ids: Token ID tensor of shape ``(batch, seq_len)``.
        marker_id: Token ID of the marker to locate.

    Returns:
        Long tensor of shape ``(batch,)`` with per-example positions.
    """
    matches = (input_ids == marker_id).nonzero(as_tuple=False)
    positions = torch.zeros(input_ids.size(0), dtype=torch.long, device=input_ids.device)
    for row, col in matches:
        positions[row] = col
    return positions


class TypedMarkersREModel(nn.Module):
    """DeBERTa-v3 relation classifier using typed entity marker representations.

    The subject entity embedding is the hidden state at the ``[e1_TYPE]`` token;
    the object entity embedding is the hidden state at the ``[e2_TYPE]`` token.
    Both are concatenated and fed to a two-layer MLP classifier.

    Args:
        model_name: HuggingFace model name (e.g. ``"microsoft/mdeberta-v3-base"``).
        num_labels: Number of relation type classes (including ``no_relation``).
        entity_types: List of all entity type strings in the dataset.
        tokenizer_vocab_size: Vocabulary size **after** adding special tokens.
            The encoder's embedding layer is resized to this value.
        dropout: Dropout probability applied before the MLP head.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        entity_types: list[str],
        tokenizer_vocab_size: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialise encoder, resize embeddings, build classification head."""
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        self.encoder.resize_token_embeddings(tokenizer_vocab_size)

        hidden = config.hidden_size
        self.dropout = nn.Dropout(dropout)
        # Two-layer MLP: concat(e1, e2) → hidden → num_labels
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

        # Pre-build a lookup from entity type string to marker token string.
        self._entity_types = entity_types

    def _marker_ids(
        self,
        entity_types: list[str],
        tokenizer_vocab: dict[str, int],
    ) -> tuple[dict[str, int], dict[str, int]]:
        """Return subj/obj marker token-ID dicts for all entity types.

        Args:
            entity_types: Entity type strings present in the batch.
            tokenizer_vocab: Full tokenizer vocabulary mapping token → id.

        Returns:
            Tuple of ``(subj_ids, obj_ids)`` dicts mapping type → token id.
        """
        subj_ids = {t: tokenizer_vocab[SUBJ_START_TMPL.format(type=t)] for t in entity_types}
        obj_ids = {t: tokenizer_vocab[OBJ_START_TMPL.format(type=t)] for t in entity_types}
        return subj_ids, obj_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        subj_marker_ids: torch.Tensor,
        obj_marker_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run a forward pass.

        Args:
            input_ids: Token ID tensor of shape ``(batch, seq_len)``.
            attention_mask: Attention mask of shape ``(batch, seq_len)``.
            subj_marker_ids: Token IDs of the subject opening markers,
                shape ``(batch,)``.  Each value is the token ID of the
                ``[e1_TYPE]`` token for that example's subject entity type.
            obj_marker_ids: Token IDs of the object opening markers,
                shape ``(batch,)``.
            token_type_ids: Optional token type IDs.
            labels: Optional ground-truth class indices.

        Returns:
            Dict with ``"logits"`` (and ``"loss"`` when ``labels`` is provided).
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden_states = outputs.last_hidden_state  # (batch, seq, hidden)

        batch_size = input_ids.size(0)

        # Gather the hidden state at each example's subject and object markers.
        subj_pos = torch.stack(
            [
                _find_marker_position(input_ids[i : i + 1], subj_marker_ids[i].item())
                for i in range(batch_size)
            ]
        ).squeeze(-1)
        obj_pos = torch.stack(
            [
                _find_marker_position(input_ids[i : i + 1], obj_marker_ids[i].item())
                for i in range(batch_size)
            ]
        ).squeeze(-1)

        subj_repr = hidden_states[torch.arange(batch_size), subj_pos]  # (batch, hidden)
        obj_repr = hidden_states[torch.arange(batch_size), obj_pos]  # (batch, hidden)

        combined = self.dropout(torch.cat([subj_repr, obj_repr], dim=-1))  # (batch, 2*hidden)
        logits = self.classifier(combined)

        result: dict[str, torch.Tensor] = {'logits': logits}
        if labels is not None:
            result['loss'] = nn.CrossEntropyLoss()(logits, labels)
        return result
