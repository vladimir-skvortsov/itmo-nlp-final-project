"""Tokenisation helpers: entity markers, special token injection, encoding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import BatchEncoding, PreTrainedTokenizerBase

    from nerel_re.data.dataset import RelationExample

# ---------------------------------------------------------------------------
# Typed entity marker templates
# ---------------------------------------------------------------------------
# Typed markers encode entity type information directly in the input string.
# Reference: Zhou et al. (2022) "An Improved Baseline for Sentence-level
# Relation Extraction", AACL-IJCNLP.
#
# Instead of:   [CLS] … [E1] Obama [/E1] … [E2] US [/E2] …
# We use:       [CLS] … [e1_PER] Obama [/e1_PER] … [e2_LOC] US [/e2_LOC] …
#
# The representation at the [e1_TYPE] token is used as the subject entity
# embedding; [e2_TYPE] token for the object entity embedding.
# ---------------------------------------------------------------------------

SUBJ_START_TMPL = '[e1_{type}]'
SUBJ_END_TMPL = '[/e1_{type}]'
OBJ_START_TMPL = '[e2_{type}]'
OBJ_END_TMPL = '[/e2_{type}]'

# Plain markers used by the baseline (no type information).
PLAIN_SUBJ_START = '[E1]'
PLAIN_SUBJ_END = '[/E1]'
PLAIN_OBJ_START = '[E2]'
PLAIN_OBJ_END = '[/E2]'


def get_typed_marker_tokens(entity_types: list[str]) -> list[str]:
    """Return all special tokens needed for a given set of entity types.

    Args:
        entity_types: List of unique entity type strings (e.g. ``["PER", "ORG"]``).

    Returns:
        Sorted list of special token strings to add to the tokenizer vocabulary.
    """
    tokens: list[str] = []
    for etype in entity_types:
        tokens += [
            SUBJ_START_TMPL.format(type=etype),
            SUBJ_END_TMPL.format(type=etype),
            OBJ_START_TMPL.format(type=etype),
            OBJ_END_TMPL.format(type=etype),
        ]
    return sorted(set(tokens))


def get_plain_marker_tokens() -> list[str]:
    """Return special tokens used by the baseline (no type information).

    Returns:
        List of four plain marker token strings.
    """
    return [PLAIN_SUBJ_START, PLAIN_SUBJ_END, PLAIN_OBJ_START, PLAIN_OBJ_END]


def mark_entities(
    text: str,
    subj_start: int,
    subj_end: int,
    subj_type: str,
    obj_start: int,
    obj_end: int,
    obj_type: str,
    *,
    use_typed_markers: bool,
) -> str:
    """Insert entity markers around subject and object spans in ``text``.

    Handles the case where the object span comes before the subject span in
    the sentence.

    Args:
        text: Raw sentence text.
        subj_start: Character start offset of the subject entity.
        subj_end: Character end offset of the subject entity.
        subj_type: Entity type of the subject (e.g. ``"PER"``).
        obj_start: Character start offset of the object entity.
        obj_end: Character end offset of the object entity.
        obj_type: Entity type of the object (e.g. ``"ORG"``).
        use_typed_markers: If ``True`` inject typed markers; else plain markers.

    Returns:
        Modified sentence string with marker tokens inserted.
    """
    if use_typed_markers:
        ss = SUBJ_START_TMPL.format(type=subj_type)
        se = SUBJ_END_TMPL.format(type=subj_type)
        os_ = OBJ_START_TMPL.format(type=obj_type)
        oe = OBJ_END_TMPL.format(type=obj_type)
    else:
        ss, se = PLAIN_SUBJ_START, PLAIN_SUBJ_END
        os_, oe = PLAIN_OBJ_START, PLAIN_OBJ_END

    # Build list of (offset, insertion_string) sorted right-to-left so that
    # earlier insertions don't shift later offsets.
    insertions: list[tuple[int, str]] = [
        (subj_start, ss),
        (subj_end, se),
        (obj_start, os_),
        (obj_end, oe),
    ]
    insertions.sort(key=lambda x: -x[0])

    for offset, token in insertions:
        text = text[:offset] + token + text[offset:]

    return text


def encode_example(
    example: RelationExample,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 256,
    *,
    use_typed_markers: bool = True,
) -> BatchEncoding:
    """Tokenize a :class:`RelationExample` with entity markers injected.

    Args:
        example: The relation extraction example to encode.
        tokenizer: A HuggingFace tokenizer with marker tokens already added.
        max_length: Maximum token sequence length (truncated/padded).
        use_typed_markers: Whether to use typed or plain entity markers.

    Returns:
        A :class:`BatchEncoding` containing ``input_ids``, ``attention_mask``,
        and optionally ``token_type_ids``.
    """
    marked_text = mark_entities(
        text=example.sentence,
        subj_start=example.entity1.start,
        subj_end=example.entity1.end,
        subj_type=example.entity1.type,
        obj_start=example.entity2.start,
        obj_end=example.entity2.end,
        obj_type=example.entity2.type,
        use_typed_markers=use_typed_markers,
    )
    encoding = tokenizer(
        marked_text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    if use_typed_markers:
        subj_marker = SUBJ_START_TMPL.format(type=example.entity1.type)
        obj_marker = OBJ_START_TMPL.format(type=example.entity2.type)
        encoding['subj_marker_ids'] = torch.tensor([tokenizer.convert_tokens_to_ids(subj_marker)])
        encoding['obj_marker_ids'] = torch.tensor([tokenizer.convert_tokens_to_ids(obj_marker)])
    return encoding
