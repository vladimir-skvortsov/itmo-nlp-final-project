"""Tokenisation helpers: entity markers, special token injection, encoding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import BatchEncoding, PreTrainedTokenizer

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


def _extract_entity_window(
    text: str,
    e1_start: int,
    e1_end: int,
    e2_start: int,
    e2_end: int,
    context_chars: int = 300,
) -> tuple[str, int, int, int, int]:
    """Extract a character window centred around both entity spans.

    Guarantees both entities are always present in the returned window,
    preventing entity markers from being silently truncated when documents
    are longer than ``max_length`` tokens.

    Args:
        text: Full document text.
        e1_start: Character start of entity 1.
        e1_end: Character end of entity 1.
        e2_start: Character start of entity 2.
        e2_end: Character end of entity 2.
        context_chars: Characters of context to keep on each side of the
            outermost entity boundary (default 300 ≈ 100 wordpiece tokens).

    Returns:
        Tuple of ``(windowed_text, adj_e1_start, adj_e1_end, adj_e2_start,
        adj_e2_end)`` with offsets adjusted to the window.
    """
    span_start = min(e1_start, e2_start)
    span_end = max(e1_end, e2_end)
    window_start = max(0, span_start - context_chars)
    window_end = min(len(text), span_end + context_chars)
    offset = window_start
    return (
        text[window_start:window_end],
        e1_start - offset,
        e1_end - offset,
        e2_start - offset,
        e2_end - offset,
    )


def encode_example(
    example: RelationExample,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256,
    *,
    use_typed_markers: bool = True,
    context_chars: int = 300,
) -> BatchEncoding:
    """Tokenize a :class:`RelationExample` with entity markers injected.

    A character window of ``context_chars`` is extracted around both entity
    spans before tokenisation so that entity markers are never truncated away,
    which is critical for the typed-markers architecture.

    Args:
        example: The relation extraction example to encode.
        tokenizer: A HuggingFace tokenizer with marker tokens already added.
        max_length: Maximum token sequence length (truncated/padded).
        use_typed_markers: Whether to use typed or plain entity markers.
        context_chars: Context window in characters around the outermost entity
            boundary (300 ≈ 100 wordpiece tokens of context per side).

    Returns:
        A :class:`BatchEncoding` containing ``input_ids``, ``attention_mask``,
        and optionally ``token_type_ids``.
    """
    windowed_text, e1_start, e1_end, e2_start, e2_end = _extract_entity_window(
        text=example.sentence,
        e1_start=example.entity1.start,
        e1_end=example.entity1.end,
        e2_start=example.entity2.start,
        e2_end=example.entity2.end,
        context_chars=context_chars,
    )
    marked_text = mark_entities(
        text=windowed_text,
        subj_start=e1_start,
        subj_end=e1_end,
        subj_type=example.entity1.type,
        obj_start=e2_start,
        obj_end=e2_end,
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
        encoding['subj_marker_ids'] = torch.tensor(
            [tokenizer.convert_tokens_to_ids(subj_marker)]
        )
        encoding['obj_marker_ids'] = torch.tensor(
            [tokenizer.convert_tokens_to_ids(obj_marker)]
        )
    return encoding
