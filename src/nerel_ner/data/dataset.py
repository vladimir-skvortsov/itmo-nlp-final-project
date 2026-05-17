"""NEREL NER dataset: BRAT annotation parser → BIO tags → PyTorch Dataset.

The NEREL corpus uses BRAT standoff format:
    doc.txt  — raw document text
    doc.ann  — entity/relation annotations with character offsets

This module handles only the entity layer (NER).  Documents are split into
sentences on newline boundaries; entity spans are aligned to subword tokens
via offset mapping, producing standard BIO labels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from collections.abc import Iterator

    from transformers import PreTrainedTokenizerFast

_ENTITY_RE = re.compile(r'^(T\d+)\t(\S+) (\d+) (\d+)\t(.+)$')

# Token positions with this label value are excluded from loss computation.
LABEL_IGNORE_INDEX: int = -100


@dataclass(frozen=True)
class Entity:
    """A named entity span with character offsets into the source document.

    Attributes:
        id: BRAT annotation identifier (e.g. ``"T1"``).
        type: Entity type string (e.g. ``"PER"``, ``"ORG"``).
        start: Inclusive character offset in the document.
        end: Exclusive character offset in the document.
        text: Surface form of the entity.
    """

    id: str
    type: str
    start: int
    end: int
    text: str


@dataclass
class NERSentence:
    """A single sentence with its entity annotations.

    Attributes:
        doc_id: Source document identifier.
        text: Raw sentence text.
        offset: Character offset of this sentence's start in the document.
        entities: Entities whose start offset falls within this sentence.
    """

    doc_id: str
    text: str
    offset: int
    entities: list[Entity] = field(default_factory=list)


def _parse_entities(ann_path: Path) -> list[Entity]:
    """Parse a BRAT ``.ann`` file, returning entity annotations only.

    Relation lines (``R…``) and attribute lines (``A…``) are silently skipped.

    Args:
        ann_path: Path to the ``.ann`` file.

    Returns:
        List of :class:`Entity` objects sorted by start offset.
    """
    entities: list[Entity] = []
    for raw in ann_path.read_text(encoding='utf-8').splitlines():
        if m := _ENTITY_RE.match(raw.strip()):
            eid, etype, start, end, surface = m.groups()
            entities.append(
                Entity(id=eid, type=etype, start=int(start), end=int(end), text=surface)
            )
    return sorted(entities, key=lambda e: e.start)


def _split_sentences(
    doc_id: str,
    text: str,
    entities: list[Entity],
) -> list[NERSentence]:
    """Split a document into sentences on newline boundaries.

    Each non-empty line becomes one :class:`NERSentence`.  Entities are
    assigned to the sentence that contains their *start* offset.

    Args:
        doc_id: Source document identifier.
        text: Full document text.
        entities: Parsed entity annotations (document-level offsets).

    Returns:
        List of :class:`NERSentence` objects in document order.
    """
    sentences: list[NERSentence] = []
    offset = 0
    for line in text.split('\n'):
        line_end = offset + len(line)
        if line.strip():
            line_entities = [e for e in entities if offset <= e.start < line_end]
            sentences.append(
                NERSentence(doc_id=doc_id, text=line, offset=offset, entities=line_entities)
            )
        offset = line_end + 1  # +1 for the consumed '\n'
    return sentences


def load_splits(
    data_dir: str | Path,
    splits: tuple[str, ...] = ('train', 'dev', 'test'),
) -> dict[str, list[NERSentence]]:
    """Load NEREL annotation files into per-split sentence lists.

    Expects the NEREL directory layout::

        data_dir/
            train/  *.txt  *.ann
            dev/    *.txt  *.ann
            test/   *.txt  *.ann

    Args:
        data_dir: Root of the NEREL dataset directory.
        splits: Which splits to load.

    Returns:
        Mapping from split name to list of :class:`NERSentence` objects.
    """
    data_dir = Path(data_dir)
    result: dict[str, list[NERSentence]] = {}

    for split in splits:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            msg = f'Split directory not found: {split_dir}'
            raise FileNotFoundError(msg)

        sentences: list[NERSentence] = []
        for txt_path in sorted(split_dir.glob('*.txt')):
            doc_id = txt_path.stem
            ann_path = txt_path.with_suffix('.ann')
            if not ann_path.exists():
                continue
            text = txt_path.read_text(encoding='utf-8')
            entities = _parse_entities(ann_path)
            sentences.extend(_split_sentences(doc_id, text, entities))

        result[split] = sentences

    return result


def collect_entity_types(splits: dict[str, list[NERSentence]]) -> list[str]:
    """Return a sorted list of all entity type strings across all splits.

    Args:
        splits: Mapping from split name to sentence list.

    Returns:
        Sorted list of unique entity type strings.
    """
    types: set[str] = set()
    for sentences in splits.values():
        for sent in sentences:
            for ent in sent.entities:
                types.add(ent.type)
    return sorted(types)


def build_label_maps(entity_types: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Build ``label2id`` / ``id2label`` for the BIO tagging scheme.

    The ``O`` label is assigned index 0; B/I labels follow in alphabetical
    order of entity type.

    Args:
        entity_types: Sorted list of unique entity type strings.

    Returns:
        Tuple of ``(label2id, id2label)`` dicts.
    """
    labels = ['O']
    for etype in sorted(entity_types):
        labels.append(f'B-{etype}')
        labels.append(f'I-{etype}')
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return label2id, id2label


def _align_labels(
    offset_mapping: list[tuple[int, int]],
    entities: list[Entity],
    label2id: dict[str, int],
    sentence_offset: int,
) -> list[int]:
    """Align BIO integer labels to subword tokens via offset mapping.

    Special and padding tokens have zero-length offsets ``(n, n)`` and are
    assigned the ignore index ``-100`` so they are excluded from the loss.

    For entity spans that cover multiple subword tokens, the first overlapping
    token receives the ``B-TYPE`` label and all subsequent tokens receive
    ``I-TYPE``.

    Args:
        offset_mapping: Per-token ``(char_start, char_end)`` relative to the
            tokenised sentence text.
        entities: Entities for this sentence (document-level offsets).
        label2id: Label-to-index mapping produced by :func:`build_label_maps`.
        sentence_offset: Character offset of the sentence start in the document.

    Returns:
        List of integer labels aligned to the token sequence.
    """
    labels: list[int] = []
    for tok_start, tok_end in offset_mapping:
        if tok_start == tok_end:  # special / padding token
            labels.append(-100)
            continue

        # Convert sentence-relative token offsets to document-level offsets.
        doc_start = sentence_offset + tok_start
        doc_end = sentence_offset + tok_end

        label = label2id['O']
        for ent in entities:
            # Skip entities with no overlap.
            if doc_end <= ent.start or doc_start >= ent.end:
                continue
            # First subword of the entity span → B; continuations → I.
            tag = 'B' if doc_start <= ent.start else 'I'
            label = label2id.get(f'{tag}-{ent.type}', label2id['O'])
            break  # entities are sorted; first match wins

        labels.append(label)
    return labels


class NERELNERDataset(Dataset):
    """PyTorch :class:`~torch.utils.data.Dataset` for NEREL NER.

    Args:
        sentences: Pre-loaded :class:`NERSentence` objects.
        tokenizer: Fast HuggingFace tokenizer with ``return_offsets_mapping``
            support.
        label2id: Label-to-index mapping from :func:`build_label_maps`.
        max_length: Maximum subword sequence length (truncation + padding).
    """

    def __init__(
        self,
        sentences: list[NERSentence],
        tokenizer: PreTrainedTokenizerFast,
        label2id: dict[str, int],
        max_length: int = 128,
    ) -> None:
        """Initialise dataset."""
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        """Return number of sentences."""
        return len(self.sentences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Tokenize sentence and return aligned BIO labels.

        Args:
            idx: Sentence index.

        Returns:
            Dict with ``input_ids``, ``attention_mask``,
            ``token_type_ids`` (if present), and ``labels``.
        """
        sent = self.sentences[idx]
        encoding = self.tokenizer(
            sent.text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True,
        )
        # offset_mapping must be consumed before passing encoding to model.
        offset_mapping: list[tuple[int, int]] = (
            encoding.pop('offset_mapping')[0].tolist()
        )
        labels = _align_labels(
            offset_mapping=offset_mapping,
            entities=sent.entities,
            label2id=self.label2id,
            sentence_offset=sent.offset,
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item['labels'] = torch.tensor(labels, dtype=torch.long)
        return item

    def iter_entity_types(self) -> Iterator[str]:
        """Yield entity type strings for all entities in the dataset.

        Yields:
            Entity type string for each entity across all sentences.
        """
        for sent in self.sentences:
            for ent in sent.entities:
                yield ent.type
