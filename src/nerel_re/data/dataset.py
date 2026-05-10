"""NEREL dataset loading: BRAT annotation parser and PyTorch Dataset."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

# tokenizer_utils imports RelationExample only under TYPE_CHECKING, so no circular import.
from nerel_re.data.tokenizer_utils import encode_example

if TYPE_CHECKING:
    from collections.abc import Iterator

    from transformers import BatchEncoding, PreTrainedTokenizerBase

# Relation label used for entity pairs with no annotated relation.
NO_RELATION = 'no_relation'


@dataclass(frozen=True)
class Entity:
    """A named entity span in a document.

    Attributes:
        id: BRAT entity identifier (e.g. ``"T1"``).
        type: Entity type string (e.g. ``"PER"``, ``"ORG"``).
        start: Character offset of the first character (inclusive).
        end: Character offset of the last character (exclusive).
        text: Surface form of the entity.
    """

    id: str
    type: str
    start: int
    end: int
    text: str


@dataclass
class RelationExample:
    """A single labelled example for relation classification.

    Attributes:
        doc_id: Source document identifier.
        sentence: Raw sentence text (full context).
        entity1: Subject entity (head).
        entity2: Object entity (tail).
        relation: Relation type label or ``NO_RELATION``.
    """

    doc_id: str
    sentence: str
    entity1: Entity
    entity2: Entity
    relation: str = NO_RELATION


_ENTITY_RE = re.compile(r'^(T\d+)\t(\S+) (\d+) (\d+)\t(.+)$')
_RELATION_RE = re.compile(r'^(R\d+)\t(\S+) Arg1:(T\d+) Arg2:(T\d+)$')


def _parse_ann_file(
    ann_path: Path,
    _text: str,
) -> tuple[dict[str, Entity], list[tuple[str, str, str]]]:
    """Parse a BRAT ``.ann`` file into entities and relation triples.

    Args:
        ann_path: Path to the ``.ann`` annotation file.
        _text: Raw document text (reserved for future span sanity-checking).

    Returns:
        A tuple of:
        - ``entities``: mapping from BRAT ID to :class:`Entity`.
        - ``relations``: list of ``(relation_type, arg1_id, arg2_id)`` triples.
    """
    entities: dict[str, Entity] = {}
    relations: list[tuple[str, str, str]] = []

    for raw_line in ann_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if m := _ENTITY_RE.match(line):
            eid, etype, start, end, surface = m.groups()
            entities[eid] = Entity(
                id=eid,
                type=etype,
                start=int(start),
                end=int(end),
                text=surface,
            )
        elif m := _RELATION_RE.match(line):
            rid, rtype, arg1, arg2 = m.groups()
            del rid  # BRAT relation IDs are not used downstream
            relations.append((rtype, arg1, arg2))

    return entities, relations


def _examples_from_doc(
    doc_id: str,
    text: str,
    entities: dict[str, Entity],
    relations: list[tuple[str, str, str]],
) -> list[RelationExample]:
    """Convert parsed annotations into :class:`RelationExample` objects.

    Only in-sentence relation pairs are returned (both entities in the same
    sentence).  Positive examples are created from annotated relations;
    negative examples are **not** created here — see :class:`NegativeSampler`.

    Args:
        doc_id: Document identifier.
        text: Full document text.
        entities: Parsed entity annotations.
        relations: Parsed relation triples.

    Returns:
        List of positive :class:`RelationExample` instances.
    """
    # Build a lookup from entity pairs to relation type.
    positive_pairs: dict[tuple[str, str], str] = {
        (arg1, arg2): rtype for rtype, arg1, arg2 in relations
    }

    examples: list[RelationExample] = []
    for (arg1_id, arg2_id), rtype in positive_pairs.items():
        e1 = entities.get(arg1_id)
        e2 = entities.get(arg2_id)
        if e1 is None or e2 is None:
            continue
        # Use the full text as sentence context (sentence splitting is optional
        # and done at tokenization time via max_length truncation).
        examples.append(
            RelationExample(
                doc_id=doc_id,
                sentence=text,
                entity1=e1,
                entity2=e2,
                relation=rtype,
            )
        )
    return examples


def load_splits(
    data_dir: str | Path,
    splits: tuple[str, ...] = ('train', 'dev', 'test'),
) -> dict[str, list[RelationExample]]:
    """Load NEREL annotation files into per-split lists of examples.

    Expects the NEREL directory layout::

        data_dir/
            train/  *.txt  *.ann
            dev/    *.txt  *.ann
            test/   *.txt  *.ann

    Args:
        data_dir: Root of the NEREL dataset directory.
        splits: Which splits to load.

    Returns:
        Mapping from split name to list of positive :class:`RelationExample`.
    """
    data_dir = Path(data_dir)
    result: dict[str, list[RelationExample]] = {}

    for split in splits:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            msg = f'Split directory not found: {split_dir}'
            raise FileNotFoundError(msg)

        examples: list[RelationExample] = []
        for txt_path in sorted(split_dir.glob('*.txt')):
            doc_id = txt_path.stem
            ann_path = txt_path.with_suffix('.ann')
            if not ann_path.exists():
                continue
            text = txt_path.read_text(encoding='utf-8')
            entities, relations = _parse_ann_file(ann_path, text)
            examples.extend(_examples_from_doc(doc_id, text, entities, relations))

        result[split] = examples

    return result


class NERELDataset(Dataset):
    """PyTorch Dataset wrapping a list of :class:`RelationExample` objects.

    Args:
        examples: Pre-processed relation examples (positive + negative).
        tokenizer: HuggingFace tokenizer with typed-marker special tokens added.
        label2id: Mapping from relation label string to integer class index.
        max_length: Maximum sequence length passed to the tokenizer.
        use_typed_markers: If ``True``, wrap entities with typed markers
            (e.g. ``[e1_PER]…[/e1_PER]``); otherwise use plain ``[E1]``/``[E2]``.
    """

    def __init__(
        self,
        examples: list[RelationExample],
        tokenizer: PreTrainedTokenizerBase,
        label2id: dict[str, int],
        max_length: int = 256,
        *,
        use_typed_markers: bool = True,
    ) -> None:
        """Initialise the dataset and pre-tokenize all examples."""
        self.examples = examples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.use_typed_markers = use_typed_markers

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a single tokenized example as a dict of tensors.

        Args:
            idx: Example index.

        Returns:
            Dict with keys ``input_ids``, ``attention_mask``,
            ``token_type_ids`` (if present), and ``labels``.
        """
        example = self.examples[idx]
        encoding: BatchEncoding = encode_example(  # type: ignore[assignment]
            example,
            self.tokenizer,
            max_length=self.max_length,
            use_typed_markers=self.use_typed_markers,
        )
        label = self.label2id[example.relation]
        return {**{k: v.squeeze(0) for k, v in encoding.items()}, 'labels': torch.tensor(label)}

    def label_distribution(self) -> dict[str, int]:
        """Return a count of examples per relation type.

        Returns:
            Dict mapping relation label to its count.
        """
        counts: dict[str, int] = {}
        for ex in self.examples:
            counts[ex.relation] = counts.get(ex.relation, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def iter_labels(self) -> Iterator[int]:
        """Yield integer labels for all examples (used for weighted sampling).

        Yields:
            Integer class index for each example.
        """
        for ex in self.examples:
            yield self.label2id[ex.relation]
