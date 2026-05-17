"""Data loading and BIO tagging for NEREL NER."""

from nerel_ner.data.dataset import (
    Entity,
    NERELNERDataset,
    NERSentence,
    build_label_maps,
    load_splits,
)

__all__ = [
    'Entity',
    'NERELNERDataset',
    'NERSentence',
    'build_label_maps',
    'load_splits',
]
