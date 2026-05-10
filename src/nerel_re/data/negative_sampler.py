"""Negative example sampling for relation extraction.

NEREL only annotates positive relation instances.  For training, we need
"no_relation" examples constructed by pairing entities that do **not** have
an annotated relation between them.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import TYPE_CHECKING

from nerel_re.data.dataset import NO_RELATION, RelationExample

if TYPE_CHECKING:
    from nerel_re.data.dataset import Entity


class NegativeSampler:
    """Samples negative (no_relation) entity pairs from a document.

    For each document we pair all entities that do not have an annotated
    relation, then sub-sample to keep a configurable ratio relative to the
    number of positive examples.

    Args:
        negative_ratio: Number of negative examples per positive example.
        seed: Random seed for reproducibility.
    """

    def __init__(self, negative_ratio: float = 3.0, seed: int = 42) -> None:
        """Initialise the sampler."""
        self.negative_ratio = negative_ratio
        self._rng = random.Random(seed)  # noqa: S311

    def sample(self, positives: list[RelationExample]) -> list[RelationExample]:
        """Generate negative examples to complement a list of positive ones.

        Negatives are drawn from entity pairs within the same document that
        are not already covered by ``positives``.

        Args:
            positives: Positive :class:`RelationExample` instances (one split).

        Returns:
            Combined list of positive + sampled negative examples.
        """
        # Group by document so we can enumerate all intra-document pairs.
        by_doc: dict[str, list[RelationExample]] = defaultdict(list)
        for ex in positives:
            by_doc[ex.doc_id].append(ex)

        negatives: list[RelationExample] = []
        for doc_id, doc_examples in by_doc.items():
            n_neg = max(1, int(len(doc_examples) * self.negative_ratio))
            neg = self._sample_negatives_for_doc(doc_id, doc_examples, n_neg)
            negatives.extend(neg)

        combined = positives + negatives
        self._rng.shuffle(combined)
        return combined

    def _sample_negatives_for_doc(
        self,
        doc_id: str,
        positives: list[RelationExample],
        n_samples: int,
    ) -> list[RelationExample]:
        """Sample negatives for a single document.

        Args:
            doc_id: Document identifier.
            positives: All positive examples from this document.
            n_samples: Maximum number of negatives to generate.

        Returns:
            List of negative :class:`RelationExample` instances.
        """
        # Collect all entities and the set of already-annotated pairs.
        entities: dict[str, Entity] = {}
        annotated_pairs: set[tuple[str, str]] = set()

        for ex in positives:
            entities[ex.entity1.id] = ex.entity1
            entities[ex.entity2.id] = ex.entity2
            annotated_pairs.add((ex.entity1.id, ex.entity2.id))
            annotated_pairs.add((ex.entity2.id, ex.entity1.id))

        entity_list = list(entities.values())
        sentence = positives[0].sentence  # same doc → same full text

        candidates: list[RelationExample] = [
            RelationExample(
                doc_id=doc_id,
                sentence=sentence,
                entity1=e1,
                entity2=e2,
                relation=NO_RELATION,
            )
            for i, e1 in enumerate(entity_list)
            for e2 in entity_list[i + 1 :]
            if (e1.id, e2.id) not in annotated_pairs
        ]

        if not candidates:
            return []

        return self._rng.sample(candidates, min(n_samples, len(candidates)))
