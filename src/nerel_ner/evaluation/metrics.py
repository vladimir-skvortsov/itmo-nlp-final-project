"""Span-level NER evaluation using seqeval.

All metrics are computed at the entity *span* level (a prediction is correct
only if both the span boundaries and the entity type match the gold label).
This is the standard evaluation protocol used in the NEREL paper.
"""

from __future__ import annotations

from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    predictions: list[list[str]],
    references: list[list[str]],
) -> dict[str, float]:
    """Compute span-level NER metrics.

    Args:
        predictions: List of BIO label sequences (model output), one per
            sentence.  Each inner list contains string labels like
            ``"B-PER"``, ``"I-PER"``, ``"O"``.
        references: Corresponding gold BIO label sequences.

    Returns:
        Dict with keys ``"macro_f1"``, ``"macro_precision"``,
        ``"macro_recall"``, ``"micro_f1"``.  All values are percentages
        (0-100) rounded to two decimal places.
    """
    if not any(references):
        return {'macro_f1': 0.0, 'macro_precision': 0.0, 'macro_recall': 0.0, 'micro_f1': 0.0}

    return {
        'macro_f1': round(f1_score(references, predictions, average='macro') * 100, 2),
        'macro_precision': round(
            precision_score(references, predictions, average='macro') * 100, 2
        ),
        'macro_recall': round(
            recall_score(references, predictions, average='macro') * 100, 2
        ),
        'micro_f1': round(f1_score(references, predictions, average='micro') * 100, 2),
    }


def per_class_report(
    predictions: list[list[str]],
    references: list[list[str]],
) -> str:
    """Return a per-entity-type classification report string.

    Returns an empty string when no entity spans are present in either
    sequence (seqeval raises on empty target_names).

    Args:
        predictions: BIO prediction sequences.
        references: BIO reference sequences.

    Returns:
        Human-readable report from :func:`seqeval.metrics.classification_report`,
        or an empty string when there are no entities to report.
    """
    has_entities = any(lbl != 'O' for seq in references for lbl in seq) or any(
        lbl != 'O' for seq in predictions for lbl in seq
    )
    if not has_entities:
        return ''
    return classification_report(references, predictions, digits=4)


def print_results_table(results: dict[str, dict[str, float]]) -> None:
    """Print a Markdown-style comparison table.

    Args:
        results: Mapping from model name to metrics dict (as returned by
            :func:`compute_metrics`).
    """
    header = f"{'Model':<30} {'Macro F1':>10} {'Micro F1':>10} {'Precision':>10} {'Recall':>10}"
    print(header)
    print('-' * len(header))
    for model_name, metrics in results.items():
        print(
            f"{model_name:<30} "
            f"{metrics.get('macro_f1', 0.0):>10.2f} "
            f"{metrics.get('micro_f1', 0.0):>10.2f} "
            f"{metrics.get('macro_precision', 0.0):>10.2f} "
            f"{metrics.get('macro_recall', 0.0):>10.2f}"
        )
