"""Evaluation metrics matching the NEREL paper's evaluation protocol.

Macro F1 is computed **excluding** the ``no_relation`` class, following the
convention in (Loukachevitch et al., 2021) and most RE benchmarks.
"""

from __future__ import annotations

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from nerel_re.data.dataset import NO_RELATION


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    id2label: dict[int, str],
) -> dict[str, float]:
    """Compute macro/micro F1, precision, recall for RE predictions.

    The ``no_relation`` class is excluded from macro averaging.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        id2label: Mapping from integer class index to relation label string.

    Returns:
        Dict with keys ``macro_f1``, ``macro_precision``, ``macro_recall``,
        ``micro_f1``, and ``accuracy``.
    """
    if not y_true:
        return {
            "macro_f1": 0.0, "macro_precision": 0.0, "macro_recall": 0.0,
            "micro_f1": 0.0, "accuracy": 0.0,
        }

    # Labels to include in macro averaging (exclude no_relation).
    relation_labels = [idx for idx, lbl in id2label.items() if lbl != NO_RELATION]

    macro_f1 = f1_score(y_true, y_pred, labels=relation_labels, average="macro", zero_division=0)
    macro_p = precision_score(
        y_true, y_pred, labels=relation_labels, average="macro", zero_division=0
    )
    macro_r = recall_score(
        y_true, y_pred, labels=relation_labels, average="macro", zero_division=0
    )
    micro_f1 = f1_score(y_true, y_pred, labels=relation_labels, average="micro", zero_division=0)

    n_correct = sum(t == p for t, p in zip(y_true, y_pred, strict=True))
    accuracy = n_correct / len(y_true) if y_true else 0.0

    return {
        "macro_f1": round(macro_f1 * 100, 2),
        "macro_precision": round(macro_p * 100, 2),
        "macro_recall": round(macro_r * 100, 2),
        "micro_f1": round(micro_f1 * 100, 2),
        "accuracy": round(accuracy * 100, 2),
    }


def per_class_report(
    y_true: list[int],
    y_pred: list[int],
    id2label: dict[int, str],
) -> str:
    """Return a human-readable per-class classification report.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        id2label: Mapping from integer class index to relation label string.

    Returns:
        Multi-line string report from :func:`sklearn.metrics.classification_report`.
    """
    labels = sorted(id2label)
    target_names = [id2label[i] for i in labels]
    return classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, zero_division=0
    )


def print_results_table(results: dict[str, dict[str, float]]) -> None:
    """Print a comparison table of results from multiple models.

    Args:
        results: Mapping from model name to its metrics dict
            (as returned by :func:`compute_metrics`).
    """
    header = f"{'Model':<30} {'Macro F1':>10} {'Micro F1':>10} {'Accuracy':>10}"
    print(header)
    print("-" * len(header))
    for model_name, metrics in results.items():
        print(
            f"{model_name:<30} "
            f"{metrics.get('macro_f1', 0):>10.2f} "
            f"{metrics.get('micro_f1', 0):>10.2f} "
            f"{metrics.get('accuracy', 0):>10.2f}"
        )
