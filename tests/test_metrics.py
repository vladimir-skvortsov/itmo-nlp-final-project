"""Tests for evaluation metrics."""

from nerel_re.data.dataset import NO_RELATION
from nerel_re.evaluation.metrics import compute_metrics, per_class_report

ID2LABEL = {0: NO_RELATION, 1: 'WORKS_FOR', 2: 'LOCATED_IN', 3: 'BORN_IN'}


def test_compute_metrics_perfect():
    y_true = [1, 2, 3, 1, 2]
    y_pred = [1, 2, 3, 1, 2]
    metrics = compute_metrics(y_true, y_pred, ID2LABEL)
    assert metrics['macro_f1'] == 100.0
    assert metrics['micro_f1'] == 100.0
    assert metrics['accuracy'] == 100.0


def test_compute_metrics_excludes_no_relation():
    # Perfect on NO_RELATION but zero on relation classes
    y_true = [0, 0, 1, 2]
    y_pred = [0, 0, 0, 0]  # predicts NO_RELATION for everything
    metrics = compute_metrics(y_true, y_pred, ID2LABEL)
    # Macro F1 should be 0 (all relation classes predicted as no_relation)
    assert metrics['macro_f1'] == 0.0
    # But accuracy > 0 because NO_RELATION is correct for first two
    assert metrics['accuracy'] > 0.0


def test_compute_metrics_no_relation_not_in_macro():
    # Macro F1 computed only over labels 1,2,3 — not 0 (no_relation)
    y_true = [1, 2, 3]
    y_pred = [1, 2, 3]
    metrics = compute_metrics(y_true, y_pred, ID2LABEL)
    assert metrics['macro_f1'] == 100.0


def test_compute_metrics_empty():
    metrics = compute_metrics([], [], ID2LABEL)
    assert metrics['accuracy'] == 0.0


def test_compute_metrics_returns_all_keys():
    metrics = compute_metrics([1], [1], ID2LABEL)
    assert set(metrics.keys()) == {
        'macro_f1',
        'macro_precision',
        'macro_recall',
        'micro_f1',
        'accuracy',
    }


def test_per_class_report_returns_string():
    y_true = [1, 2, 1, 2]
    y_pred = [1, 2, 2, 1]
    report = per_class_report(y_true, y_pred, ID2LABEL)
    assert isinstance(report, str)
    assert 'WORKS_FOR' in report
    assert 'LOCATED_IN' in report
