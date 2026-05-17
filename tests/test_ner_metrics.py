"""Unit tests for NER evaluation metrics."""


from nerel_ner.evaluation.metrics import compute_metrics, per_class_report

# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

def test_perfect_predictions():
    preds = [['B-PER', 'I-PER', 'O', 'B-ORG']]
    refs  = [['B-PER', 'I-PER', 'O', 'B-ORG']]
    m = compute_metrics(preds, refs)
    assert m['macro_f1'] == 100.0
    assert m['micro_f1'] == 100.0


def test_all_wrong():
    preds = [['O', 'O', 'O']]
    refs  = [['B-PER', 'I-PER', 'B-ORG']]
    m = compute_metrics(preds, refs)
    assert m['macro_f1'] == 0.0
    assert m['macro_recall'] == 0.0


def test_partial_correct():
    preds = [['B-PER', 'O', 'B-ORG']]
    refs  = [['B-PER', 'B-PER', 'B-ORG']]
    m = compute_metrics(preds, refs)
    # PER span mismatch; ORG correct
    assert 0.0 < m['macro_f1'] < 100.0


def test_empty_references():
    m = compute_metrics([], [])
    assert m['macro_f1'] == 0.0
    assert m['micro_f1'] == 0.0


def test_returns_percentages():
    preds = [['B-PER']]
    refs  = [['B-PER']]
    m = compute_metrics(preds, refs)
    for v in m.values():
        assert 0.0 <= v <= 100.0


def test_multiple_sentences():
    preds = [['B-PER', 'O'], ['B-ORG', 'I-ORG']]
    refs  = [['B-PER', 'O'], ['B-ORG', 'I-ORG']]
    m = compute_metrics(preds, refs)
    assert m['macro_f1'] == 100.0


# ---------------------------------------------------------------------------
# per_class_report
# ---------------------------------------------------------------------------

def test_per_class_report_returns_string():
    preds = [['B-PER', 'O', 'B-ORG']]
    refs  = [['B-PER', 'O', 'B-ORG']]
    report = per_class_report(preds, refs)
    assert isinstance(report, str)
    assert 'PER' in report
    assert 'ORG' in report


def test_per_class_report_all_outside():
    preds = [['O', 'O']]
    refs  = [['O', 'O']]
    report = per_class_report(preds, refs)
    assert isinstance(report, str)
