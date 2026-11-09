import pandas as pd
import pytest
from scipy.stats import binomtest

from aratox_nlp.comparison import (
    align_predictions,
    disagreement_summary,
    error_counts_by_label,
    mcnemar_binary,
    per_label_table,
    wins_by_label,
)
from aratox_nlp.constants import TOXIC_LABELS


def _frame(prefix, gold_rows, pred_rows, texts=None):
    rows = []
    for index, (gold, pred) in enumerate(zip(gold_rows, pred_rows)):
        row = {
            "text_raw": f"text {index}" if texts is None else texts[index],
            "text_norm": f"text {index}" if texts is None else texts[index],
            "binary_toxic": int(sum(gold) > 0),
        }
        for label, value, predicted in zip(TOXIC_LABELS, gold, pred):
            row[label] = value
            row[f"{prefix}_pred_{label}"] = predicted
            row[f"{prefix}_score_{label}"] = float(predicted)
        row[f"{prefix}_pred_binary"] = int(sum(pred) > 0)
        rows.append(row)
    return pd.DataFrame(rows)


def _pair():
    gold = [[1, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
    svm = [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]]
    qarib = [[1, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
    return _frame("svm", gold, svm), _frame("qarib", gold, qarib)


def test_alignment_rejects_different_lengths():
    svm, qarib = _pair()
    with pytest.raises(ValueError, match="Row counts"):
        align_predictions(svm, qarib.iloc[:2])


def test_alignment_rejects_label_mismatch():
    svm, qarib = _pair()
    qarib = qarib.copy()
    qarib.loc[0, "Hatred"] = 1
    with pytest.raises(ValueError, match="Hatred"):
        align_predictions(svm, qarib)


def test_per_label_difference_and_disagreement_groups():
    aligned = align_predictions(*_pair())
    table = per_label_table(aligned)
    assert table.label.tolist() == TOXIC_LABELS
    hatred = table.set_index("label").loc["Hatred"]
    assert hatred.f1_difference == pytest.approx(hatred.qarib_f1 - hatred.tfidf_f1)
    assert hatred.f1_difference > 0
    from aratox_nlp.comparison import disagreement_frame
    groups = disagreement_frame(aligned).comparison_group.tolist()
    assert groups == ["both_correct", "qarib_only_correct", "qarib_only_correct"]
    summary = disagreement_summary(disagreement_frame(aligned))
    assert summary == {"n": 3, "both_correct": 1, "qarib_only_correct": 2, "tfidf_only_correct": 0, "both_wrong": 0}


def test_label_wins_and_error_counts():
    aligned = align_predictions(*_pair())
    wins = wins_by_label(aligned).set_index("label")
    assert wins.loc["Hatred", "qarib_only_correct"] == 1
    assert wins.loc["Appearance", "both_correct"] == 2
    counts = error_counts_by_label(aligned).set_index("label")
    assert counts.loc["Appearance", "tfidf_false_positives"] == 1
    assert counts.loc["Appearance", "qarib_false_positives"] == 0
    assert counts.loc["Hatred", "tfidf_false_negatives"] == 1


def test_mcnemar_exact_binomial():
    gold = [[0, 0, 0, 0, 0, 0]] * 5
    svm_pred = [[1, 0, 0, 0, 0, 0]] * 4 + [[0, 0, 0, 0, 0, 0]]
    qarib_pred = [[0, 0, 0, 0, 0, 0]] * 5
    aligned = align_predictions(_frame("svm", gold, svm_pred), _frame("qarib", gold, qarib_pred))
    result = mcnemar_binary(aligned)
    assert result["n01"] == 4
    assert result["n10"] == 0
    assert result["p_value"] == pytest.approx(float(binomtest(4, 4, 0.5, alternative="two-sided").pvalue))
    assert result["statistic"] == pytest.approx(4.0)
