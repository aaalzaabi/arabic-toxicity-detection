import inspect

import numpy as np
import pandas as pd
import pytest

import aratox_nlp.ethics as ethics
from aratox_nlp.comparison import QARIB_NAME, SVM_NAME
from aratox_nlp.constants import TOXIC_LABELS
from aratox_nlp.ethics import (
    CURRENT_THRESHOLD,
    DIALECT_CAVEAT,
    DIALECT_NOT_IDENTITY,
    EXPLORER_NOTE,
    FALSE_POSITIVE_COLUMNS,
    HUMAN_REVIEW,
    THRESHOLD_GRID,
    blank_review_frame,
    false_positive_rate,
    false_positive_summary,
    nearest_threshold_row,
    operating_points,
    threshold_direction,
    threshold_tradeoff,
)


def _aligned():
    row = {
        "text_raw": "مثال",
        "text_norm": "مثال",
        "binary_toxic": 0,
    }
    for label in TOXIC_LABELS:
        row[label] = 0
        row[f"svm_pred_{label}"] = 0
        row[f"qarib_pred_{label}"] = 0
        row[f"svm_score_{label}"] = 0.1
        row[f"qarib_score_{label}"] = 0.1
    row["svm_pred_binary"] = 0
    row["qarib_pred_binary"] = 0
    first = dict(row)
    second = dict(row)
    second["Cussing"] = 0
    second["svm_pred_Cussing"] = 1
    second["qarib_pred_Cussing"] = 0
    third = dict(row)
    third["Hatred"] = 1
    third["binary_toxic"] = 1
    third["svm_pred_Hatred"] = 0
    third["qarib_pred_Hatred"] = 1
    third["qarib_pred_binary"] = 1
    return pd.DataFrame([first, second, third])


def test_false_positive_rate_and_zero_true_negatives():
    assert false_positive_rate(2, 8) == pytest.approx(0.2)
    assert false_positive_rate(0, 0) == 0.0
    assert false_positive_rate(4, 0) == 1.0


def test_false_positive_summary_schema_excludes_dialect():
    table = false_positive_summary(_aligned())
    assert tuple(table.columns) == FALSE_POSITIVE_COLUMNS
    assert "dialect" not in table.columns
    cussing = table[(table.model == SVM_NAME) & (table.label == "Cussing")].iloc[0]
    assert cussing.false_positives == 1
    assert cussing.true_negatives == 2
    assert cussing.false_positive_rate == pytest.approx(1 / 3)
    assert set(table.model) == {SVM_NAME, QARIB_NAME}
    assert table.label.nunique() == 6


def test_threshold_table_is_sorted_and_lookup_does_not_train():
    gold = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]])
    scores = np.array([[0.9, 0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2, 0.2], [0.6, 0.4, 0.1, 0.1, 0.1, 0.1]])
    overall, per_label = threshold_tradeoff(gold, scores, (0.3, 0.5, 0.8))
    assert overall.threshold.tolist() == [0.3, 0.5, 0.8]
    assert overall.threshold.is_monotonic_increasing
    assert bool(per_label.groupby("label").threshold.apply(lambda values: values.is_monotonic_increasing).all())
    assert list(per_label.columns) == [
        "label", "threshold", "precision", "recall", "f1",
        "false_positives", "false_negatives", "false_positive_rate",
    ]
    direction = threshold_direction(overall)
    assert direction["from_threshold"] < direction["to_threshold"]
    points = operating_points(overall, CURRENT_THRESHOLD)
    assert set(points) == {"highest_macro_f1", "highest_macro_precision", "lowest_false_positives", "current"}
    assert points["current"]["threshold"] == pytest.approx(0.5)
    assert nearest_threshold_row(overall, 0.45).threshold == pytest.approx(0.5)
    source = inspect.getsource(ethics)
    assert "import torch" not in source
    assert "joblib" not in source
    assert "train_" not in source


def test_review_fields_stay_blank_and_warnings_are_explicit():
    reviewed = blank_review_frame(pd.DataFrame([{"text": "مثال", "notes": None}]))
    assert reviewed.loc[0, "possible_slang"] == ""
    assert reviewed.loc[0, "possible_reclaimed_expression"] == ""
    assert reviewed.loc[0, "notes"] == ""
    assert "not ground-truth" in DIALECT_CAVEAT
    assert "nationality" in DIALECT_NOT_IDENTITY
    assert "human review is required" in HUMAN_REVIEW
    assert "does not change the deployed model" in EXPLORER_NOTE
    assert THRESHOLD_GRID[0] == 0.3
    assert THRESHOLD_GRID[-1] == 0.8
    assert 0.5 in THRESHOLD_GRID


def test_app_keeps_ethics_warnings_and_does_not_feed_dialect_into_toxicity():
    source = (ethics.__file__.replace("ethics.py", "app.py"))
    text = open(source, encoding="utf-8").read()
    assert "DIALECT_CAVEAT" in text
    assert "HUMAN_REVIEW" in text
    assert "Responsible AI" in text
    responsible = text.split("def responsible_ai_view")[1].split("\nanalyze_tab")[0]
    assert "dialect_for" not in responsible
    assert "load_dialect" not in responsible
    assert "load_qarib" not in responsible
    assert "score_pair" not in responsible
