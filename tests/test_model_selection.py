import inspect
from pathlib import Path

import pandas as pd
import pytest

import aratox_nlp.error_review as error_review
import aratox_nlp.model_selection as model_selection
import aratox_nlp.operating_modes as operating_modes
from aratox_nlp.comparison import QARIB_NAME, SVM_NAME
from aratox_nlp.constants import TOXIC_LABELS
from aratox_nlp.error_review import (
    REVIEW_CATEGORIES,
    REVIEW_COLUMNS,
    apply_review,
    false_positive_events,
    false_positives_by_predicted_region,
    review_frame,
    review_summary,
    stratified_review_sample,
)
from aratox_nlp.model_selection import JAIS_STATUS, load_model_selection, scorecard
from aratox_nlp.operating_modes import MODES, mode_at, operating_mode_table

ROOT = Path(__file__).resolve().parents[1]


def _events():
    rows = []
    for label in TOXIC_LABELS:
        for index in range(8):
            rows.append({
                "row_index": index + TOXIC_LABELS.index(label) * 10,
                "text_raw": f"{label} {index}",
                "true_labels": "none",
                "tfidf_predicted_labels": label,
                "qarib_predicted_labels": "none",
                "trigger_model": SVM_NAME,
                "trigger_label": label,
            })
    return pd.DataFrame(rows)


def _frame():
    rows = []
    for index in range(4):
        row = {"text_raw": f"text {index}", "text_norm": f"text {index}", "binary_toxic": 0}
        for label in TOXIC_LABELS:
            row[label] = 0
            row[f"svm_pred_{label}"] = 0
            row[f"qarib_pred_{label}"] = 0
        row["svm_pred_binary"] = 0
        row["qarib_pred_binary"] = 0
        rows.append(row)
    rows[0]["svm_pred_Appearance"] = 1
    rows[1]["qarib_pred_Cussing"] = 1
    rows[2]["svm_pred_Violence"] = 1
    rows[2]["qarib_pred_Violence"] = 1
    return pd.DataFrame(rows)


def test_operating_modes_load_saved_thresholds_without_inference():
    tradeoff = pd.DataFrame([
        {"threshold": 0.50, "macro_f1": 0.904, "macro_precision": 0.911, "macro_recall": 0.897, "total_false_positives": 400, "total_false_negatives": 460, "subset_accuracy": 0.863},
        {"threshold": 0.65, "macro_f1": 0.906, "macro_precision": 0.923, "macro_recall": 0.890, "total_false_positives": 338, "total_false_negatives": 489, "subset_accuracy": 0.867},
        {"threshold": 0.80, "macro_f1": 0.904, "macro_precision": 0.935, "macro_recall": 0.876, "total_false_positives": 273, "total_false_negatives": 553, "subset_accuracy": 0.868},
    ])
    table = operating_mode_table(tradeoff)
    assert table.threshold.tolist() == [0.50, 0.65, 0.80]
    assert mode_at(table, 0.65)["mode"] == "F1-focused exploratory"
    assert "reported" in mode_at(table, 0.50).interpretation.lower() or "Preserve" in mode_at(table, 0.50).interpretation
    source = inspect.getsource(operating_modes)
    assert "import torch" not in source
    assert "joblib" not in source
    assert len(MODES) == 3


def test_model_selection_uses_saved_metrics_and_excludes_jais(tmp_path):
    pd.DataFrame([
        {"model": SVM_NAME, "macro_f1": 0.891, "micro_f1": 0.906},
        {"model": QARIB_NAME, "macro_f1": 0.904, "micro_f1": 0.917},
    ]).to_csv(tmp_path / "tfidf_vs_qarib_summary.csv", index=False)
    pd.DataFrame([
        {"model": SVM_NAME, "mean_ms_per_sample": 0.05, "samples_per_second": 18000, "device": "cpu"},
        {"model": QARIB_NAME, "mean_ms_per_sample": 1.97, "samples_per_second": 507, "device": "mps"},
    ]).to_csv(tmp_path / "tfidf_vs_qarib_latency.csv", index=False)
    pd.DataFrame({
        "label": TOXIC_LABELS,
        "tfidf_false_positives": [48, 51, 133, 133, 25, 17],
        "qarib_false_positives": [52, 62, 128, 105, 23, 30],
    }).to_csv(tmp_path / "error_counts_by_label.csv", index=False)
    table = load_model_selection(tmp_path)
    assert table.model.tolist() == [SVM_NAME, QARIB_NAME]
    assert "Jais" not in table.model.tolist()
    card = scorecard(table)
    assert card.loc[card.Dimension == "Macro-F1", "QARiB"].iloc[0] == "0.904"
    assert "ethical_score" not in card.columns
    assert "evaluation pending" in JAIS_STATUS
    assert "ethical_score =" not in inspect.getsource(model_selection)
    assert "0.4 *" not in inspect.getsource(model_selection)


def test_review_sample_is_reproducible_and_unlabeled():
    events = _events()
    first = stratified_review_sample(events, size=20, seed=42)
    second = stratified_review_sample(events, size=20, seed=42)
    assert first.row_index.tolist() == second.row_index.tolist()
    assert first.trigger_label.nunique() == 6
    dialect = pd.DataFrame({
        "source_row": first.row_index.tolist(),
        "predicted_country_dialect": ["Egyptian"] * len(first),
        "predicted_region": ["Egyptian"] * len(first),
    })
    review = review_frame(attach_dialect(first, dialect))
    assert list(review.columns) == list(REVIEW_COLUMNS)
    assert set(review.review_category) == {""}
    assert set(review.reviewed) == {""}
    assert set(review.review_notes) == {""}
    assert review.review_category.tolist() != list(REVIEW_CATEGORIES)


def attach_dialect(sample, dialect):
    from aratox_nlp.error_review import attach_predicted_dialect
    return attach_predicted_dialect(sample, dialect)


def test_review_summary_uses_only_reviewed_rows_and_preserves_predictions():
    events = false_positive_events(_frame())
    sample = review_frame(attach_predicted_dialect_blank(stratified_review_sample(events, size=3, seed=42)))
    assert review_summary(sample).empty
    updated = apply_review(sample, sample.iloc[0].row_id, ["lexical_trigger", "other"], "manual note")
    assert updated.iloc[0].true_labels == sample.iloc[0].true_labels
    assert updated.iloc[0].tfidf_predicted_labels == sample.iloc[0].tfidf_predicted_labels
    summary = review_summary(updated)
    assert set(summary.review_category) == {"lexical_trigger", "other"}
    assert summary["count"].sum() == 2
    assert summary.percentage_of_reviewed_examples.iloc[0] == pytest.approx(100.0)
    with pytest.raises(ValueError):
        apply_review(sample, sample.iloc[0].row_id, ["not_a_category"], "")


def attach_predicted_dialect_blank(sample):
    from aratox_nlp.error_review import attach_predicted_dialect
    return attach_predicted_dialect(sample, None)


def test_region_table_is_exploratory_and_sources_avoid_the_test_split():
    events = false_positive_events(_frame())
    dialect = pd.DataFrame({
        "source_row": [0, 1, 2, 3],
        "predicted_region": ["Gulf", "Gulf", "Egyptian", "Iraqi"],
    })
    table = false_positives_by_predicted_region(events, dialect)
    assert {"predicted_region", "n_examples", "n_false_positives", "false_positive_share"} <= set(table.columns)
    assert "fairness" not in table.columns
    for module in (error_review, model_selection, operating_modes):
        source = inspect.getsource(module)
        assert "test.csv" not in source
        assert "Dialect Fairness" not in source
    script = (ROOT / "scripts" / "build_false_positive_review_sample.py").read_text(encoding="utf-8")
    assert "test.csv" not in script
    assert "train.csv" not in script
    assert "cannot establish fairness" in model_selection.LINGUISTIC_ROBUSTNESS


def test_saved_reported_qarib_threshold_remains_half():
    tradeoff_path = ROOT / "outputs" / "ethics" / "qarib_threshold_tradeoff.csv"
    summary_path = ROOT / "outputs" / "metrics" / "tfidf_vs_qarib_summary.csv"
    if not tradeoff_path.is_file() or not summary_path.is_file():
        pytest.skip("saved validation metrics are not present")
    tradeoff = pd.read_csv(tradeoff_path)
    reported = operating_mode_table(tradeoff)
    current = mode_at(reported, 0.50)
    qarib = pd.read_csv(summary_path)
    qarib = qarib[qarib.model == QARIB_NAME].iloc[0]
    assert current.threshold == pytest.approx(0.5)
    assert current.macro_f1 == pytest.approx(float(qarib.macro_f1))
    assert float(qarib.macro_f1) == pytest.approx(0.903923, abs=1e-6)
