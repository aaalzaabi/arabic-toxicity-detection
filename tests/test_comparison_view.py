import json
from pathlib import Path

import pandas as pd
import pytest

from aratox_nlp.comparison_view import (
    QARIB_SCORE_NOTE,
    SCORE_MAGNITUDE_NOTE,
    TFIDF_SCORE_NOTE,
    correctness_mark,
    disagreement_stats,
    displayed_latency_ratio,
    example_label_view,
    filter_validation_examples,
    fewer_tfidf_false_positives,
    missing_comparison_files,
    observed_latency_ratio,
    overall_improvement,
    per_label_interpretation,
    read_json,
    read_table,
    sorted_label_improvements,
    step_example_index,
)
from aratox_nlp.constants import TOXIC_LABELS

ROOT = Path(__file__).resolve().parents[1]


def test_disagreement_percentage_and_advantage():
    stats = disagreement_stats({
        "n": 4956,
        "both_correct": 3956,
        "qarib_only_correct": 321,
        "tfidf_only_correct": 219,
        "both_wrong": 460,
    })
    assert stats["disagree"] == 1000
    assert stats["disagree_percent"] == pytest.approx(1000 / 4956 * 100)
    assert round(stats["disagree_percent"], 1) == 20.2
    assert stats["qarib_exact_advantage"] == 321 - 219
    assert stats["qarib_exact_advantage"] == 102


def test_metric_file_loading_and_overall_delta(tmp_path):
    summary = pd.DataFrame([
        {"model": "TF-IDF + Linear SVM", "macro_f1": 0.890914, "micro_f1": 0.906247},
        {"model": "QARiB", "macro_f1": 0.903923, "micro_f1": 0.917180},
    ])
    path = tmp_path / "summary.csv"
    summary.to_csv(path, index=False)
    loaded = read_table(path)
    tfidf = loaded[loaded.model == "TF-IDF + Linear SVM"].iloc[0]
    qarib = loaded[loaded.model == "QARiB"].iloc[0]
    delta = overall_improvement(tfidf, qarib)
    assert delta["macro_f1"] == pytest.approx(0.013009, abs=1e-6)
    assert f"{delta['macro_f1']:+.3f}" == "+0.013"
    assert f"{delta['micro_f1']:+.3f}" == "+0.011"
    payload = {"n": 1, "both_correct": 1, "qarib_only_correct": 0, "tfidf_only_correct": 0, "both_wrong": 0}
    json_path = tmp_path / "summary.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    assert read_json(json_path)["n"] == 1


def test_missing_comparison_file_handling(tmp_path):
    metrics = tmp_path / "metrics"
    errors = tmp_path / "errors"
    metrics.mkdir()
    errors.mkdir()
    (metrics / "tfidf_vs_qarib_summary.csv").write_text("model\n", encoding="utf-8")
    missing = missing_comparison_files(metrics, errors)
    assert "tfidf_vs_qarib_summary.csv" not in missing
    assert "tfidf_vs_qarib_per_label.csv" in missing
    assert "tfidf_vs_qarib_disagreements.csv" in missing
    with pytest.raises(FileNotFoundError):
        read_table(metrics / "tfidf_vs_qarib_per_label.csv")


def test_sorted_per_label_improvement_and_interpretation():
    frame = pd.DataFrame({
        "label": TOXIC_LABELS,
        "tfidf_f1": [0.917, 0.951, 0.851, 0.759, 0.970, 0.898],
        "qarib_f1": [0.933, 0.954, 0.865, 0.781, 0.975, 0.916],
        "f1_difference": [0.016, 0.003, 0.015, 0.022, 0.004, 0.018],
    })
    ranked = sorted_label_improvements(frame)
    assert ranked.label.tolist() == ["Racial", "Violence", "Appearance", "Hatred", "Sexual", "Cussing"]
    assert ranked.f1_difference.is_monotonic_decreasing
    text = per_label_interpretation(ranked)
    assert text.startswith("QARiB improves all six toxicity labels.")
    assert "Racial and Violence" in text
    assert "Cussing and Sexual" in text


def test_validation_example_filtering_and_navigation():
    frame = pd.DataFrame({
        "text": ["a", "b", "c", "d"],
        "comparison_group": ["both_correct", "qarib_only_correct", "tfidf_only_correct", "both_wrong"],
        "true_labels": ["Hatred", "Hatred", "none", "Cussing"],
        "tfidf_predicted_labels": ["Hatred", "none", "none", "none"],
        "qarib_predicted_labels": ["Hatred", "Hatred", "Cussing", "none"],
    })
    qarib_wins = filter_validation_examples(frame, "qarib_only_correct")
    assert qarib_wins.text.tolist() == ["b"]
    assert step_example_index(0, 5, -1) == 4
    assert step_example_index(4, 5, 1) == 0
    view = example_label_view(qarib_wins.iloc[0])
    hatred = next(item for item in view if item["label"] == "Hatred")
    assert hatred["gold"] is True
    assert hatred["tfidf_correct"] is False
    assert hatred["qarib_correct"] is True
    assert hatred["models_differ"] is True


def test_custom_text_has_no_correctness_mark_and_score_notes_differ():
    assert correctness_mark(None) == ""
    assert correctness_mark(True) == "✓ correct"
    assert correctness_mark(False) == "✗ incorrect"
    assert "not calibrated probabilities" in TFIDF_SCORE_NOTE
    assert "Calibration is not assumed" in QARIB_SCORE_NOTE
    assert "should not be compared directly" in SCORE_MAGNITUDE_NOTE
    assert TFIDF_SCORE_NOTE != QARIB_SCORE_NOTE


def test_latency_ratio_uses_measured_milliseconds():
    assert observed_latency_ratio(1.970969916, 0.054185334) == pytest.approx(36.375, rel=1e-3)
    assert displayed_latency_ratio(1.970969916, 0.054185334) == pytest.approx(39.4)
    with pytest.raises(ValueError):
        observed_latency_ratio(1.0, 0.0)


def test_fewer_false_positives_come_from_the_table():
    frame = pd.DataFrame({
        "label": TOXIC_LABELS,
        "tfidf_false_positives": [48, 51, 133, 133, 25, 17],
        "qarib_false_positives": [52, 62, 128, 105, 23, 30],
    })
    chosen = fewer_tfidf_false_positives(frame)
    assert chosen.label.tolist() == ["Appearance", "Cussing", "Violence"]


def test_saved_comparison_files_match_reported_validation_counts():
    summary_path = ROOT / "outputs" / "metrics" / "tfidf_vs_qarib_disagreement_summary.json"
    if not summary_path.is_file():
        pytest.skip("saved disagreement summary is not present")
    stats = disagreement_stats(read_json(summary_path))
    assert stats["n"] == 4956
    assert stats["disagree"] == 1000
    assert stats["qarib_exact_advantage"] == 102
    per_label = read_table(ROOT / "outputs" / "metrics" / "tfidf_vs_qarib_per_label.csv")
    ranked = sorted_label_improvements(per_label)
    assert ranked.iloc[0].label == "Racial"
    assert ranked.iloc[0].f1_difference == pytest.approx(0.021903, abs=1e-4)
