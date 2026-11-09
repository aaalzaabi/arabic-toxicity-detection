import numpy as np
import pandas as pd
import pytest

from aratox_nlp.bootstrap import compare_systems, point_scores
from aratox_nlp.calibration import (
    apply_platt,
    cross_fitted_systems,
    expected_calibration_error,
    fit_platt,
    tune_thresholds,
)
from aratox_nlp.constants import TOXIC_LABELS
from aratox_nlp.dialect_check import NOT_SCORED, agreement, build_sample
from aratox_nlp.explain import top_words, word_occlusion
from aratox_nlp.inference import ENSEMBLE_SYSTEM, REPORTED_POLICY, TUNED_POLICY, needs_calibration, threshold_vector
from aratox_nlp.metrics import multilabel_metrics
from aratox_nlp.preprocessing import config_from_dict, normalize_arabic
from aratox_nlp.robustness import PERTURBATIONS, add_diacritics, add_tatweel, arabizi


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=(400, 6))
    logits = (y * 2 - 1) * 1.5 + rng.normal(0, 1, size=y.shape)
    return y, logits


def test_platt_improves_calibration_of_overconfident_scores(toy):
    y, logits = toy
    overconfident = 1 / (1 + np.exp(-4 * logits))
    params = fit_platt(4 * logits, y)
    calibrated = apply_platt(4 * logits, params)
    assert expected_calibration_error(y, calibrated)[0] < expected_calibration_error(y, overconfident)[0]


def test_tuned_thresholds_come_from_grid_and_beat_a_bad_cutoff(toy):
    y, logits = toy
    probs = 1 / (1 + np.exp(-logits))
    thresholds = tune_thresholds(probs, y)
    assert thresholds.shape == (6,)
    assert np.all((thresholds >= 0.1) & (thresholds <= 0.9))
    tuned = multilabel_metrics(y, (probs >= thresholds).astype(int), TOXIC_LABELS)["macro_f1"]
    bad = multilabel_metrics(y, (probs >= 0.9).astype(int), TOXIC_LABELS)["macro_f1"]
    assert tuned >= bad


def test_cross_fitting_fills_every_row(toy):
    y, logits = toy
    probs, preds = cross_fitted_systems(logits, 1 / (1 + np.exp(-logits)), y, folds=4)
    assert set(preds) == {"svm_tuned", "qarib_tuned", "ensemble_tuned"}
    assert all(p.shape == y.shape for p in probs.values())
    assert np.all((probs["ensemble"] > 0) & (probs["ensemble"] < 1))


def test_bootstrap_ci_contains_point_estimate_and_paired_difference(toy):
    y, logits = toy
    good = (logits > 0).astype(int)
    worse = good.copy()
    worse[:80] = 1 - worse[:80]
    rows = {row["system"]: row for row in compare_systems(y, {"good": good, "worse": worse}, reference="worse", n_boot=300)}
    assert rows["good"]["macro_f1_ci_low"] <= rows["good"]["macro_f1"] <= rows["good"]["macro_f1_ci_high"]
    assert rows["good"]["diff_ci_low"] > 0
    assert point_scores(y, y)[0] == pytest.approx(1.0)


def test_preprocessing_undoes_diacritics_and_tatweel_but_not_arabizi():
    cfg = config_from_dict({"remove_diacritics": True, "remove_tatweel": True, "collapse_whitespace": True})
    text = "انت انسان محترم"
    assert normalize_arabic(add_diacritics(text, 1), cfg) == normalize_arabic(text, cfg)
    assert normalize_arabic(add_tatweel(text, 1), cfg) == normalize_arabic(text, cfg)
    assert not any("؀" <= ch <= "ۿ" for ch in arabizi(text))
    assert all(fn(text, 3) == fn(text, 3) for fn in PERTURBATIONS.values())


def test_word_occlusion_finds_the_trigger_word():
    def score_fn(texts):
        return np.array([[1.0 if "سيء" in t.split() else 0.0] * 6 for t in texts])

    words, base, deltas = word_occlusion("هذا كلام سيء جدا", score_fn)
    assert words[2] == "سيء"
    assert base[0] == 1.0
    assert top_words(words, deltas, 0) == [("سيء", 1.0)]


def test_dialect_check_sample_is_stratified_blank_and_scores_only_human_rows():
    frame = pd.DataFrame({
        "text": [f"جملة رقم {i}" for i in range(60)],
        "Sexual": [0] * 60,
        "predicted_country_dialect": ["Saudi"] * 30 + ["Egyptian"] * 30,
        "predicted_region": ["Gulf"] * 30 + ["Egyptian"] * 30,
        "model_confidence": [0.9] * 60,
    })
    sample = build_sample(frame, per_region=5)
    assert len(sample) == 10
    assert (sample.human_region == "").all()
    assert agreement(sample)["accuracy"] is None
    sample.loc[0, "human_region"] = sample.loc[0, "predicted_region"]
    sample.loc[1, "human_region"] = next(iter(NOT_SCORED))
    stats = agreement(sample)
    assert stats["labelled"] == 2 and stats["scored"] == 1 and stats["accuracy"] == 1.0


def test_inference_policies():
    assert needs_calibration(ENSEMBLE_SYSTEM, REPORTED_POLICY)
    assert list(threshold_vector("QARiB", REPORTED_POLICY, None)) == [0.5] * 6
    calibration = {"qarib": {"thresholds_tuned": [0.4] * 6}}
    assert list(threshold_vector("QARiB", TUNED_POLICY, calibration)) == [0.4] * 6
