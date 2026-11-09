"""Score calibration, per-label thresholds and the calibrated ensemble.

Everything here is fitted on validation predictions only. Cross-fitted helpers
give an honest validation estimate: each fold is scored by calibrators and
thresholds fitted on the other folds.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

from .constants import TOXIC_LABELS

EPS = 1e-6
THRESHOLD_GRID = np.round(np.arange(0.10, 0.90 + 1e-9, 0.05), 2)


def to_logit(probs):
    p = np.clip(np.asarray(probs, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibration_input(scores, kind):
    """SVM scores are raw decision values; QARiB scores are sigmoid probabilities."""
    scores = np.asarray(scores, dtype=float)
    return scores if kind == "svm" else to_logit(scores)


def fit_platt(features, y_true):
    """One Platt scaler (slope, intercept) per label on a 1-D score."""
    params = []
    for i in range(features.shape[1]):
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(features[:, [i]], y_true[:, i])
        params.append([float(lr.coef_[0, 0]), float(lr.intercept_[0])])
    return np.asarray(params)


def apply_platt(features, params):
    params = np.asarray(params, dtype=float)
    return 1 / (1 + np.exp(-(features * params[:, 0] + params[:, 1])))


def tune_thresholds(probs, y_true, grid=THRESHOLD_GRID):
    """Per-label threshold that maximizes validation F1."""
    best = []
    for i in range(probs.shape[1]):
        f1s = [f1_score(y_true[:, i], (probs[:, i] >= t).astype(int), zero_division=0) for t in grid]
        best.append(float(grid[int(np.argmax(f1s))]))
    return np.asarray(best)


def ensemble_probs(svm_cal, qarib_cal, weight=0.5):
    return weight * np.asarray(svm_cal) + (1 - weight) * np.asarray(qarib_cal)


def expected_calibration_error(y_true, probs, bins=10):
    """Mean over labels of the binned |accuracy - confidence| gap."""
    edges = np.linspace(0, 1, bins + 1)
    per_label = []
    for i in range(probs.shape[1]):
        p, y = probs[:, i], y_true[:, i]
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
        gap = 0.0
        for b in range(bins):
            mask = idx == b
            if mask.any():
                gap += mask.mean() * abs(y[mask].mean() - p[mask].mean())
        per_label.append(gap)
    return float(np.mean(per_label)), dict(zip(TOXIC_LABELS, map(float, per_label)))


def reliability_curve(y_true, probs, bins=10):
    """Pooled over labels: mean predicted probability vs observed rate per bin."""
    p, y = np.asarray(probs).ravel(), np.asarray(y_true).ravel()
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if mask.any():
            rows.append({"bin": b, "mean_predicted": float(p[mask].mean()), "observed_rate": float(y[mask].mean()), "n": int(mask.sum())})
    return rows


def cross_fitted_systems(svm_scores, qarib_probs, y_true, folds=5, seed=42):
    """Out-of-fold calibrated probabilities and tuned-threshold predictions."""
    svm_x = calibration_input(svm_scores, "svm")
    qarib_x = calibration_input(qarib_probs, "qarib")
    out = {name: np.zeros_like(y_true, dtype=float) for name in ("svm_cal", "qarib_cal", "ensemble")}
    preds = {name: np.zeros_like(y_true) for name in ("svm_tuned", "qarib_tuned", "ensemble_tuned")}
    for fit_idx, eval_idx in KFold(folds, shuffle=True, random_state=seed).split(y_true):
        y_fit = y_true[fit_idx]
        svm_params = fit_platt(svm_x[fit_idx], y_fit)
        qarib_params = fit_platt(qarib_x[fit_idx], y_fit)
        fit_probs = {
            "svm_cal": apply_platt(svm_x[fit_idx], svm_params),
            "qarib_cal": apply_platt(qarib_x[fit_idx], qarib_params),
        }
        fit_probs["ensemble"] = ensemble_probs(fit_probs["svm_cal"], fit_probs["qarib_cal"])
        eval_probs = {
            "svm_cal": apply_platt(svm_x[eval_idx], svm_params),
            "qarib_cal": apply_platt(qarib_x[eval_idx], qarib_params),
        }
        eval_probs["ensemble"] = ensemble_probs(eval_probs["svm_cal"], eval_probs["qarib_cal"])
        for name in out:
            out[name][eval_idx] = eval_probs[name]
        for name, source in (("svm_tuned", "svm_cal"), ("qarib_tuned", "qarib_cal"), ("ensemble_tuned", "ensemble")):
            thresholds = tune_thresholds(fit_probs[source], y_fit)
            preds[name][eval_idx] = (eval_probs[source] >= thresholds).astype(int)
    return out, preds
