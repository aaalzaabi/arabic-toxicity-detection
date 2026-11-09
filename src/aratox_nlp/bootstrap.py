"""Paired bootstrap confidence intervals for multi-label F1.

All systems are resampled with the same row weights, so differences between
systems are paired. Per-row TP/FP/FN counts make each resample a matrix product.
"""

from __future__ import annotations

import numpy as np


def _row_counts(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return y_true & y_pred, (1 - y_true) & y_pred, y_true & (1 - y_pred)


def _f1_from_counts(tp, fp, fn):
    denom = 2 * tp + fp + fn
    per_label = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
    macro = per_label.mean(axis=-1)
    tp_s, fp_s, fn_s = tp.sum(axis=-1), fp.sum(axis=-1), fn.sum(axis=-1)
    micro_denom = 2 * tp_s + fp_s + fn_s
    micro = np.divide(2 * tp_s, micro_denom, out=np.zeros_like(tp_s, dtype=float), where=micro_denom > 0)
    return macro, micro


def resample_weights(n_rows, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_rows, np.full(n_rows, 1 / n_rows), size=n_boot)


def bootstrap_scores(y_true, y_pred, weights):
    tp, fp, fn = _row_counts(y_true, y_pred)
    return _f1_from_counts(weights @ tp, weights @ fp, weights @ fn)


def interval(values, level=0.95):
    lo, hi = np.quantile(values, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def point_scores(y_true, y_pred):
    tp, fp, fn = _row_counts(y_true, y_pred)
    macro, micro = _f1_from_counts(tp.sum(axis=0), fp.sum(axis=0), fn.sum(axis=0))
    return float(macro), float(micro)


def compare_systems(y_true, predictions, reference, n_boot=2000, seed=42):
    """CI for each system's Macro/Micro-F1 and for its Macro-F1 gap to `reference`."""
    weights = resample_weights(len(y_true), n_boot, seed)
    boots = {name: bootstrap_scores(y_true, pred, weights) for name, pred in predictions.items()}
    rows = []
    for name, pred in predictions.items():
        macro, micro = point_scores(y_true, pred)
        row = {
            "system": name,
            "macro_f1": macro,
            "macro_f1_ci_low": interval(boots[name][0])[0],
            "macro_f1_ci_high": interval(boots[name][0])[1],
            "micro_f1": micro,
            "micro_f1_ci_low": interval(boots[name][1])[0],
            "micro_f1_ci_high": interval(boots[name][1])[1],
        }
        if name != reference:
            diff = boots[name][0] - boots[reference][0]
            row["macro_f1_diff_vs_reference"] = macro - point_scores(y_true, predictions[reference])[0]
            row["diff_ci_low"], row["diff_ci_high"] = interval(diff)
            row["share_of_resamples_not_better"] = float((diff <= 0).mean())
        rows.append(row)
    return rows
