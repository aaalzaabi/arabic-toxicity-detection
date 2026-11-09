"""Paired comparison of TF-IDF + Linear SVM and QARiB on the same rows.

Scores are computed from prediction files. This module does not retrain models,
change thresholds, or read the held-out test split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .constants import TOXIC_LABELS
from .metrics import binary_metrics, multilabel_metrics

SVM_PREFIX = "svm"
QARIB_PREFIX = "qarib"
SVM_NAME = "TF-IDF + Linear SVM"
QARIB_NAME = "QARiB"


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def align_predictions(svm: pd.DataFrame, qarib: pd.DataFrame) -> pd.DataFrame:
    """Return one frame only when both files describe the same validation rows."""
    if len(svm) != len(qarib):
        raise ValueError(f"Row counts differ: TF-IDF has {len(svm)}, QARiB has {len(qarib)}.")
    identity = ["text_raw", "text_norm", "binary_toxic", *TOXIC_LABELS]
    _require_columns(svm, identity, "TF-IDF predictions")
    _require_columns(qarib, identity, "QARiB predictions")
    for column in identity:
        left = svm[column].astype(str).tolist()
        right = qarib[column].astype(str).tolist()
        if left != right:
            mismatches = sum(a != b for a, b in zip(left, right))
            raise ValueError(f"Column {column} is not aligned ({mismatches} mismatched rows).")
    for prefix, frame, name in ((SVM_PREFIX, svm, "TF-IDF"), (QARIB_PREFIX, qarib, "QARiB")):
        needed = [f"{prefix}_pred_{label}" for label in TOXIC_LABELS] + [f"{prefix}_pred_binary"]
        _require_columns(frame, needed, name)
    aligned = svm.copy()
    for label in TOXIC_LABELS:
        aligned[f"qarib_pred_{label}"] = qarib[f"qarib_pred_{label}"].astype(int).values
        score_col = f"qarib_score_{label}"
        if score_col in qarib.columns:
            aligned[score_col] = qarib[score_col].values
    aligned["qarib_pred_binary"] = qarib["qarib_pred_binary"].astype(int).values
    for label in TOXIC_LABELS:
        aligned[f"svm_pred_{label}"] = aligned[f"svm_pred_{label}"].astype(int)
        aligned[label] = aligned[label].astype(int)
    aligned["binary_toxic"] = aligned["binary_toxic"].astype(int)
    aligned["svm_pred_binary"] = aligned["svm_pred_binary"].astype(int)
    return aligned.reset_index(drop=True)


def _pred_matrix(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    return frame[[f"{prefix}_pred_{label}" for label in TOXIC_LABELS]].astype(int).values


def summary_row(frame: pd.DataFrame, prefix: str, model_name: str) -> dict:
    gold = frame[TOXIC_LABELS].astype(int).values
    pred = _pred_matrix(frame, prefix)
    multi = multilabel_metrics(gold, pred, TOXIC_LABELS)
    binary = binary_metrics(frame.binary_toxic.astype(int).values, frame[f"{prefix}_pred_binary"].astype(int).values)
    return {
        "model": model_name,
        "macro_f1": multi["macro_f1"],
        "micro_f1": multi["micro_f1"],
        "weighted_f1": multi["weighted_f1"],
        "hamming_loss": multi["hamming_loss"],
        "subset_accuracy": multi["subset_accuracy"],
        "binary_macro_f1": binary["macro_f1"],
        "binary_toxic_f1": binary["toxic_f1"],
        "n": int(len(frame)),
    }


def summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        summary_row(frame, SVM_PREFIX, SVM_NAME),
        summary_row(frame, QARIB_PREFIX, QARIB_NAME),
    ])


def per_label_table(frame: pd.DataFrame) -> pd.DataFrame:
    gold = frame[TOXIC_LABELS].astype(int).values
    svm_metrics = multilabel_metrics(gold, _pred_matrix(frame, SVM_PREFIX), TOXIC_LABELS)["per_label"]
    qarib_metrics = multilabel_metrics(gold, _pred_matrix(frame, QARIB_PREFIX), TOXIC_LABELS)["per_label"]
    rows = []
    for label in TOXIC_LABELS:
        svm = svm_metrics[label]
        qarib = qarib_metrics[label]
        rows.append({
            "label": label,
            "tfidf_precision": svm["precision"],
            "tfidf_recall": svm["recall"],
            "tfidf_f1": svm["f1"],
            "qarib_precision": qarib["precision"],
            "qarib_recall": qarib["recall"],
            "qarib_f1": qarib["f1"],
            "f1_difference": qarib["f1"] - svm["f1"],
        })
    return pd.DataFrame(rows)


def _positive_names(values: np.ndarray) -> str:
    names = [label for label, value in zip(TOXIC_LABELS, values) if int(value) == 1]
    return "|".join(names) if names else "none"


def disagreement_frame(frame: pd.DataFrame) -> pd.DataFrame:
    gold = frame[TOXIC_LABELS].astype(int).values
    svm = _pred_matrix(frame, SVM_PREFIX)
    qarib = _pred_matrix(frame, QARIB_PREFIX)
    svm_exact = (svm == gold).all(axis=1)
    qarib_exact = (qarib == gold).all(axis=1)
    groups = np.full(len(frame), "both_wrong", dtype=object)
    groups[svm_exact & qarib_exact] = "both_correct"
    groups[svm_exact & ~qarib_exact] = "tfidf_only_correct"
    groups[~svm_exact & qarib_exact] = "qarib_only_correct"
    out = pd.DataFrame({
        "text": frame["text_raw"].astype(str).values,
        "true_labels": [_positive_names(row) for row in gold],
        "tfidf_predicted_labels": [_positive_names(row) for row in svm],
        "qarib_predicted_labels": [_positive_names(row) for row in qarib],
        "tfidf_exact_correct": svm_exact,
        "qarib_exact_correct": qarib_exact,
        "comparison_group": groups,
    })
    for label in TOXIC_LABELS:
        for prefix in (SVM_PREFIX, QARIB_PREFIX):
            score_col = f"{prefix}_score_{label}"
            if score_col in frame.columns:
                out[score_col] = frame[score_col].values
    return out


def disagreement_summary(disagreements: pd.DataFrame) -> dict:
    counts = disagreements["comparison_group"].value_counts()
    return {
        "n": int(len(disagreements)),
        "both_correct": int(counts.get("both_correct", 0)),
        "qarib_only_correct": int(counts.get("qarib_only_correct", 0)),
        "tfidf_only_correct": int(counts.get("tfidf_only_correct", 0)),
        "both_wrong": int(counts.get("both_wrong", 0)),
    }


def focused_examples(disagreements: pd.DataFrame, group: str) -> pd.DataFrame:
    chosen = disagreements[disagreements.comparison_group == group].copy()
    chosen["error_type"] = ""
    chosen["notes"] = ""
    return chosen.reset_index(drop=True)


def wins_by_label(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in TOXIC_LABELS:
        gold = frame[label].astype(int).values
        svm_right = frame[f"svm_pred_{label}"].astype(int).values == gold
        qarib_right = frame[f"qarib_pred_{label}"].astype(int).values == gold
        rows.append({
            "label": label,
            "qarib_only_correct": int((qarib_right & ~svm_right).sum()),
            "tfidf_only_correct": int((svm_right & ~qarib_right).sum()),
            "both_correct": int((svm_right & qarib_right).sum()),
            "both_wrong": int((~svm_right & ~qarib_right).sum()),
        })
    return pd.DataFrame(rows)


def error_counts_by_label(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in TOXIC_LABELS:
        gold = frame[label].astype(int).values
        svm = frame[f"svm_pred_{label}"].astype(int).values
        qarib = frame[f"qarib_pred_{label}"].astype(int).values
        rows.append({
            "label": label,
            "tfidf_false_positives": int(((svm == 1) & (gold == 0)).sum()),
            "qarib_false_positives": int(((qarib == 1) & (gold == 0)).sum()),
            "tfidf_false_negatives": int(((svm == 0) & (gold == 1)).sum()),
            "qarib_false_negatives": int(((qarib == 0) & (gold == 1)).sum()),
        })
    return pd.DataFrame(rows)


def mcnemar_binary(frame: pd.DataFrame) -> dict:
    """Exact binomial McNemar test on binary toxic/non-toxic decisions."""
    gold = frame.binary_toxic.astype(int).values
    svm_correct = frame.svm_pred_binary.astype(int).values == gold
    qarib_correct = frame.qarib_pred_binary.astype(int).values == gold
    n01 = int((~svm_correct & qarib_correct).sum())
    n10 = int((svm_correct & ~qarib_correct).sum())
    discordant = n01 + n10
    if discordant == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = float((n01 - n10) ** 2 / discordant)
        p_value = float(binomtest(n01, discordant, 0.5, alternative="two-sided").pvalue)
    return {
        "n01": n01,
        "n10": n10,
        "statistic": statistic,
        "p_value": p_value,
        "method": "exact binomial McNemar test on binary toxic/non-toxic decisions",
        "definition": "n01 = TF-IDF wrong and QARiB correct; n10 = TF-IDF correct and QARiB wrong. statistic is (n01-n10)^2/(n01+n10). p_value is the two-sided exact binomial test.",
    }
