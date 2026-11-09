"""Responsible-AI summaries and validation-only error calculations.

This module does not train models, read the held-out test split, or change a
decision threshold. Predicted dialect is not a toxicity feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from .comparison import QARIB_NAME, SVM_NAME, SVM_PREFIX, QARIB_PREFIX
from .constants import TOXIC_LABELS
from .metrics import multilabel_metrics

CURRENT_THRESHOLD = 0.5
THRESHOLD_GRID = tuple(round(0.30 + step * 0.05, 2) for step in range(11))

HUMAN_REVIEW = "Research demo — human review is required for consequential moderation decisions."
FALSE_POSITIVE_CONSEQUENCE = (
    "In a real moderation system, false positives could negatively affect users "
    "if predictions are acted on without human review."
)
DIALECT_CAVEAT = (
    "Dialect categories are produced by an independent classifier and are not "
    "ground-truth AraTox annotations. Any dialect-related fairness analysis is exploratory."
)
DIALECT_NOT_IDENTITY = (
    "The predicted dialect describes linguistic style and should not be interpreted "
    "as nationality, identity, or origin."
)
ANNOTATION_BIAS = (
    "Toxicity is partly context-dependent. Human annotators may differ in how they "
    "interpret insults, slang, sarcasm, cultural references, or acceptable speech. "
    "A model trained on these labels may reproduce inconsistencies or biases present in the annotations."
)
THRESHOLD_POLICY = (
    "Threshold selection depends on the intended moderation policy and the relative "
    "cost of false positives versus false negatives."
)
POLICY_CHOICES = "Different operating points reflect different policy choices."
EXPLORER_NOTE = "Validation analysis only — this does not change the deployed model."
F1_NOT_FEWEST_FALSE_POSITIVES = (
    "The model with the highest overall F1 is not necessarily the model with the "
    "fewest false positives for every category."
)
RESPONSIBLE_SUMMARY = (
    "Higher F1 does not automatically mean lower risk.\n\n"
    "For moderation systems, false positives matter because legitimate content may be incorrectly flagged.\n\n"
    "Threshold selection creates a trade-off between precision and recall.\n\n"
    "Dialect variation, slang, cultural expressions, and annotation choices may influence errors.\n\n"
    "This research demo should support human review, not replace it."
)
RESEARCH_CONCLUSION = (
    "The project evaluates not only predictive performance but also error behavior. "
    "QARiB achieves higher validation F1 than TF-IDF + SVM, but false-positive behavior "
    "varies by toxicity category. This demonstrates that model selection for moderation "
    "should consider precision, recall, false-positive risk, computational cost, and the "
    "intended consequences of classification."
)

REVIEW_FIELDS = (
    "error_type",
    "notes",
    "possible_slang",
    "possible_sarcasm",
    "possible_code_switching",
    "possible_reclaimed_expression",
    "possible_annotation_ambiguity",
    "possible_context_missing",
    "other",
)

FALSE_POSITIVE_COLUMNS = (
    "model",
    "label",
    "true_negatives",
    "false_positives",
    "false_positive_rate",
    "precision",
    "recall",
    "f1",
)


def false_positive_rate(false_positives: int, true_negatives: int) -> float:
    denominator = int(false_positives) + int(true_negatives)
    if denominator == 0:
        return 0.0
    return float(false_positives) / float(denominator)


def _label_error_row(model: str, label: str, gold: np.ndarray, pred: np.ndarray) -> dict:
    gold = np.asarray(gold).astype(int)
    pred = np.asarray(pred).astype(int)
    true_negatives = int(((gold == 0) & (pred == 0)).sum())
    false_positives = int(((gold == 0) & (pred == 1)).sum())
    return {
        "model": model,
        "label": label,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_positive_rate": false_positive_rate(false_positives, true_negatives),
        "precision": float(precision_score(gold, pred, zero_division=0)),
        "recall": float(recall_score(gold, pred, zero_division=0)),
        "f1": float(f1_score(gold, pred, zero_division=0)),
    }


def false_positive_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-label false-positive rates for the two saved toxicity models."""
    rows = []
    for model, prefix in ((SVM_NAME, SVM_PREFIX), (QARIB_NAME, QARIB_PREFIX)):
        for label in TOXIC_LABELS:
            rows.append(_label_error_row(
                model,
                label,
                frame[label].astype(int).values,
                frame[f"{prefix}_pred_{label}"].astype(int).values,
            ))
    table = pd.DataFrame(rows, columns=list(FALSE_POSITIVE_COLUMNS))
    if "dialect" in table.columns:
        raise ValueError("Predicted dialect must not appear in the toxicity false-positive table.")
    return table


def threshold_tradeoff(gold: np.ndarray, scores: np.ndarray, thresholds=THRESHOLD_GRID, labels=TOXIC_LABELS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply one shared threshold at a time to saved scores. This does not train."""
    gold = np.asarray(gold).astype(int)
    scores = np.asarray(scores, dtype=float)
    if gold.shape != scores.shape:
        raise ValueError("Gold labels and scores must have the same shape.")
    overall_rows = []
    per_label_rows = []
    for threshold in thresholds:
        pred = (scores >= float(threshold)).astype(int)
        metrics = multilabel_metrics(gold, pred, list(labels))
        false_positives = int(((pred == 1) & (gold == 0)).sum())
        false_negatives = int(((pred == 0) & (gold == 1)).sum())
        overall_rows.append({
            "threshold": float(threshold),
            "macro_f1": metrics["macro_f1"],
            "micro_f1": metrics["micro_f1"],
            "macro_precision": float(precision_score(gold, pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(gold, pred, average="macro", zero_division=0)),
            "total_false_positives": false_positives,
            "total_false_negatives": false_negatives,
            "subset_accuracy": metrics["subset_accuracy"],
        })
        for index, label in enumerate(labels):
            per_label_rows.append(_label_error_row(QARIB_NAME, label, gold[:, index], pred[:, index]) | {
                "threshold": float(threshold),
                "false_negatives": int(((pred[:, index] == 0) & (gold[:, index] == 1)).sum()),
            })
    overall = pd.DataFrame(overall_rows).sort_values("threshold").reset_index(drop=True)
    per_label = pd.DataFrame(per_label_rows)
    per_label = per_label[[
        "label", "threshold", "precision", "recall", "f1",
        "false_positives", "false_negatives", "false_positive_rate",
    ]].sort_values(["threshold", "label"]).reset_index(drop=True)
    return overall, per_label


def nearest_threshold_row(table: pd.DataFrame, value: float) -> pd.Series:
    ordered = table.sort_values("threshold").reset_index(drop=True)
    position = int((ordered.threshold - float(value)).abs().idxmin())
    return ordered.iloc[position]


def operating_points(table: pd.DataFrame, current: float = CURRENT_THRESHOLD) -> dict:
    ordered = table.sort_values("threshold").reset_index(drop=True)
    return {
        "highest_macro_f1": _point(ordered.loc[ordered.macro_f1.idxmax()]),
        "highest_macro_precision": _point(ordered.loc[ordered.macro_precision.idxmax()]),
        "lowest_false_positives": _point(ordered.loc[ordered.total_false_positives.idxmin()]),
        "current": _point(nearest_threshold_row(ordered, current)),
    }


def _point(row: pd.Series) -> dict:
    return {
        "threshold": float(row.threshold),
        "macro_f1": float(row.macro_f1),
        "macro_precision": float(row.macro_precision),
        "macro_recall": float(row.macro_recall),
        "total_false_positives": int(row.total_false_positives),
        "total_false_negatives": int(row.total_false_negatives),
    }


def threshold_direction(table: pd.DataFrame) -> dict:
    ordered = table.sort_values("threshold")
    first = ordered.iloc[0]
    last = ordered.iloc[-1]
    precision_change = float(last.macro_precision - first.macro_precision)
    recall_change = float(last.macro_recall - first.macro_recall)
    return {
        "from_threshold": float(first.threshold),
        "to_threshold": float(last.threshold),
        "precision_change": precision_change,
        "recall_change": recall_change,
        "precision_increases": precision_change > 0,
        "recall_decreases": recall_change < 0,
    }


def blank_review_frame(examples: pd.DataFrame) -> pd.DataFrame:
    """Attach empty manual-review fields. Nothing here is inferred automatically."""
    out = examples.copy()
    for field in REVIEW_FIELDS:
        if field not in out.columns:
            out[field] = ""
        else:
            out[field] = out[field].fillna("")
    return out


def score_matrix(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    return frame[[f"{prefix}_score_{label}" for label in TOXIC_LABELS]].astype(float).values
