"""Named QARiB threshold policies read from the saved sensitivity table.

Selecting a mode does not run the model and does not replace the reported 0.50 result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MODES = (
    {
        "mode": "Reported / Balanced",
        "threshold": 0.50,
        "interpretation": "Preserve the originally reported validation evaluation.",
    },
    {
        "mode": "F1-focused exploratory",
        "threshold": 0.65,
        "interpretation": "Highest Macro-F1 observed on the current validation threshold grid.",
    },
    {
        "mode": "Precision-focused review",
        "threshold": 0.80,
        "interpretation": "Reduce false positives and favor precision at the cost of recall.",
    },
)
MODE_NOTE = "These operating modes represent different moderation priorities. They do not change the trained QARiB model."
REPORTED_THRESHOLD = 0.50


def operating_mode_table(tradeoff: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in MODES:
        matched = tradeoff[np.isclose(tradeoff.threshold.astype(float), spec["threshold"])]
        if matched.empty:
            raise ValueError(f"Threshold {spec['threshold']:.2f} is missing from the saved trade-off table.")
        source = matched.iloc[0]
        rows.append({
            "mode": spec["mode"],
            "threshold": float(source.threshold),
            "macro_f1": float(source.macro_f1),
            "macro_precision": float(source.macro_precision),
            "macro_recall": float(source.macro_recall),
            "false_positives": int(source.total_false_positives),
            "false_negatives": int(source.total_false_negatives),
            "subset_accuracy": float(source.subset_accuracy),
            "interpretation": spec["interpretation"],
        })
    return pd.DataFrame(rows)


def mode_at(table: pd.DataFrame, threshold: float) -> pd.Series:
    matched = table[np.isclose(table.threshold.astype(float), float(threshold))]
    if matched.empty:
        raise ValueError(f"No operating mode at threshold {threshold:.2f}.")
    return matched.iloc[0]
