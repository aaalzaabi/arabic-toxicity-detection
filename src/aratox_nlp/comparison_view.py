"""Presentation helpers for the TF-IDF vs QARiB comparison.

These functions only read saved tables. They do not load or run either model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .comparison import QARIB_NAME, SVM_NAME
from .constants import TOXIC_LABELS

METRIC_FILES = (
    "tfidf_vs_qarib_summary.csv",
    "tfidf_vs_qarib_per_label.csv",
    "error_counts_by_label.csv",
    "tfidf_vs_qarib_latency.csv",
    "tfidf_vs_qarib_disagreement_summary.json",
)
ERROR_FILES = ("tfidf_vs_qarib_disagreements.csv",)

EXAMPLE_SOURCES = (
    "Custom text",
    "QARiB correct / TF-IDF wrong",
    "TF-IDF correct / QARiB wrong",
    "Both wrong",
    "Both correct",
)
SOURCE_TO_GROUP = {
    "QARiB correct / TF-IDF wrong": "qarib_only_correct",
    "TF-IDF correct / QARiB wrong": "tfidf_only_correct",
    "Both wrong": "both_wrong",
    "Both correct": "both_correct",
}

TFIDF_SCORE_NOTE = "These are scaled SVM decision scores, not calibrated probabilities."
QARIB_SCORE_NOTE = "Sigmoid scores. Calibration is not assumed."
SCORE_MAGNITUDE_NOTE = (
    "Score magnitudes should not be compared directly between TF-IDF and QARiB "
    "because the two models produce scores using different mechanisms."
)
LATENCY_WARNING = (
    "The models ran on different execution backends, so this is a practical deployment "
    "measurement on this machine, not a hardware-normalized benchmark."
)
OVERALL_INTERPRETATION = (
    "QARiB performs better overall, but the margin is modest. "
    "This shows that the classical TF-IDF baseline remains strong."
)
FALSE_POSITIVE_INTERPRETATION = (
    "Higher overall F1 does not mean a model is better on every error type. "
    "TF-IDF produced fewer false positives for several categories."
)
DEVICE_LABELS = {"cpu": "CPU", "mps": "Apple MPS", "cuda": "CUDA"}


def missing_comparison_files(metrics_dir: Path, errors_dir: Path) -> list[str]:
    missing = [name for name in METRIC_FILES if not (Path(metrics_dir) / name).is_file()]
    missing.extend(name for name in ERROR_FILES if not (Path(errors_dir) / name).is_file())
    return missing


def read_table(path: Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    return pd.read_csv(file_path)


def read_json(path: Path) -> dict:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    return json.loads(file_path.read_text(encoding="utf-8"))


def model_metrics(summary: pd.DataFrame, model_name: str) -> pd.Series:
    rows = summary[summary.model == model_name]
    if rows.empty:
        raise ValueError(f"Saved summary has no row for {model_name}.")
    return rows.iloc[0]


def overall_improvement(tfidf_row: pd.Series, qarib_row: pd.Series) -> dict[str, float]:
    return {
        "macro_f1": float(qarib_row.macro_f1) - float(tfidf_row.macro_f1),
        "micro_f1": float(qarib_row.micro_f1) - float(tfidf_row.micro_f1),
    }


def disagreement_stats(summary: dict) -> dict[str, float]:
    both_correct = int(summary["both_correct"])
    qarib_only = int(summary["qarib_only_correct"])
    tfidf_only = int(summary["tfidf_only_correct"])
    both_wrong = int(summary["both_wrong"])
    n = int(summary["n"])
    disagree = qarib_only + tfidf_only + both_wrong
    return {
        "n": n,
        "both_correct": both_correct,
        "qarib_only_correct": qarib_only,
        "tfidf_only_correct": tfidf_only,
        "both_wrong": both_wrong,
        "disagree": disagree,
        "disagree_percent": (100.0 * disagree / n) if n else 0.0,
        "qarib_exact_advantage": qarib_only - tfidf_only,
    }


def key_finding_text(macro_qarib: float, macro_tfidf: float, largest_labels: list[str]) -> str:
    gains = " and ".join(largest_labels[:2])
    return (
        "QARiB achieved the best validation performance, with Macro-F1 of "
        f"{macro_qarib:.3f} compared with {macro_tfidf:.3f} for TF-IDF + SVM.\n\n"
        "QARiB improved F1 across all six toxicity categories, with the largest gains "
        f"for {gains}.\n\n"
        "However, TF-IDF + SVM remained highly competitive and was substantially faster "
        "in the measured environment."
    )


def sorted_label_improvements(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["f1_difference", "label"], ascending=[False, True]).reset_index(drop=True)


def per_label_interpretation(sorted_frame: pd.DataFrame) -> str:
    if len(sorted_frame) == len(TOXIC_LABELS) and bool((sorted_frame.f1_difference > 0).all()):
        largest = " and ".join(sorted_frame.head(2).label.tolist())
        smallest = " and ".join(sorted_frame.tail(2).sort_values("f1_difference").label.tolist())
        return (
            "QARiB improves all six toxicity labels. "
            f"The largest gains are for {largest}, while {smallest} show only small improvements."
        )
    names = ", ".join(
        f"{row.label} ({row.f1_difference:+.3f})" for row in sorted_frame.itertuples()
    )
    return f"Per-label F1 differences (QARiB minus TF-IDF): {names}."


def fewer_tfidf_false_positives(frame: pd.DataFrame) -> pd.DataFrame:
    chosen = frame[frame.tfidf_false_positives < frame.qarib_false_positives].copy()
    order = {label: index for index, label in enumerate(TOXIC_LABELS)}
    chosen["_order"] = chosen.label.map(order)
    return chosen.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def observed_latency_ratio(qarib_ms: float, tfidf_ms: float) -> float:
    if tfidf_ms <= 0:
        raise ValueError("TF-IDF latency must be positive.")
    return float(qarib_ms) / float(tfidf_ms)


def displayed_latency_ratio(qarib_ms: float, tfidf_ms: float, ndigits: int = 2) -> float:
    """Ratio of the milliseconds shown in the UI, after rounding for display."""
    return observed_latency_ratio(round(float(qarib_ms), ndigits), round(float(tfidf_ms), ndigits))


def device_label(device: str) -> str:
    return DEVICE_LABELS.get(str(device), str(device).upper())


def filter_validation_examples(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    if "comparison_group" not in frame.columns:
        raise ValueError("Disagreement file has no comparison_group column.")
    return frame[frame.comparison_group == group].reset_index(drop=True)


def step_example_index(index: int, count: int, step: int) -> int:
    if count <= 0:
        return 0
    return (int(index) + int(step)) % count


def positive_labels(value) -> set[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    text = str(value).strip()
    if text in {"", "none", "nan"}:
        return set()
    return {part for part in text.split("|") if part}


def example_label_view(row: pd.Series) -> list[dict]:
    gold = positive_labels(row.get("true_labels"))
    tfidf = positive_labels(row.get("tfidf_predicted_labels"))
    qarib = positive_labels(row.get("qarib_predicted_labels"))
    view = []
    for label in TOXIC_LABELS:
        tfidf_yes = label in tfidf
        qarib_yes = label in qarib
        gold_yes = label in gold
        view.append({
            "label": label,
            "gold": gold_yes,
            "tfidf": tfidf_yes,
            "qarib": qarib_yes,
            "tfidf_correct": tfidf_yes == gold_yes,
            "qarib_correct": qarib_yes == gold_yes,
            "models_differ": tfidf_yes != qarib_yes,
        })
    return view


def correctness_mark(correct: bool | None) -> str:
    if correct is None:
        return ""
    return "✓ correct" if correct else "✗ incorrect"


def yes_no(flag: bool) -> str:
    return "Yes" if flag else "No"


def binary_toxic(flags: dict[str, bool]) -> bool:
    return any(flags.values())
