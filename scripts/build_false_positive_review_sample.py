import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aratox_nlp.comparison import align_predictions
from aratox_nlp.config import ROOT
from aratox_nlp.error_review import (
    REVIEW_COLUMNS,
    attach_predicted_dialect,
    false_positive_events,
    false_positives_by_predicted_region,
    review_frame,
    reviewed_rows,
    stratified_review_sample,
)
from aratox_nlp.operating_modes import operating_mode_table

ETHICS = ROOT / "outputs" / "ethics"
FIGURES = ROOT / "outputs" / "figures"
REVIEW_PATH = ETHICS / "manual_false_positive_review.csv"


def plot_modes(table: pd.DataFrame, path: Path):
    figure, axis = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(len(table))
    width = 0.25
    series = (
        ("Macro precision", table.macro_precision.tolist()),
        ("Macro recall", table.macro_recall.tolist()),
        ("Macro-F1", table.macro_f1.tolist()),
    )
    for index, (name, values) in enumerate(series):
        axis.bar(x + (index - 1) * width, values, width, label=name)
    axis.set_xticks(x, [f"{mode}\n{threshold:.2f}" for mode, threshold in zip(table["mode"], table.threshold)])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_xlabel("Operating policy")
    axis.set_title("QARiB validation operating policies")
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    ETHICS.mkdir(parents=True, exist_ok=True)
    tradeoff = pd.read_csv(ETHICS / "qarib_threshold_tradeoff.csv")
    modes = operating_mode_table(tradeoff)
    modes.to_csv(ETHICS / "qarib_operating_modes.csv", index=False)
    plot_modes(pd.read_csv(ETHICS / "qarib_operating_modes.csv"), FIGURES / "qarib_operating_modes.png")

    aligned = align_predictions(
        pd.read_csv(ROOT / "outputs" / "predictions" / "svm_validation.csv"),
        pd.read_csv(ROOT / "outputs" / "predictions" / "qarib_validation.csv"),
    )
    events = false_positive_events(aligned)
    dialect_path = ROOT / "outputs" / "predictions" / "dialect_validation.csv"
    dialect = pd.read_csv(dialect_path) if dialect_path.is_file() else None
    if dialect is not None:
        false_positives_by_predicted_region(events, dialect).to_csv(
            ETHICS / "false_positive_by_predicted_region.csv", index=False
        )
    if REVIEW_PATH.is_file():
        existing = pd.read_csv(REVIEW_PATH, dtype=str).fillna("")
        if not reviewed_rows(existing).empty:
            print(f"Kept existing reviewed file: {REVIEW_PATH}")
        else:
            _write_sample(events, dialect)
    else:
        _write_sample(events, dialect)
    print(modes.to_string(index=False))
    if REVIEW_PATH.is_file():
        saved = pd.read_csv(REVIEW_PATH)
        print(saved.trigger_label.value_counts().to_string())
        print("rows", len(saved), "reviewed", int((saved.reviewed.fillna("") == "yes").sum()) if "reviewed" in saved.columns else 0)


def _write_sample(events, dialect):
    sample = stratified_review_sample(events, size=100, seed=42)
    sample = attach_predicted_dialect(sample, dialect)
    review = review_frame(sample)
    if list(review.columns) != list(REVIEW_COLUMNS):
        raise RuntimeError("Review schema changed.")
    if review.review_category.ne("").any() or review.reviewed.ne("").any():
        raise RuntimeError("Review categories were assigned automatically.")
    review.to_csv(REVIEW_PATH, index=False)


if __name__ == "__main__":
    main()
