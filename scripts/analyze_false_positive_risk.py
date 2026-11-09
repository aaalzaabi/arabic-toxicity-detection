import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aratox_nlp.comparison import QARIB_NAME, SVM_NAME, align_predictions
from aratox_nlp.config import ROOT
from aratox_nlp.constants import TOXIC_LABELS
from aratox_nlp.ethics import blank_review_frame, false_positive_summary

ETHICS = ROOT / "outputs" / "ethics"
FIGURES = ROOT / "outputs" / "figures"


def plot_false_positive_rates(table: pd.DataFrame, path: Path):
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(TOXIC_LABELS))
    width = 0.36
    for index, model in enumerate((SVM_NAME, QARIB_NAME)):
        subset = table[table.model == model].set_index("label").loc[list(TOXIC_LABELS)]
        offset = (index - 0.5) * width
        axis.bar(x + offset, subset.false_positive_rate.tolist(), width, label=model)
    axis.set_xticks(x, list(TOXIC_LABELS))
    axis.set_ylabel("False-positive rate")
    axis.set_xlabel("Toxicity label")
    axis.set_title("Validation false-positive rate by label")
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    svm = pd.read_csv(ROOT / "outputs" / "predictions" / "svm_validation.csv")
    qarib = pd.read_csv(ROOT / "outputs" / "predictions" / "qarib_validation.csv")
    aligned = align_predictions(svm, qarib)
    summary = false_positive_summary(aligned)
    ETHICS.mkdir(parents=True, exist_ok=True)
    summary_path = ETHICS / "false_positive_summary.csv"
    summary.to_csv(summary_path, index=False)
    plot_false_positive_rates(pd.read_csv(summary_path), FIGURES / "false_positive_rate_by_label.png")

    review_rows = []
    for model, prefix in ((SVM_NAME, "svm"), (QARIB_NAME, "qarib")):
        for label in TOXIC_LABELS:
            mask = (aligned[label].astype(int) == 0) & (aligned[f"{prefix}_pred_{label}"].astype(int) == 1)
            chosen = aligned.loc[mask, ["text_raw"]].copy()
            chosen = chosen.rename(columns={"text_raw": "text"})
            chosen.insert(0, "model", model)
            chosen.insert(2, "label", label)
            review_rows.append(chosen)
    review = blank_review_frame(pd.concat(review_rows, ignore_index=True))
    review.to_csv(ETHICS / "false_positive_review_queue.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
