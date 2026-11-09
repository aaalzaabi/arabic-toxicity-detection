import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from aratox_nlp.config import ROOT
from aratox_nlp.constants import TOXIC_LABELS
from aratox_nlp.ethics import (
    CURRENT_THRESHOLD,
    POLICY_CHOICES,
    THRESHOLD_GRID,
    THRESHOLD_POLICY,
    operating_points,
    score_matrix,
    threshold_direction,
    threshold_tradeoff,
)

ETHICS = ROOT / "outputs" / "ethics"
FIGURES = ROOT / "outputs" / "figures"


def _line(path, thresholds, series, ylabel, title):
    figure, axis = plt.subplots(figsize=(7.4, 4.4))
    for name, values in series:
        axis.plot(thresholds, values, marker="o", label=name)
    axis.set_xlabel("Decision threshold")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    predictions = pd.read_csv(ROOT / "outputs" / "predictions" / "qarib_validation.csv")
    gold = predictions[list(TOXIC_LABELS)].astype(int).values
    scores = score_matrix(predictions, "qarib")
    overall, per_label = threshold_tradeoff(gold, scores, THRESHOLD_GRID, TOXIC_LABELS)
    ETHICS.mkdir(parents=True, exist_ok=True)
    overall_path = ETHICS / "qarib_threshold_tradeoff.csv"
    per_label_path = ETHICS / "qarib_threshold_tradeoff_per_label.csv"
    overall.to_csv(overall_path, index=False)
    per_label.to_csv(per_label_path, index=False)
    overall = pd.read_csv(overall_path)
    points = operating_points(overall, CURRENT_THRESHOLD)
    direction = threshold_direction(overall)
    (ETHICS / "qarib_operating_points.json").write_text(
        json.dumps({"points": points, "direction": direction, "note": THRESHOLD_POLICY, "policy": POLICY_CHOICES}, indent=2),
        encoding="utf-8",
    )
    _line(
        FIGURES / "qarib_threshold_precision_recall.png",
        overall.threshold.tolist(),
        [
            ("Macro precision", overall.macro_precision.tolist()),
            ("Macro recall", overall.macro_recall.tolist()),
        ],
        "Score",
        "QARiB validation threshold trade-off",
    )
    figure, axis = plt.subplots(figsize=(7.4, 4.4))
    axis.plot(overall.threshold, overall.total_false_positives, marker="o")
    axis.set_xlabel("Decision threshold")
    axis.set_ylabel("Total false positives")
    axis.set_title("QARiB validation false positives by threshold")
    figure.tight_layout()
    figure.savefig(FIGURES / "qarib_threshold_false_positives.png", dpi=180)
    plt.close(figure)
    print(overall.to_string(index=False))
    print(json.dumps(points, indent=2))
    print(json.dumps(direction, indent=2))


if __name__ == "__main__":
    main()
