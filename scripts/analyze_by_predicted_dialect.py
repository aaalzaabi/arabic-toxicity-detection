import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from aratox_nlp.config import ROOT, load_yaml
from aratox_nlp.constants import TOXIC_LABELS
from aratox_nlp.dialect import combine_with_toxicity
from aratox_nlp.metrics import multilabel_metrics


def group_rows(frame: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    rows = []
    pred_cols = [f"{prefix}_pred_{label}" for label in TOXIC_LABELS]
    binary_col = f"{prefix}_pred_binary"
    for name, group in frame.groupby(column, dropna=False):
        yt = group[TOXIC_LABELS].astype(int).values
        yp = group[pred_cols].astype(int).values
        metrics = multilabel_metrics(yt, yp, TOXIC_LABELS)
        gold = group.binary_toxic.astype(int)
        pred = group[binary_col].astype(int)
        label = "(missing)" if pd.isna(name) else str(name)
        rows.append({
            column: label,
            "n": int(len(group)),
            "binary_toxic_prevalence": float(gold.mean()) if len(group) else 0.0,
            "toxicity_macro_f1": metrics["macro_f1"],
            "toxicity_micro_f1": metrics["micro_f1"],
            "false_positives": int(((pred == 1) & (gold == 0)).sum()),
            "false_negatives": int(((pred == 0) & (gold == 1)).sum()),
        })
    table = pd.DataFrame(rows).sort_values("n", ascending=False)
    return table


def plot_groups(table: pd.DataFrame, column: str, minimum: int, path: Path):
    shown = table[table.n >= minimum]
    if shown.empty:
        return
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.barh(shown[column].astype(str), shown.toxicity_macro_f1)
    axis.set_xlabel("Toxicity macro-F1")
    axis.set_ylabel(column.replace("_", " "))
    axis.set_xlim(0, 1)
    axis.set_title(f"QARiB toxicity macro-F1 by {column.replace('_', ' ')} (n ≥ {minimum})")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Describe toxicity performance grouped by predicted dialect. This is not performance by true dialect.")
    parser.add_argument("--dialect-predictions", default="outputs/predictions/dialect_validation.csv")
    parser.add_argument("--toxicity-predictions", default="outputs/predictions/qarib_validation.csv")
    args = parser.parse_args()
    cfg = load_yaml("configs/dialect.yaml")["dialect_analysis"]
    prefix = cfg["toxicity_prefix"]
    minimum = int(cfg["minimum_group_size_for_plot"])
    dialect_path = Path(args.dialect_predictions)
    toxicity_path = Path(args.toxicity_predictions)
    if not dialect_path.is_absolute():
        dialect_path = ROOT / dialect_path
    if not toxicity_path.is_absolute():
        toxicity_path = ROOT / toxicity_path
    combined = combine_with_toxicity(pd.read_csv(dialect_path), pd.read_csv(toxicity_path), prefix)
    combined_path = ROOT / "outputs" / "predictions" / "validation_with_predicted_dialect.csv"
    combined.to_csv(combined_path, index=False)

    region = group_rows(combined, "predicted_region", prefix)
    country = group_rows(combined, "predicted_country_dialect", prefix)
    region["included_in_plot"] = region.n >= minimum
    country["included_in_plot"] = country.n >= minimum
    region_path = ROOT / "outputs" / "metrics" / "toxicity_by_predicted_region.csv"
    country_path = ROOT / "outputs" / "metrics" / "toxicity_by_predicted_country.csv"
    region.to_csv(region_path, index=False)
    country.to_csv(country_path, index=False)
    plot_groups(region, "predicted_region", minimum, ROOT / "outputs" / "figures" / "toxicity_by_predicted_region.png")
    plot_groups(country, "predicted_country_dialect", minimum, ROOT / "outputs" / "figures" / "toxicity_by_predicted_country.png")
    print("Toxicity performance grouped by predicted dialect, not by true dialect.")
    print(region.to_string(index=False))
    print(f"Saved {region_path}")
    print(f"Saved {country_path}")


if __name__ == "__main__":
    main()
