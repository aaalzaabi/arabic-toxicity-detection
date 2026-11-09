import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from aratox_nlp.config import ROOT, load_yaml
from aratox_nlp.dialect import ArabicDialectClassifier, combine_with_toxicity, predictions_frame


def main():
    parser = argparse.ArgumentParser(description="Run the separate dialect classifier. This does not score AraTox test and does not change toxicity models.")
    parser.add_argument("--input", default="data/processed/validation.csv")
    parser.add_argument("--text-column", default="text_raw")
    parser.add_argument("--output", default=None)
    parser.add_argument("--toxicity-predictions", default=None)
    args = parser.parse_args()

    cfg = load_yaml("configs/dialect.yaml")["dialect"]
    source = Path(args.input)
    if not source.is_absolute():
        source = ROOT / source
    frame = pd.read_csv(source)
    if args.text_column not in frame.columns:
        raise SystemExit(f"Column {args.text_column!r} is not in {source}.")
    texts = frame[args.text_column].astype(str).tolist()
    classifier = ArabicDialectClassifier(
        model_name=cfg["model_name"],
        low_arabic_ratio_threshold=float(cfg["low_arabic_ratio_threshold"]),
        max_length=int(cfg["max_length"]),
        batch_size=int(cfg["batch_size"]),
    )
    rows = classifier.predict_batch(texts)
    out = predictions_frame(texts, rows)
    out.insert(0, "source_row", range(len(out)))
    destination = Path(args.output) if args.output else ROOT / "outputs" / "predictions" / f"dialect_{source.stem}.csv"
    if not destination.is_absolute():
        destination = ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(destination, index=False)
    print(f"Saved {len(out)} dialect predictions to {destination}")
    print(out.predicted_region.value_counts(dropna=False).to_string())

    toxicity_path = Path(args.toxicity_predictions) if args.toxicity_predictions else ROOT / "outputs" / "predictions" / "qarib_validation.csv"
    if source.stem == "validation" and toxicity_path.exists():
        prefix = load_yaml("configs/dialect.yaml")["dialect_analysis"]["toxicity_prefix"]
        combined = combine_with_toxicity(out, pd.read_csv(toxicity_path), prefix)
        combined_path = ROOT / "outputs" / "predictions" / "validation_with_predicted_dialect.csv"
        combined.to_csv(combined_path, index=False)
        print(f"Saved exploratory join to {combined_path}")
        print("These dialect columns are predictions, not AraTox ground truth.")


if __name__ == "__main__":
    main()
