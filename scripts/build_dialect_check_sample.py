"""Create the blank dialect spot-check sheet (stratified by predicted region).

Fill it in on the app's Dialect Analysis tab, or edit human_region in the CSV.
Refuses to overwrite a sheet that already has human labels.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aratox_nlp.config import ROOT, ensure_output_dirs
from aratox_nlp.dialect_check import build_sample, labelled

SOURCE = ROOT / "outputs/predictions/validation_with_predicted_dialect.csv"
DEST = ROOT / "outputs/ethics/dialect_check_sample.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-region", type=int, default=17)
    ap.add_argument("--force", action="store_true", help="Overwrite even if rows are already labelled.")
    args = ap.parse_args()
    ensure_output_dirs()
    if DEST.is_file() and not args.force and len(labelled(pd.read_csv(DEST, dtype=str).fillna(""))):
        raise SystemExit(f"{DEST} already has human labels. Use --force to replace it.")
    sample = build_sample(pd.read_csv(SOURCE), per_region=args.per_region)
    sample.to_csv(DEST, index=False)
    print(f"Saved {len(sample)} rows to {DEST}")
    print(sample.predicted_region.value_counts().to_string())


if __name__ == "__main__":
    main()
