"""Choose the Jais-2 prompt mode on validation, freeze it, then score test once.

  python scripts/jais_final.py select   # after both validation runs
  python scripts/jais_final.py test     # runs the frozen mode on the test subset, once

Unparseable Jais outputs count as "no label" (all zeros), so a parse failure is
never silently dropped from the score. Metrics on parsed rows are reported too.
The comparison with the other systems uses the same test rows.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aratox_nlp.bootstrap import compare_systems
from aratox_nlp.config import ROOT
from aratox_nlp.constants import TOXIC_LABELS
from aratox_nlp.evaluation import save_json
from aratox_nlp.metrics import binary_metrics, multilabel_metrics

PRED = ROOT / "outputs" / "predictions"
FINAL = ROOT / "outputs" / "final"
FROZEN = FINAL / "jais_frozen.json"
RESULTS = FINAL / "jais_test_results.json"
N_ITEMS = 500
MODES = {"zeroshot": "Jais-2 zero-shot", "fewshot": "Jais-2 few-shot"}


def score_file(path):
    df = pd.read_csv(path)
    parsed = df.parse_error.fillna("").astype(str) == ""
    y = df[TOXIC_LABELS].astype(int).values
    pred = df[[f"jais_pred_{l}" for l in TOXIC_LABELS]].fillna(0).astype(int).values
    ml = multilabel_metrics(y, pred, TOXIC_LABELS)
    ml_parsed = multilabel_metrics(y[parsed.values], pred[parsed.values], TOXIC_LABELS) if parsed.any() else None
    binary = binary_metrics(df.binary_toxic.astype(int).values, (pred.sum(1) > 0).astype(int))
    return df, y, pred, {
        "n": len(df),
        "parse_failure_rate": float((~parsed).mean()),
        "macro_f1": ml["macro_f1"],
        "micro_f1": ml["micro_f1"],
        "macro_f1_parsed_rows_only": None if ml_parsed is None else ml_parsed["macro_f1"],
        "binary_macro_f1": binary["macro_f1"],
        "mean_latency_seconds": float(df.latency_seconds.mean()),
        "median_latency_seconds": float(df.latency_seconds.median()),
        "per_label": ml["per_label"],
    }


def select():
    if RESULTS.exists():
        raise SystemExit("Jais test results already exist; the mode cannot be re-selected.")
    rows = {}
    for mode, name in MODES.items():
        path = PRED / f"jais_{mode}_validation.csv"
        if not path.is_file():
            raise SystemExit(f"Missing {path}. Run both validation runs first.")
        rows[mode] = score_file(path)[3]
    table = pd.DataFrame([{"mode": MODES[m], **{k: v for k, v in r.items() if k != "per_label"}} for m, r in rows.items()])
    table.to_csv(ROOT / "outputs/metrics/jais_validation_summary.csv", index=False)
    chosen = max(rows, key=lambda m: rows[m]["macro_f1"])
    FINAL.mkdir(parents=True, exist_ok=True)
    save_json({
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "chosen_mode": chosen,
        "chosen_name": MODES[chosen],
        "rule": f"Higher Macro-F1 on the first {N_ITEMS} validation rows (the validation split is shuffled); parse failures count as no label.",
        "validation": rows,
    }, FROZEN)
    print(table.round(4).to_string(index=False))
    print("Chosen:", MODES[chosen])


def test():
    if RESULTS.exists():
        raise SystemExit(f"{RESULTS} exists: Jais has already been scored on test once.")
    if not FROZEN.is_file():
        raise SystemExit("Run `python scripts/jais_final.py select` first.")
    if not (ROOT / "outputs/predictions/final_test_predictions.csv").is_file():
        raise SystemExit("Run scripts/final_test_evaluation.py first so the other systems can be compared on the same rows.")
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    mode = frozen["chosen_mode"]
    # The official test file is ordered by label, so a head() slice is not representative: score all of it.
    cmd = [sys.executable, str(ROOT / "scripts/run_jais.py"), "--backend", "transformers", "--split", "test"]
    if mode == "fewshot":
        cmd.append("--few-shot")
    subprocess.run(cmd, check=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})

    df, y, jais_pred, jais_metrics = score_file(PRED / f"jais_{mode}_test.csv")
    others = pd.read_csv(ROOT / "outputs/predictions/final_test_predictions.csv")
    if len(others) != len(df):
        raise SystemExit(f"Expected the full test split ({len(others)} rows), got {len(df)} Jais rows.")
    if not (others.text_norm.astype(str).values == df.text_norm.astype(str).values).all():
        raise SystemExit("Jais test rows are not aligned with the frozen test predictions.")
    systems = json.loads((FINAL / "frozen_decisions.json").read_text(encoding="utf-8"))["systems"]
    predictions = {MODES[mode]: jais_pred}
    for item in systems:
        name = f"{item['system']} ({item['policy'].lower()})"
        predictions[name] = others[[f"{item['key']}_pred_{l}" for l in TOXIC_LABELS]].astype(int).values
    boot = pd.DataFrame(compare_systems(y, predictions, reference=MODES[mode]))
    boot.to_csv(FINAL / "jais_test_comparison.csv", index=False)
    save_json({
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "mode": MODES[mode],
        "n": len(df),
        "subset": f"Full held-out test split ({len(df):,} rows); every system is compared on these same rows. An earlier run on the first 500 test rows was invalid (the test file is ordered by label) and is kept in outputs/final/invalid_first500_test/.",
        "jais": jais_metrics,
        "comparison_same_rows": boot.to_dict(orient="records"),
    }, RESULTS)
    print(boot[["system", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high"]].round(4).to_string(index=False))
    print(f"Jais parse failures: {jais_metrics['parse_failure_rate']:.1%}; mean latency {jais_metrics['mean_latency_seconds']:.2f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["select", "test"])
    {"select": select, "test": test}[ap.parse_args().step]()
