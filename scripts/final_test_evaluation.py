"""Score the held-out AraTox test split once, with frozen decisions.

Refuses to run if the frozen fingerprints changed or if test results already exist.
Writes outputs/final/test_results.json, test_summary.csv, test_bootstrap.csv,
outputs/predictions/final_test_predictions.csv and a summary figure.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from freeze_decisions import FINAL, FROZEN, fingerprints
from quantize_qarib import quantized

from aratox_nlp.bootstrap import compare_systems
from aratox_nlp.calibration import expected_calibration_error
from aratox_nlp.config import ROOT
from aratox_nlp.constants import TOXIC_LABELS
from aratox_nlp.data import load_processed
from aratox_nlp.evaluation import save_json
from aratox_nlp.inference import load_calibration, load_qarib, load_svm, qarib_probs, score_texts
from aratox_nlp.metrics import binary_metrics, multilabel_metrics

RESULTS = FINAL / "test_results.json"


def check_frozen(dry_run=False):
    if RESULTS.exists() and not dry_run:
        raise SystemExit(f"{RESULTS} exists: the test set has already been scored once. Not scoring it again.")
    if not FROZEN.is_file():
        raise SystemExit("Run scripts/freeze_decisions.py first.")
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    changed = [path for path, digest in fingerprints().items() if frozen["fingerprints"].get(path) != digest]
    if changed:
        raise SystemExit(f"Frozen files changed since freezing: {changed}")
    return frozen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run-on-validation", action="store_true", help="Run the whole pipeline on validation to check it works. Never touches test.")
    args = ap.parse_args()
    dry = args.dry_run_on_validation
    frozen = check_frozen(dry)
    split = "validation" if dry else "test"
    out_dir = FINAL / "dry_run_validation" if dry else FINAL
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "test_results.json"
    test = load_processed(split, ROOT / "data/processed")
    texts = test.text_norm.astype(str).tolist()
    y = test[TOXIC_LABELS].astype(int).values
    svm, qarib, calibration = load_svm(), load_qarib(), load_calibration()

    scores, preds = {}, {}
    for item in frozen["systems"]:
        key = item["key"]
        if key == "qarib_int8":
            tok, _, _ = qarib
            model = quantized(load_qarib(device=torch.device("cpu"))[1])
            scores[key] = qarib_probs((tok, model, torch.device("cpu")), texts)
            preds[key] = (scores[key] >= 0.5).astype(int)
        else:
            scores[key], preds[key], _ = score_texts(texts, item["system"], item["policy"], svm=svm, qarib=qarib, calibration=calibration)
        print("scored", key, flush=True)

    names = {item["key"]: f"{item['system']} ({item['policy'].lower()})" for item in frozen["systems"]}
    summary = []
    details = {}
    for key, pred in preds.items():
        ml = multilabel_metrics(y, pred, TOXIC_LABELS)
        binary = binary_metrics(test.binary_toxic.astype(int).values, (pred.sum(1) > 0).astype(int))
        details[key] = {"name": names[key], "multilabel": ml, "binary": binary}
        summary.append({
            "key": key,
            "system": names[key],
            "primary": key == frozen["primary_system_key"],
            "macro_f1": ml["macro_f1"],
            "micro_f1": ml["micro_f1"],
            "hamming_loss": ml["hamming_loss"],
            "subset_accuracy": ml["subset_accuracy"],
            "binary_macro_f1": binary["macro_f1"],
            "binary_toxic_f1": binary["toxic_f1"],
            "false_positives": int(((pred == 1) & (y == 0)).sum()),
            "false_negatives": int(((pred == 0) & (y == 1)).sum()),
            "n": len(test),
        })
    summary = pd.DataFrame(summary)

    by_name = {names[key]: pred for key, pred in preds.items()}
    reference = names[frozen["reference_system_key"]]
    boot = pd.DataFrame(compare_systems(y, by_name, reference=reference))
    primary_vs_qarib = pd.DataFrame(compare_systems(
        y,
        {names[frozen["primary_system_key"]]: preds[frozen["primary_system_key"]], names["qarib_reported"]: preds["qarib_reported"]},
        reference=names["qarib_reported"],
    ))

    ece = {
        "TF-IDF scaled score (uncalibrated)": expected_calibration_error(y, scores["svm_reported"])[0],
        "TF-IDF calibrated": expected_calibration_error(y, scores["svm_tuned"])[0],
        "QARiB sigmoid (uncalibrated)": expected_calibration_error(y, scores["qarib_reported"])[0],
        "QARiB calibrated": expected_calibration_error(y, scores["qarib_tuned"])[0],
        "Ensemble calibrated": expected_calibration_error(y, scores["ensemble_tuned"])[0],
    }

    out = test[["text_raw", "text_norm", "binary_toxic", *TOXIC_LABELS]].copy()
    for key in preds:
        for i, label in enumerate(TOXIC_LABELS):
            out[f"{key}_score_{label}"] = scores[key][:, i]
            out[f"{key}_pred_{label}"] = preds[key][:, i]
    out.to_csv(out_dir / "dry_run_predictions.csv" if dry else ROOT / "outputs/predictions/final_test_predictions.csv", index=False)
    summary.to_csv(out_dir / "test_summary.csv", index=False)
    boot.to_csv(out_dir / "test_bootstrap.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    shown = boot.set_index("system").loc[summary.system]
    err = np.vstack([shown.macro_f1 - shown.macro_f1_ci_low, shown.macro_f1_ci_high - shown.macro_f1])
    colors = ["#0f766e" if primary else "#64748b" for primary in summary.primary]
    ax.barh(summary.system, shown.macro_f1, xerr=err, color=colors, capsize=4)
    ax.set_xlim(max(0.0, float(shown.macro_f1_ci_low.min()) - 0.03), min(1.0, float(shown.macro_f1_ci_high.max()) + 0.01))
    ax.set_xlabel("Macro-F1 on held-out test (95% bootstrap CI)")
    ax.set_title("Final test evaluation — primary system in teal")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / "final_test_macro_f1.png" if dry else ROOT / "outputs/figures/final_test_macro_f1.png", dpi=180)
    plt.close(fig)

    save_json({
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "n": len(test),
        "frozen_at": frozen["frozen_at"],
        "primary_system": names[frozen["primary_system_key"]],
        "systems": details,
        "bootstrap_vs_reference": boot.to_dict(orient="records"),
        "primary_vs_qarib": primary_vs_qarib.to_dict(orient="records"),
        "expected_calibration_error": ece,
        "jais": frozen["jais"],
    }, results_path)
    print(summary.round(4).to_string(index=False))
    print(boot.round(4).to_string(index=False))
    print(json.dumps(ece, indent=2))


if __name__ == "__main__":
    main()
