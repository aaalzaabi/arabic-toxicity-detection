"""Fit calibration, per-label thresholds and the ensemble on validation only.

Writes:
- outputs/metrics/calibration.json            fitted parameters + cross-fitted validation estimates
- outputs/metrics/bootstrap_validation.csv    95% paired-bootstrap CIs
- outputs/metrics/calibration_validation.csv  expected calibration error before/after
- outputs/figures/reliability_validation.png
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aratox_nlp.bootstrap import compare_systems
from aratox_nlp.calibration import (
    apply_platt,
    calibration_input,
    cross_fitted_systems,
    ensemble_probs,
    expected_calibration_error,
    fit_platt,
    reliability_curve,
    tune_thresholds,
)
from aratox_nlp.classical import decision_to_unit_interval
from aratox_nlp.config import ROOT, ensure_output_dirs
from aratox_nlp.constants import TOXIC_LABELS
from aratox_nlp.evaluation import save_json

PRED = ROOT / "outputs" / "predictions"
METRICS = ROOT / "outputs" / "metrics"
FIGURES = ROOT / "outputs" / "figures"
ENSEMBLE_WEIGHT = 0.5

SYSTEM_NAMES = {
    "svm_reported": "TF-IDF + Linear SVM (reported cutoff)",
    "qarib_reported": "QARiB (reported 0.50)",
    "svm_tuned": "TF-IDF + Linear SVM (calibrated, tuned cutoffs)",
    "qarib_tuned": "QARiB (calibrated, tuned cutoffs)",
    "ensemble_tuned": "Ensemble (calibrated, tuned cutoffs)",
}


def load_aligned():
    svm = pd.read_csv(PRED / "svm_validation.csv")
    qarib = pd.read_csv(PRED / "qarib_validation.csv")
    if len(svm) != len(qarib) or not (svm.text_norm.astype(str).values == qarib.text_norm.astype(str).values).all():
        raise ValueError("SVM and QARiB validation predictions are not row-aligned.")
    if not (svm[TOXIC_LABELS].values == qarib[TOXIC_LABELS].values).all():
        raise ValueError("SVM and QARiB validation files disagree on the true labels.")
    y = svm[TOXIC_LABELS].astype(int).values
    svm_scores = svm[[f"svm_score_{l}" for l in TOXIC_LABELS]].values.astype(float)
    qarib_scores = qarib[[f"qarib_score_{l}" for l in TOXIC_LABELS]].values.astype(float)
    return y, svm_scores, qarib_scores


def reliability_figure(y, curves, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    for name, probs in curves.items():
        rows = reliability_curve(y, probs)
        ax.plot([r["mean_predicted"] for r in rows], [r["observed_rate"] for r in rows], marker="o", label=name)
    ax.set_xlabel("Mean predicted score")
    ax.set_ylabel("Observed positive rate")
    ax.set_title("Reliability on validation (all labels pooled)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    ensure_output_dirs()
    y, svm_scores, qarib_scores = load_aligned()

    svm_x = calibration_input(svm_scores, "svm")
    qarib_x = calibration_input(qarib_scores, "qarib")
    svm_params = fit_platt(svm_x, y)
    qarib_params = fit_platt(qarib_x, y)
    svm_cal = apply_platt(svm_x, svm_params)
    qarib_cal = apply_platt(qarib_x, qarib_params)
    ens = ensemble_probs(svm_cal, qarib_cal, ENSEMBLE_WEIGHT)
    thresholds = {
        "svm": tune_thresholds(svm_cal, y),
        "qarib": tune_thresholds(qarib_cal, y),
        "ensemble": tune_thresholds(ens, y),
    }

    oof_probs, oof_preds = cross_fitted_systems(svm_scores, qarib_scores, y)
    predictions = {
        SYSTEM_NAMES["svm_reported"]: (svm_scores >= 0).astype(int),
        SYSTEM_NAMES["qarib_reported"]: (qarib_scores >= 0.5).astype(int),
        **{SYSTEM_NAMES[key]: pred for key, pred in oof_preds.items()},
    }
    ci = pd.DataFrame(compare_systems(y, predictions, reference=SYSTEM_NAMES["svm_reported"]))
    ci.insert(1, "estimate", ["validation" if "reported" in name else "validation, 5-fold cross-fitted" for name in ci.system])
    ci.to_csv(METRICS / "bootstrap_validation.csv", index=False)

    # The headline question: is QARiB's gap over TF-IDF larger than resampling noise?
    head_to_head = pd.DataFrame(compare_systems(
        y,
        {SYSTEM_NAMES["qarib_reported"]: predictions[SYSTEM_NAMES["qarib_reported"]], SYSTEM_NAMES["svm_reported"]: predictions[SYSTEM_NAMES["svm_reported"]]},
        reference=SYSTEM_NAMES["svm_reported"],
    ))

    ece_rows = []
    for name, raw, cal in (
        ("TF-IDF + Linear SVM", decision_to_unit_interval(svm_scores), oof_probs["svm_cal"]),
        ("QARiB", qarib_scores, oof_probs["qarib_cal"]),
    ):
        before, _ = expected_calibration_error(y, raw)
        after, per_label = expected_calibration_error(y, cal)
        ece_rows.append({"model": name, "ece_before": before, "ece_after_platt": after, **{f"ece_after_{k}": v for k, v in per_label.items()}})
    ece_rows.append({"model": "Ensemble", "ece_before": np.nan, "ece_after_platt": expected_calibration_error(y, oof_probs["ensemble"])[0]})
    pd.DataFrame(ece_rows).to_csv(METRICS / "calibration_validation.csv", index=False)
    reliability_figure(y, {
        "TF-IDF scaled score (before)": decision_to_unit_interval(svm_scores),
        "TF-IDF calibrated": oof_probs["svm_cal"],
        "QARiB sigmoid (before)": qarib_scores,
        "QARiB calibrated": oof_probs["qarib_cal"],
    }, FIGURES / "reliability_validation.png")

    cross_fitted = ci.set_index("system").macro_f1.to_dict()
    tuned_keys = ("svm_tuned", "qarib_tuned", "ensemble_tuned")
    selected = max(tuned_keys + ("svm_reported", "qarib_reported"), key=lambda key: cross_fitted[SYSTEM_NAMES[key]])
    calibration = {
        "fitted_on": "validation",
        "labels": TOXIC_LABELS,
        "svm": {"score_input": "decision_function", "platt": svm_params.tolist(), "thresholds_tuned": thresholds["svm"].tolist()},
        "qarib": {"score_input": "logit(sigmoid probability)", "platt": qarib_params.tolist(), "thresholds_tuned": thresholds["qarib"].tolist()},
        "ensemble": {"weight": ENSEMBLE_WEIGHT, "description": "weight * calibrated TF-IDF + (1 - weight) * calibrated QARiB", "thresholds_tuned": thresholds["ensemble"].tolist()},
        "cross_fitted_validation_macro_f1": {SYSTEM_NAMES[k]: cross_fitted[SYSTEM_NAMES[k]] for k in SYSTEM_NAMES},
        "selected_system_key": selected,
        "selected_system": SYSTEM_NAMES[selected],
        "selection_rule": "Highest Macro-F1 on validation; tuned systems use 5-fold cross-fitted estimates so the threshold search does not see the rows it is scored on.",
        "qarib_vs_tfidf_validation": head_to_head[head_to_head.system == SYSTEM_NAMES["qarib_reported"]].iloc[0].to_dict(),
    }
    save_json(calibration, METRICS / "calibration.json")
    print(ci.to_string(index=False))
    print(pd.DataFrame(ece_rows)[["model", "ece_before", "ece_after_platt"]].to_string(index=False))
    print("Selected system:", calibration["selected_system"])
    print(json.dumps({label: [round(t, 2) for t in thresholds[k]] for k, label in (("svm", "svm"), ("qarib", "qarib"), ("ensemble", "ensemble"))}))


if __name__ == "__main__":
    main()
