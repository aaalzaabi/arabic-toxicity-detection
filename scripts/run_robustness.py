"""Stress-test the models with meaning-preserving perturbations (validation only).

Each perturbation is applied to the raw comment, then the normal preprocessing
runs, as it would in the app. Reports Macro-F1 change and decision flip rates.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aratox_nlp.config import ROOT, ensure_output_dirs, load_yaml
from aratox_nlp.constants import TOXIC_LABELS
from aratox_nlp.data import load_processed
from aratox_nlp.inference import (
    ENSEMBLE_SYSTEM,
    QARIB_SYSTEM,
    REPORTED_POLICY,
    SVM_SYSTEM,
    TUNED_POLICY,
    load_calibration,
    load_qarib,
    load_svm,
    score_texts,
)
from aratox_nlp.metrics import multilabel_metrics
from aratox_nlp.preprocessing import config_from_dict, normalize_arabic
from aratox_nlp.robustness import PERTURBATIONS

SYSTEMS = ((SVM_SYSTEM, REPORTED_POLICY), (QARIB_SYSTEM, REPORTED_POLICY), (ENSEMBLE_SYSTEM, TUNED_POLICY))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    ensure_output_dirs()
    prep = config_from_dict(load_yaml("configs/data.yaml")["preprocessing"])
    df = load_processed("validation", ROOT / "data/processed").sample(n=args.n, random_state=args.seed)
    y = df[TOXIC_LABELS].astype(int).values
    raw = df.text_raw.astype(str).tolist()
    svm, qarib, calibration = load_svm(), load_qarib(), load_calibration()

    variants = {"Original": raw, **{name: [fn(text, args.seed) for text in raw] for name, fn in PERTURBATIONS.items()}}
    rows = []
    for system, policy in SYSTEMS:
        clean = None
        for name, texts in variants.items():
            norm = [normalize_arabic(text, prep) for text in texts]
            _, pred, _ = score_texts(norm, system, policy, svm=svm, qarib=qarib, calibration=calibration)
            if clean is None:
                clean = pred
            f1 = multilabel_metrics(y, pred, TOXIC_LABELS)["macro_f1"]
            rows.append({
                "system": f"{system} ({policy.lower()})",
                "perturbation": name,
                "macro_f1": f1,
                "binary_flip_rate": float(((pred.sum(1) > 0) != (clean.sum(1) > 0)).mean()),
                "label_set_change_rate": float((pred != clean).any(1).mean()),
                "normalization_undoes_it": bool(np.all([a == b for a, b in zip(norm, df.text_norm.astype(str))])),
                "n": len(df),
            })
            print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    base = table[table.perturbation == "Original"].set_index("system").macro_f1
    table["macro_f1_change"] = table.macro_f1 - table.system.map(base)
    table.to_csv(ROOT / "outputs/metrics/robustness_validation.csv", index=False)

    shown = table[table.perturbation != "Original"]
    pivot = shown.pivot(index="perturbation", columns="system", values="binary_flip_rate").loc[list(PERTURBATIONS)]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    pivot.plot.barh(ax=ax)
    ax.set_xlabel("Share of comments whose toxic / non-toxic decision flipped")
    ax.set_title(f"Robustness to perturbations (validation sample, n={len(df)})")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs/figures/robustness_flip_rate.png", dpi=180)
    plt.close(fig)
    print(shown.pivot(index="perturbation", columns="system", values="macro_f1_change").round(3).to_string())


if __name__ == "__main__":
    main()
