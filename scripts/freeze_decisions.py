"""Freeze every decision before the held-out test set is scored.

Records SHA-256 fingerprints of the models, calibration and configs, the
systems that will be scored and the pre-registered primary system.
final_test_evaluation.py refuses to run if any fingerprint changes.
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aratox_nlp.config import ROOT
from aratox_nlp.inference import load_calibration

FINAL = ROOT / "outputs" / "final"
FROZEN = FINAL / "frozen_decisions.json"
FINGERPRINTED = [
    "outputs/models/tfidf_svm_multilabel.joblib",
    "outputs/models/qarib/model.safetensors",
    "outputs/models/qarib/config.json",
    "outputs/models/qarib/tokenizer.json",
    "outputs/metrics/calibration.json",
    "configs/data.yaml",
    "configs/models.yaml",
]
TEST_SYSTEMS = [
    {"key": "svm_reported", "system": "TF-IDF + Linear SVM", "policy": "Reported cutoff"},
    {"key": "svm_tuned", "system": "TF-IDF + Linear SVM", "policy": "Tuned per label"},
    {"key": "qarib_reported", "system": "QARiB", "policy": "Reported cutoff"},
    {"key": "qarib_tuned", "system": "QARiB", "policy": "Tuned per label"},
    {"key": "ensemble_tuned", "system": "Ensemble (TF-IDF + QARiB)", "policy": "Tuned per label"},
    {"key": "qarib_int8", "system": "QARiB int8 (CPU)", "policy": "Reported cutoff"},
]


def sha256(path):
    digest = hashlib.sha256()
    with open(ROOT / path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprints():
    return {path: sha256(path) for path in FINGERPRINTED}


def main():
    if (FINAL / "test_results.json").exists():
        raise SystemExit("The test set has already been scored. Decisions cannot be re-frozen afterwards.")
    calibration = load_calibration()
    if calibration is None:
        raise SystemExit("Run scripts/fit_calibration.py first.")
    FINAL.mkdir(parents=True, exist_ok=True)
    frozen = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "fingerprints": fingerprints(),
        "systems": TEST_SYSTEMS,
        "primary_system_key": calibration["selected_system_key"],
        "primary_system": calibration["selected_system"],
        "primary_selection_rule": calibration["selection_rule"],
        "reference_system_key": "svm_reported",
        "jais": "Not included: Jais-2 has no validation run yet, so its prompt mode is not frozen. Score it on test separately after its validation run.",
        "protocol": "Thresholds, calibration, ensemble weight and preprocessing were chosen on validation only. The test split is scored once.",
    }
    FROZEN.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(frozen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
