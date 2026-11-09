"""One place for model loading and scoring.

Used by the Streamlit app, batch scoring, robustness tests and the final test
evaluation, so every caller applies the same thresholds and calibration.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from .calibration import apply_platt, calibration_input, ensemble_probs
from .classical import decision_to_unit_interval
from .config import ROOT

SVM_SYSTEM = "TF-IDF + Linear SVM"
QARIB_SYSTEM = "QARiB"
ENSEMBLE_SYSTEM = "Ensemble (TF-IDF + QARiB)"
SYSTEMS = (SVM_SYSTEM, QARIB_SYSTEM, ENSEMBLE_SYSTEM)

REPORTED_POLICY = "Reported cutoff"
TUNED_POLICY = "Tuned per label"
POLICIES = (REPORTED_POLICY, TUNED_POLICY)

SVM_PATH = ROOT / "outputs" / "models" / "tfidf_svm_multilabel.joblib"
QARIB_PATH = ROOT / "outputs" / "models" / "qarib"
CALIBRATION_PATH = ROOT / "outputs" / "metrics" / "calibration.json"


def load_svm(path: Path = SVM_PATH):
    if not Path(path).exists():
        raise FileNotFoundError("Run scripts/train_svm.py first.")
    return joblib.load(path)


def pick_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_qarib(path: Path = QARIB_PATH, device=None):
    if not Path(path).exists():
        raise FileNotFoundError("Run scripts/train_qarib.py first.")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = device or pick_device()
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
    return tok, model, device


def load_calibration(path: Path = CALIBRATION_PATH):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def svm_decisions(svm, texts_norm):
    return np.asarray(svm["model"].decision_function(svm["features"].transform(list(texts_norm)))).reshape(len(texts_norm), -1)


def qarib_probs(qarib, texts_norm, batch_size=32, max_length=192):
    import torch

    tok, model, device = qarib
    out = []
    texts = list(texts_norm)
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            enc = tok(texts[start : start + batch_size], return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(device)
            out.append(torch.sigmoid(model(**enc).logits).cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, 6))


def needs_qarib(system):
    return system != SVM_SYSTEM


def needs_calibration(system, policy):
    return system == ENSEMBLE_SYSTEM or policy == TUNED_POLICY


def score_texts(texts_norm, system, policy, svm=None, qarib=None, calibration=None):
    """Return display scores in [0, 1], 0/1 predictions and a note on what the scores mean."""
    if needs_calibration(system, policy) and calibration is None:
        raise FileNotFoundError("Run scripts/fit_calibration.py first.")
    svm_raw = svm_decisions(svm, texts_norm) if system != QARIB_SYSTEM else None
    qarib_raw = qarib_probs(qarib, texts_norm) if needs_qarib(system) else None

    def calibrated(kind, raw):
        return apply_platt(calibration_input(raw, kind), calibration[kind]["platt"])

    if system == SVM_SYSTEM and policy == REPORTED_POLICY:
        return decision_to_unit_interval(svm_raw), (svm_raw >= 0).astype(int), "Scaled SVM decision scores, not calibrated probabilities. Cutoff: decision score 0."
    if system == QARIB_SYSTEM and policy == REPORTED_POLICY:
        return qarib_raw, (qarib_raw >= 0.5).astype(int), "Sigmoid scores; calibration is not assumed. Cutoff: 0.50."
    if system == SVM_SYSTEM:
        probs, key = calibrated("svm", svm_raw), "svm"
    elif system == QARIB_SYSTEM:
        probs, key = calibrated("qarib", qarib_raw), "qarib"
    else:
        probs, key = ensemble_probs(calibrated("svm", svm_raw), calibrated("qarib", qarib_raw), calibration["ensemble"]["weight"]), "ensemble"
    if policy == TUNED_POLICY:
        thresholds = np.asarray(calibration[key]["thresholds_tuned"])
        note = "Calibrated probabilities (Platt scaling on validation). Cutoffs tuned per label on validation."
    else:
        thresholds = np.full(probs.shape[1], 0.5)
        note = "Calibrated probabilities (Platt scaling on validation). Cutoff: 0.50."
    return probs, (probs >= thresholds).astype(int), note


def threshold_vector(system, policy, calibration):
    """Cutoffs in the same units as the display scores from score_texts."""
    if policy == REPORTED_POLICY:
        return np.full(6, 0.5)
    key = {SVM_SYSTEM: "svm", QARIB_SYSTEM: "qarib", ENSEMBLE_SYSTEM: "ensemble"}[system]
    return np.asarray(calibration[key]["thresholds_tuned"])
