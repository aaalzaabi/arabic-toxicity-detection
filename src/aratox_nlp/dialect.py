"""Separate Arabic dialect identification.

The classifier is the CAMeL Lab NADI country model
``CAMeL-Lab/bert-base-arabic-camelbert-mix-did-nadi`` (Inoue et al., 2021),
fine-tuned from ``CAMeL-Lab/bert-base-arabic-camelbert-mix`` on the NADI
country-level task (21 countries). It does not read AraTox labels, and its
output is a prediction of linguistic style, not nationality and not AraTox
ground truth.

``NADI_CONFIG_LABELS`` is the exact ``id2label`` order shipped in the model
config. ``NADI_LABELS`` are the display names for that same order. Do not
rebuild either map by sorting names.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .constants import TOXIC_LABELS

MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-mix-did-nadi"

# Exact id2label strings from the model config, index 0..20.
NADI_CONFIG_LABELS = {
    0: "Algeria",
    1: "Bahrain",
    2: "Djibouti",
    3: "Egypt",
    4: "Iraq",
    5: "Jordan",
    6: "Kuwait",
    7: "Lebanon",
    8: "Libya",
    9: "Mauritania",
    10: "Morocco",
    11: "Oman",
    12: "Palestine",
    13: "Qatar",
    14: "Saudi_Arabia",
    15: "Somalia",
    16: "Sudan",
    17: "Syria",
    18: "Tunisia",
    19: "United_Arab_Emirates",
    20: "Yemen",
}

# Display names in the same index order as NADI_CONFIG_LABELS.
NADI_LABELS = {
    0: "Algerian",
    1: "Bahraini",
    2: "Djiboutian",
    3: "Egyptian",
    4: "Iraqi",
    5: "Jordanian",
    6: "Kuwaiti",
    7: "Lebanese",
    8: "Libyan",
    9: "Mauritanian",
    10: "Moroccan",
    11: "Omani",
    12: "Palestinian",
    13: "Qatari",
    14: "Saudi",
    15: "Somali",
    16: "Sudanese",
    17: "Syrian",
    18: "Tunisian",
    19: "Emirati",
    20: "Yemeni",
}

# Regional families for the NADI countries.
# Gulf / Egyptian / Levantine / Iraqi / Maghrebi follow the country list
# requested for this project and the usual MADAR-style split of those countries.
# Yemen and Sudan stay outside Gulf and Egyptian: NADI treats them as their own
# countries, and this project does not fold them into a larger family.
# Djibouti, Mauritania, and Somalia are NADI classes with no requested family,
# so they stay Other rather than being collapsed into Maghrebi or Gulf.
COUNTRY_TO_REGION = {
    "Emirati": "Gulf",
    "Saudi": "Gulf",
    "Kuwaiti": "Gulf",
    "Qatari": "Gulf",
    "Bahraini": "Gulf",
    "Omani": "Gulf",
    "Egyptian": "Egyptian",
    "Lebanese": "Levantine",
    "Syrian": "Levantine",
    "Jordanian": "Levantine",
    "Palestinian": "Levantine",
    "Iraqi": "Iraqi",
    "Moroccan": "Maghrebi",
    "Algerian": "Maghrebi",
    "Tunisian": "Maghrebi",
    "Libyan": "Maghrebi",
    "Yemeni": "Other",
    "Sudanese": "Other",
    "Djiboutian": "Other",
    "Mauritanian": "Other",
    "Somali": "Other",
}

OUTPUT_FIELDS = (
    "predicted_country_dialect",
    "predicted_region",
    "dialect_score",
    "model_confidence",
    "dialect_model",
    "dialect_status",
    "arabic_character_ratio",
    "latin_character_ratio",
)

ARABIC_LETTER = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
LATIN_LETTER = re.compile(r"[A-Za-z]")


def pick_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def script_ratios(text: str) -> tuple[float, float]:
    arabic = len(ARABIC_LETTER.findall(text or ""))
    latin = len(LATIN_LETTER.findall(text or ""))
    total = arabic + latin
    if total == 0:
        return 0.0, 0.0
    return arabic / total, latin / total


def validate_nadi_config(config) -> None:
    id2 = {int(k): v for k, v in dict(config.id2label).items()}
    num_labels = getattr(config, "num_labels", None)
    if num_labels is None:
        num_labels = len(id2)
    if num_labels != len(NADI_LABELS) or len(id2) != len(NADI_CONFIG_LABELS):
        raise ValueError(
            f"NADI label mapping has {len(NADI_LABELS)} entries but the model config has num_labels={num_labels} and {len(id2)} id2label entries."
        )
    for index, raw in NADI_CONFIG_LABELS.items():
        if id2.get(index) != raw:
            raise ValueError(f"Model id2label[{index}] is {id2.get(index)!r}, expected {raw!r}.")


def country_from_index(index: int) -> str:
    if index not in NADI_LABELS:
        raise ValueError(f"Unsupported model label index {index}.")
    return NADI_LABELS[index]


def region_for_country(country: str) -> str:
    if country not in COUNTRY_TO_REGION:
        raise ValueError(f"No regional mapping for predicted country dialect {country!r}.")
    return COUNTRY_TO_REGION[country]


def decode_logits(logits, model_name: str = MODEL_NAME) -> dict:
    scores = np.asarray(logits, dtype=np.float64).reshape(-1)
    scores = scores - scores.max()
    exp = np.exp(scores)
    probs = exp / exp.sum()
    index = int(probs.argmax())
    country = country_from_index(index)
    confidence = float(probs[index])
    return {
        "predicted_country_dialect": country,
        "predicted_region": region_for_country(country),
        "dialect_score": confidence,
        "model_confidence": confidence,
        "dialect_model": model_name,
    }


def _blank_prediction(model_name: str, status: str, arabic: float, latin: float) -> dict:
    return {
        "predicted_country_dialect": None,
        "predicted_region": None,
        "dialect_score": None,
        "model_confidence": None,
        "dialect_model": model_name,
        "dialect_status": status,
        "arabic_character_ratio": arabic,
        "latin_character_ratio": latin,
    }


class ArabicDialectClassifier:
    def __init__(self, model_name: str = MODEL_NAME, low_arabic_ratio_threshold: float = 0.40, max_length: int = 128, batch_size: int = 32, model=None, tokenizer=None):
        self.model_name = model_name
        self.low_arabic_ratio_threshold = float(low_arabic_ratio_threshold)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self._model = model
        self._tokenizer = tokenizer
        self._validated = False

    def _ensure_loaded(self):
        if self._model is not None and self._tokenizer is not None:
            if not self._validated and hasattr(self._model, "config"):
                validate_nadi_config(self._model.config)
                self._validated = True
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        validate_nadi_config(self._model.config)
        self._validated = True
        device = pick_device()
        self._model.to(device).eval()

    def _logits(self, texts: list[str]):
        import torch

        self._ensure_loaded()
        device = getattr(self._model, "device", None) or pick_device()
        encoded = self._tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = self._model(**encoded).logits
        return logits.detach().cpu().numpy()

    def predict(self, text: str) -> dict:
        rows = self.predict_batch([text])
        return rows[0]

    def predict_batch(self, texts: list[str]) -> list[dict]:
        prepared = []
        pending_at = []
        pending_texts = []
        for index, text in enumerate(texts):
            raw = "" if text is None else str(text)
            arabic, latin = script_ratios(raw)
            if not raw.strip():
                prepared.append(_blank_prediction(self.model_name, "empty_input", arabic, latin))
                continue
            pending_at.append(index)
            pending_texts.append(raw)
            prepared.append(None)
        for start in range(0, len(pending_texts), self.batch_size):
            chunk = pending_texts[start:start + self.batch_size]
            logits = self._logits(chunk)
            for offset, row_logits in enumerate(logits):
                slot = pending_at[start + offset]
                raw = pending_texts[start + offset]
                arabic, latin = script_ratios(raw)
                decoded = decode_logits(row_logits, self.model_name)
                status = "ok"
                if arabic < self.low_arabic_ratio_threshold:
                    status = "low_confidence_input"
                decoded["dialect_status"] = status
                decoded["arabic_character_ratio"] = arabic
                decoded["latin_character_ratio"] = latin
                prepared[slot] = decoded
        return prepared


def predictions_frame(texts: list[str], rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame.insert(0, "text", ["" if text is None else str(text) for text in texts])
    return frame


def combine_with_toxicity(dialect_df: pd.DataFrame, toxicity_df: pd.DataFrame, prefix: str = "qarib") -> pd.DataFrame:
    if len(dialect_df) != len(toxicity_df):
        raise ValueError(f"Dialect rows ({len(dialect_df)}) and toxicity rows ({len(toxicity_df)}) differ.")
    out = pd.DataFrame({"text": toxicity_df["text_raw"].astype(str).values})
    out["binary_toxic"] = toxicity_df["binary_toxic"].astype(int).values
    for label in TOXIC_LABELS:
        out[label] = toxicity_df[label].astype(int).values
        out[f"{prefix}_pred_{label}"] = toxicity_df[f"{prefix}_pred_{label}"].astype(int).values
    binary_col = f"{prefix}_pred_binary"
    if binary_col in toxicity_df.columns:
        out[binary_col] = toxicity_df[binary_col].astype(int).values
    else:
        pred_cols = [f"{prefix}_pred_{label}" for label in TOXIC_LABELS]
        out[binary_col] = (toxicity_df[pred_cols].astype(int).sum(axis=1) > 0).astype(int).values
    for column in (
        "predicted_country_dialect",
        "predicted_region",
        "dialect_score",
        "model_confidence",
        "dialect_model",
        "dialect_status",
        "arabic_character_ratio",
        "latin_character_ratio",
    ):
        out[column] = dialect_df[column].values
    return out
