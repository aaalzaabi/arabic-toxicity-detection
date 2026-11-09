import torch
import pytest

from aratox_nlp.dialect import (
    COUNTRY_TO_REGION,
    NADI_CONFIG_LABELS,
    NADI_LABELS,
    ArabicDialectClassifier,
    country_from_index,
    decode_logits,
    pick_device,
    region_for_country,
    script_ratios,
    validate_nadi_config,
)


class _Config:
    def __init__(self, num_labels=21, id2label=None):
        self.num_labels = num_labels
        self.id2label = id2label if id2label is not None else {str(i): name for i, name in NADI_CONFIG_LABELS.items()}


class _Output:
    def __init__(self, logits):
        self.logits = logits


class _Tokenizer:
    def __call__(self, texts, return_tensors="pt", truncation=True, padding=True, max_length=128):
        return {"input_ids": torch.ones(len(texts), 3, dtype=torch.long), "attention_mask": torch.ones(len(texts), 3, dtype=torch.long)}


class _Model:
    def __init__(self, index):
        self.index = index
        self.device = torch.device("cpu")
        self.config = _Config()

    def __call__(self, **kwargs):
        count = kwargs["input_ids"].shape[0]
        logits = torch.full((count, 21), -8.0)
        logits[:, self.index] = 4.0
        return _Output(logits)


class _Boom:
    def __call__(self, **kwargs):
        raise AssertionError("model should not run")


def test_nadi_mapping_matches_config_order_and_length():
    validate_nadi_config(_Config())
    assert list(NADI_LABELS) == list(range(21))
    assert len(NADI_LABELS) == len(NADI_CONFIG_LABELS) == 21
    assert NADI_LABELS[19] == "Emirati"
    assert NADI_CONFIG_LABELS[19] == "United_Arab_Emirates"
    assert [NADI_LABELS[i] for i in range(21)] != sorted(NADI_LABELS.values())


def test_region_mapping_covers_every_country_label():
    assert set(COUNTRY_TO_REGION) == set(NADI_LABELS.values())
    assert region_for_country("Emirati") == "Gulf"
    assert region_for_country("Libyan") == "Maghrebi"
    assert region_for_country("Yemeni") == "Other"
    assert region_for_country("Sudanese") == "Other"
    assert region_for_country("Djiboutian") == "Other"


def test_output_schema_from_logits():
    logits = torch.full((21,), -8.0)
    logits[19] = 4.0
    row = decode_logits(logits.numpy(), "test-model")
    assert row["predicted_country_dialect"] == "Emirati"
    assert row["predicted_region"] == "Gulf"
    assert row["dialect_score"] == pytest.approx(row["model_confidence"])
    assert row["dialect_score"] > 0.9
    assert row["dialect_model"] == "test-model"


def test_batch_inference_schema():
    classifier = ArabicDialectClassifier(model_name="test-model", model=_Model(3), tokenizer=_Tokenizer())
    rows = classifier.predict_batch(["عامل ايه", "hello"])
    assert len(rows) == 2
    assert rows[0]["predicted_country_dialect"] == "Egyptian"
    assert rows[0]["predicted_region"] == "Egyptian"
    assert rows[0]["dialect_status"] == "ok"
    assert rows[1]["dialect_status"] == "low_confidence_input"
    for row in rows:
        assert {"predicted_country_dialect", "predicted_region", "dialect_score", "dialect_model", "model_confidence"} <= set(row)


def test_empty_string_does_not_call_the_model():
    classifier = ArabicDialectClassifier(model_name="test-model", model=_Boom(), tokenizer=_Tokenizer())
    row = classifier.predict("   ")
    assert row["dialect_status"] == "empty_input"
    assert row["predicted_country_dialect"] is None
    assert row["predicted_region"] is None
    assert row["dialect_score"] is None


def test_unsupported_model_label():
    with pytest.raises(ValueError):
        country_from_index(21)
    with pytest.raises(ValueError):
        validate_nadi_config(_Config(num_labels=18))


def test_pick_device_cpu_cuda_and_mps(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert pick_device().type == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert pick_device().type == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert pick_device().type == "mps"


def test_low_arabic_ratio_flag():
    arabic, latin = script_ratios("hello #uae")
    assert arabic == 0.0
    assert latin == 1.0
    classifier = ArabicDialectClassifier(model_name="test-model", low_arabic_ratio_threshold=0.40, model=_Model(19), tokenizer=_Tokenizer())
    row = classifier.predict("Bank of text only")
    assert row["dialect_status"] == "low_confidence_input"
    assert row["arabic_character_ratio"] < 0.40
    assert row["predicted_country_dialect"] == "Emirati"
