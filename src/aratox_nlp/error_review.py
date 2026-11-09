"""Reproducible false-positive review samples.

Review categories are left blank. Predicted dialect is attached only as
exploratory metadata and is not a toxicity feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .comparison import QARIB_NAME, SVM_NAME
from .constants import TOXIC_LABELS

REVIEW_CATEGORIES = (
    "slang_or_dialect_expression",
    "sarcasm_or_irony",
    "reclaimed_or_context_dependent_expression",
    "code_switching",
    "missing_conversation_context",
    "ambiguous_wording",
    "possible_annotation_disagreement",
    "lexical_trigger",
    "other",
)
REVIEW_COLUMNS = (
    "row_id",
    "text_raw",
    "true_labels",
    "tfidf_predicted_labels",
    "qarib_predicted_labels",
    "trigger_model",
    "trigger_label",
    "predicted_dialect",
    "predicted_region",
    "review_category",
    "review_notes",
    "reviewed",
)
REVIEWER_WARNING = (
    "Review categories are human interpretations for error analysis. They are not "
    "automatically generated model explanations and should not be treated as ground truth."
)
PREDICTION_COLUMNS = (
    "text_raw",
    "true_labels",
    "tfidf_predicted_labels",
    "qarib_predicted_labels",
    "trigger_model",
    "trigger_label",
    "predicted_dialect",
    "predicted_region",
)


def joined_labels(values) -> str:
    names = [label for label, value in zip(TOXIC_LABELS, values) if int(value) == 1]
    return "|".join(names) if names else "none"


def false_positive_events(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, row in frame.iterrows():
        true_labels = joined_labels([row[label] for label in TOXIC_LABELS])
        tfidf_labels = joined_labels([row[f"svm_pred_{label}"] for label in TOXIC_LABELS])
        qarib_labels = joined_labels([row[f"qarib_pred_{label}"] for label in TOXIC_LABELS])
        for model, prefix in ((SVM_NAME, "svm"), (QARIB_NAME, "qarib")):
            for label in TOXIC_LABELS:
                if int(row[label]) == 0 and int(row[f"{prefix}_pred_{label}"]) == 1:
                    rows.append({
                        "row_index": int(index),
                        "text_raw": row["text_raw"],
                        "true_labels": true_labels,
                        "tfidf_predicted_labels": tfidf_labels,
                        "qarib_predicted_labels": qarib_labels,
                        "trigger_model": model,
                        "trigger_label": label,
                    })
    return pd.DataFrame(rows)


def stratified_review_sample(events: pd.DataFrame, size: int = 100, seed: int = 42) -> pd.DataFrame:
    if events.empty or size <= 0:
        return events.iloc[0:0].copy()
    rng = np.random.default_rng(seed)
    remaining = events.copy()
    chosen = []
    quota = {label: size // len(TOXIC_LABELS) for label in TOXIC_LABELS}
    for label in TOXIC_LABELS[: size % len(TOXIC_LABELS)]:
        quota[label] += 1
    unmet = 0
    for label in TOXIC_LABELS:
        pool = remaining[remaining.trigger_label == label]
        take = min(quota[label], len(pool))
        unmet += quota[label] - take
        if take:
            picked = rng.choice(pool.index.to_numpy(), size=take, replace=False)
            chosen.extend(picked.tolist())
            remaining = remaining.drop(index=picked)
    if unmet and not remaining.empty:
        extra = rng.choice(remaining.index.to_numpy(), size=min(unmet, len(remaining)), replace=False)
        chosen.extend(np.atleast_1d(extra).tolist())
    sample = events.loc[chosen].copy()
    order = {label: position for position, label in enumerate(TOXIC_LABELS)}
    sample["_order"] = sample.trigger_label.map(order)
    return sample.sort_values(["_order", "row_index", "trigger_model"]).drop(columns="_order").reset_index(drop=True)


def attach_predicted_dialect(sample: pd.DataFrame, dialect: pd.DataFrame | None) -> pd.DataFrame:
    out = sample.copy()
    out["predicted_dialect"] = ""
    out["predicted_region"] = ""
    if dialect is None or dialect.empty or "source_row" not in dialect.columns:
        return out
    dialect = dialect.copy()
    dialect["source_row"] = dialect["source_row"].astype(int)
    lookup = dialect.set_index("source_row")
    def value(row_index, column):
        if row_index not in lookup.index or column not in lookup.columns:
            return ""
        item = lookup.at[row_index, column]
        return "" if pd.isna(item) else str(item)
    country_column = "predicted_country_dialect" if "predicted_country_dialect" in lookup.columns else "predicted_dialect"
    out["predicted_dialect"] = [value(index, country_column) for index in out.row_index]
    out["predicted_region"] = [value(index, "predicted_region") for index in out.row_index]
    return out


def review_frame(sample: pd.DataFrame) -> pd.DataFrame:
    out = sample.copy()
    out["row_id"] = [
        f"{int(row.row_index)}:{row.trigger_model}:{row.trigger_label}" for row in out.itertuples()
    ]
    for column in ("review_category", "review_notes", "reviewed"):
        out[column] = ""
    return out[list(REVIEW_COLUMNS)]


def apply_review(frame: pd.DataFrame, row_id: str, categories: list[str], notes: str) -> pd.DataFrame:
    out = frame.copy()
    unknown = [category for category in categories if category not in REVIEW_CATEGORIES]
    if unknown:
        raise ValueError(f"Unknown review categories: {unknown}")
    mask = out.row_id.astype(str) == str(row_id)
    if not mask.any():
        raise KeyError(row_id)
    before = out.loc[mask, list(PREDICTION_COLUMNS)].copy()
    out.loc[mask, "review_category"] = "|".join(categories)
    out.loc[mask, "review_notes"] = notes
    out.loc[mask, "reviewed"] = "yes"
    after = out.loc[mask, list(PREDICTION_COLUMNS)]
    if not before.reset_index(drop=True).equals(after.reset_index(drop=True)):
        raise RuntimeError("Review saving changed a model prediction field.")
    return out


def reviewed_rows(frame: pd.DataFrame) -> pd.DataFrame:
    flag = frame.get("reviewed", pd.Series("", index=frame.index)).fillna("").astype(str).str.lower()
    return frame[flag.isin({"yes", "true", "1"})].copy()


def review_summary(frame: pd.DataFrame) -> pd.DataFrame:
    reviewed = reviewed_rows(frame)
    columns = ["review_category", "count", "percentage_of_reviewed_examples"]
    if reviewed.empty:
        return pd.DataFrame(columns=columns)
    total = len(reviewed)
    counts: dict[str, int] = {}
    for value in reviewed.review_category.fillna(""):
        for category in str(value).split("|"):
            if category:
                counts[category] = counts.get(category, 0) + 1
    rows = [
        {
            "review_category": category,
            "count": count,
            "percentage_of_reviewed_examples": 100.0 * count / total,
        }
        for category, count in sorted(counts.items())
    ]
    return pd.DataFrame(rows, columns=columns)


def false_positives_by_predicted_region(events: pd.DataFrame, dialect: pd.DataFrame) -> pd.DataFrame:
    """Distribution of false-positive events across predicted regions.

    This is not a fairness audit. Predicted region is not AraTox ground truth.
    """
    dialect = dialect.copy()
    dialect["source_row"] = dialect["source_row"].astype(int)
    regions = dialect.set_index("source_row")["predicted_region"]
    example_counts = regions.fillna("").astype(str).value_counts()
    tagged = events.copy()
    tagged["predicted_region"] = tagged.row_index.map(regions).fillna("").astype(str)
    event_counts = tagged.predicted_region.value_counts()
    total_events = int(event_counts.sum())
    rows = []
    for region in sorted(set(example_counts.index) | set(event_counts.index)):
        false_positives = int(event_counts.get(region, 0))
        rows.append({
            "predicted_region": region,
            "n_examples": int(example_counts.get(region, 0)),
            "n_false_positives": false_positives,
            "false_positive_share": (false_positives / total_events) if total_events else 0.0,
        })
    return pd.DataFrame(rows)
