"""Human spot-check of the predicted dialect regions.

A person labels a small stratified validation sample by region without seeing
the model's guess. Agreement is only computed on rows a person has labelled;
nothing is filled in automatically.
"""

from __future__ import annotations

import pandas as pd

from .presentation import without_sexual_examples

REGIONS = ["Gulf", "Egyptian", "Levantine", "Iraqi", "Maghrebi", "Other"]
HUMAN_CHOICES = REGIONS + ["MSA (formal Arabic)", "Can't tell"]
NOT_SCORED = {"MSA (formal Arabic)", "Can't tell"}
SAMPLE_COLUMNS = ["check_id", "text", "predicted_country_dialect", "predicted_region", "model_confidence", "human_region", "human_notes"]


def build_sample(predictions: pd.DataFrame, per_region: int = 17, seed: int = 42) -> pd.DataFrame:
    frame = predictions[predictions.get("Sexual", 0).astype(int) == 0]
    frame = without_sexual_examples(frame)
    frame = frame[frame.predicted_region.isin(REGIONS)]
    parts = [group.sample(n=min(per_region, len(group)), random_state=seed) for _, group in frame.groupby("predicted_region")]
    sample = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    sample.insert(0, "check_id", range(1, len(sample) + 1))
    sample["human_region"] = ""
    sample["human_notes"] = ""
    return sample[SAMPLE_COLUMNS]


def labelled(sample: pd.DataFrame) -> pd.DataFrame:
    return sample[sample.human_region.fillna("").astype(str).str.strip() != ""]


def agreement(sample: pd.DataFrame) -> dict:
    done = labelled(sample)
    scored = done[~done.human_region.isin(NOT_SCORED)]
    out = {
        "labelled": int(len(done)),
        "total": int(len(sample)),
        "scored": int(len(scored)),
        "msa": int((done.human_region == "MSA (formal Arabic)").sum()),
        "cant_tell": int((done.human_region == "Can't tell").sum()),
        "accuracy": float((scored.human_region == scored.predicted_region).mean()) if len(scored) else None,
    }
    per_region = []
    for region in REGIONS:
        predicted = scored[scored.predicted_region == region]
        if len(predicted):
            per_region.append({"predicted_region": region, "n_scored": len(predicted), "human_agrees": float((predicted.human_region == region).mean())})
    out["per_region"] = per_region
    return out


def confusion(sample: pd.DataFrame) -> pd.DataFrame:
    done = labelled(sample)
    if done.empty:
        return pd.DataFrame()
    return pd.crosstab(done.predicted_region, done.human_region, rownames=["Model guess"], colnames=["Human label"])
