"""Decision-oriented summary of the saved TF-IDF and QARiB validation results.

Qualitative fields are project metadata. Numeric fields are read from saved
metric files. This module does not score models or build a composite ethics score.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .comparison import QARIB_NAME, SVM_NAME

PROJECT_FRAMING = (
    "This project does not only ask which Arabic toxicity classifier achieves the highest F1. "
    "It asks which model and operating policy are most appropriate when predictive quality, "
    "inference efficiency, false-positive risk, language variation, and human-review requirements "
    "are considered together."
)
PRIMARY_RESEARCH_QUESTION = (
    "Does a contextual Arabic transformer improve multi-label toxicity detection enough to "
    "justify its additional computational cost over a classical TF-IDF + Linear SVM baseline?"
)
RESPONSIBLE_RESEARCH_QUESTION = (
    "Does the model with the highest predictive performance also provide acceptable error "
    "behavior for a human-reviewed moderation setting?"
)
THRESHOLD_RESEARCH_QUESTION = (
    "How do different decision thresholds change the trade-off between false positives and missed toxic content?"
)
JAIS_STATUS = (
    "Jais-2-8B-Chat (zero-shot prompt, no training) was evaluated separately: Macro-F1 0.369 on 500 validation "
    "comments and 0.346 on the full test split. It is not in this scorecard, which compares the trained models; "
    "see the Final Results tab."
)
SCENARIO_GUIDANCE = (
    "If predictive performance is the main priority:\n"
    "QARiB is preferred on the current validation set.\n\n"
    "If extremely low inference cost and simple deployment are priorities:\n"
    "TF-IDF + Linear SVM remains attractive.\n\n"
    "If false positives are especially costly:\n"
    "The threshold and per-label error behavior should be reviewed before deployment.\n\n"
    "For consequential moderation:\n"
    "Neither model should act without human review."
)
SELECTION_CONCLUSION = (
    "QARiB achieves the strongest validation F1, but its advantage over TF-IDF is modest.\n\n"
    "TF-IDF remains attractive for low-latency and simpler deployments.\n\n"
    "Error costs differ by toxicity category, so aggregate F1 alone is not sufficient for deployment decisions.\n\n"
    "Threshold selection changes the balance between false positives and false negatives.\n\n"
    "Human review remains necessary for consequential moderation.\n\n"
    "QARiB has the best predictive performance on the current validation set."
)
LINGUISTIC_ROBUSTNESS = (
    "The separate NADI classifier provides predicted linguistic categories. These predictions "
    "can help explore possible error patterns, but they cannot establish fairness across true dialect groups."
)
EXPLORATORY_REGION_LABEL = "exploratory false-positive distribution by predicted region"

MODEL_METADATA = {
    SVM_NAME: {
        "contextual_model": "No",
        "deployment_complexity": "Low",
        "context_modeling": "Limited",
        "human_review": "Required",
        "notes": "Classical sparse word and character features. Fast on CPU in the measured environment.",
    },
    QARIB_NAME: {
        "contextual_model": "Yes",
        "deployment_complexity": "Higher",
        "context_modeling": "Contextual",
        "human_review": "Required",
        "notes": "Pretrained Arabic transformer. Heavier than TF-IDF and measured here on Apple MPS.",
    },
}


def _fewer_false_positive_labels(error_counts: pd.DataFrame) -> dict[str, str]:
    tfidf_labels = []
    qarib_labels = []
    tied = []
    for row in error_counts.itertuples():
        tfidf_count = int(row.tfidf_false_positives)
        qarib_count = int(row.qarib_false_positives)
        if tfidf_count < qarib_count:
            tfidf_labels.append(row.label)
        elif qarib_count < tfidf_count:
            qarib_labels.append(row.label)
        else:
            tied.append(row.label)
    def phrase(labels):
        if not labels:
            return "No label with fewer false positives"
        return "Fewer false positives on " + ", ".join(labels)
    note = {"tied": ", ".join(tied)} if tied else {}
    return {SVM_NAME: phrase(tfidf_labels), QARIB_NAME: phrase(qarib_labels), **note}


def load_model_selection(metrics_dir: Path) -> pd.DataFrame:
    metrics_dir = Path(metrics_dir)
    summary = pd.read_csv(metrics_dir / "tfidf_vs_qarib_summary.csv")
    latency = pd.read_csv(metrics_dir / "tfidf_vs_qarib_latency.csv")
    errors = pd.read_csv(metrics_dir / "error_counts_by_label.csv")
    behavior = _fewer_false_positive_labels(errors)
    rows = []
    for model in (SVM_NAME, QARIB_NAME):
        score = summary[summary.model == model]
        timing = latency[latency.model == model]
        if score.empty or timing.empty:
            raise ValueError(f"Saved metrics are missing {model}.")
        meta = MODEL_METADATA[model]
        score_row = score.iloc[0]
        time_row = timing.iloc[0]
        rows.append({
            "model": model,
            "macro_f1": float(score_row.macro_f1),
            "micro_f1": float(score_row.micro_f1),
            "false_positive_behavior": behavior[model],
            "mean_latency_ms": float(time_row.mean_ms_per_sample),
            "samples_per_second": float(time_row.samples_per_second),
            "device": str(time_row.device),
            "contextual_model": meta["contextual_model"],
            "deployment_complexity": meta["deployment_complexity"],
            "notes": meta["notes"],
        })
    table = pd.DataFrame(rows)
    if "ethical_score" in table.columns:
        raise ValueError("A composite ethics score is not part of this comparison.")
    return table


def scorecard(table: pd.DataFrame) -> pd.DataFrame:
    """Wide comparison. Quantitative cells come from saved metrics."""
    indexed = table.set_index("model")
    tfidf = indexed.loc[SVM_NAME]
    qarib = indexed.loc[QARIB_NAME]
    device = {"cpu": "CPU", "mps": "Apple MPS", "cuda": "CUDA"}
    return pd.DataFrame([
        {"Dimension": "Macro-F1", "TF-IDF + SVM": f"{tfidf.macro_f1:.3f}", "QARiB": f"{qarib.macro_f1:.3f}"},
        {"Dimension": "Micro-F1", "TF-IDF + SVM": f"{tfidf.micro_f1:.3f}", "QARiB": f"{qarib.micro_f1:.3f}"},
        {"Dimension": "Context modeling", "TF-IDF + SVM": MODEL_METADATA[SVM_NAME]["context_modeling"], "QARiB": MODEL_METADATA[QARIB_NAME]["context_modeling"]},
        {
            "Dimension": "Inference latency",
            "TF-IDF + SVM": f"{tfidf.mean_latency_ms:.2f} ms/sample on {device.get(tfidf.device, tfidf.device)}",
            "QARiB": f"{qarib.mean_latency_ms:.2f} ms/sample on {device.get(qarib.device, qarib.device)}",
        },
        {"Dimension": "Deployment complexity", "TF-IDF + SVM": tfidf.deployment_complexity, "QARiB": qarib.deployment_complexity},
        {"Dimension": "False positives", "TF-IDF + SVM": tfidf.false_positive_behavior, "QARiB": qarib.false_positive_behavior},
        {"Dimension": "Human review", "TF-IDF + SVM": "Required", "QARiB": "Required"},
    ])
