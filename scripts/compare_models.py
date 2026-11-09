import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aratox_nlp.comparison import (
    QARIB_NAME,
    SVM_NAME,
    align_predictions,
    disagreement_frame,
    disagreement_summary,
    error_counts_by_label,
    focused_examples,
    mcnemar_binary,
    per_label_table,
    summary_table,
    wins_by_label,
)
from aratox_nlp.config import ROOT
from aratox_nlp.dialect import pick_device

METRICS = ROOT / "outputs" / "metrics"
FIGURES = ROOT / "outputs" / "figures"
ERRORS = ROOT / "outputs" / "errors"


def _save_bars(path: Path, categories: list[str], series: list[tuple[str, list[float]]], ylabel: str, title: str, ylim=None):
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(categories))
    width = 0.36
    for index, (name, values) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * width
        axis.bar(x + offset, values, width, label=name)
    axis.set_xticks(x, categories)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    if ylim is not None:
        axis.set_ylim(*ylim)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_from_tables(per_label: pd.DataFrame, summary: pd.DataFrame):
    _save_bars(
        FIGURES / "tfidf_vs_qarib_per_label_f1.png",
        per_label.label.tolist(),
        [
            ("TF-IDF + Linear SVM", per_label.tfidf_f1.tolist()),
            ("QARiB", per_label.qarib_f1.tolist()),
        ],
        "F1-score",
        "Validation F1 by toxicity label",
        (0, 1),
    )
    overall = summary.set_index("model")
    metrics = ["macro_f1", "micro_f1", "subset_accuracy"]
    labels = ["Macro-F1", "Micro-F1", "Subset accuracy"]
    _save_bars(
        FIGURES / "tfidf_vs_qarib_overall.png",
        labels,
        [
            ("TF-IDF + Linear SVM", [float(overall.loc[SVM_NAME, metric]) for metric in metrics]),
            ("QARiB", [float(overall.loc[QARIB_NAME, metric]) for metric in metrics]),
        ],
        "Score",
        "Validation overall comparison",
        (0, 1),
    )
    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    names = [SVM_NAME, QARIB_NAME]
    values = [float(overall.loc[name, "hamming_loss"]) for name in names]
    axis.bar(names, values)
    axis.set_ylabel("Hamming loss")
    axis.set_title("Validation Hamming loss (lower is better)")
    figure.tight_layout()
    figure.savefig(FIGURES / "tfidf_vs_qarib_hamming.png", dpi=180)
    plt.close(figure)


def _time_svm(texts: list[str]) -> tuple[float, str]:
    obj = joblib.load(ROOT / "outputs" / "models" / "tfidf_svm_multilabel.joblib")
    warmup = texts[:8]
    obj["model"].decision_function(obj["features"].transform(warmup))
    started = time.perf_counter()
    obj["model"].decision_function(obj["features"].transform(texts))
    return time.perf_counter() - started, "cpu"


def _time_qarib(texts: list[str]) -> tuple[float, str]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "outputs" / "models" / "qarib")
    model = AutoModelForSequenceClassification.from_pretrained(ROOT / "outputs" / "models" / "qarib")
    model.to(device).eval()

    def run(batch):
        encoded = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=192)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            model(**encoded)

    run(texts[:8])
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for start in range(0, len(texts), 16):
        run(texts[start:start + 16])
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - started, device.type


def latency_table(texts: list[str]) -> pd.DataFrame:
    rows = []
    for name, timer in ((SVM_NAME, _time_svm), (QARIB_NAME, _time_qarib)):
        seconds, device = timer(texts)
        rows.append({
            "model": name,
            "n": len(texts),
            "total_seconds": seconds,
            "mean_ms_per_sample": seconds / len(texts) * 1000,
            "samples_per_second": len(texts) / seconds,
            "device": device,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Compare TF-IDF + SVM and QARiB on aligned validation predictions.")
    parser.add_argument("--svm", default="outputs/predictions/svm_validation.csv")
    parser.add_argument("--qarib", default="outputs/predictions/qarib_validation.csv")
    parser.add_argument("--latency-n", type=int, default=500)
    parser.add_argument("--skip-latency", action="store_true")
    args = parser.parse_args()
    svm_path = Path(args.svm) if Path(args.svm).is_absolute() else ROOT / args.svm
    qarib_path = Path(args.qarib) if Path(args.qarib).is_absolute() else ROOT / args.qarib
    aligned = align_predictions(pd.read_csv(svm_path), pd.read_csv(qarib_path))
    METRICS.mkdir(parents=True, exist_ok=True)
    ERRORS.mkdir(parents=True, exist_ok=True)

    summary = summary_table(aligned)
    per_label = per_label_table(aligned)
    summary_path = METRICS / "tfidf_vs_qarib_summary.csv"
    per_label_path = METRICS / "tfidf_vs_qarib_per_label.csv"
    summary.to_csv(summary_path, index=False)
    per_label.to_csv(per_label_path, index=False)
    plot_from_tables(pd.read_csv(per_label_path), pd.read_csv(summary_path))

    disagreements = disagreement_frame(aligned)
    disagreements.to_csv(ERRORS / "tfidf_vs_qarib_disagreements.csv", index=False)
    focused_examples(disagreements, "qarib_only_correct").to_csv(ERRORS / "qarib_correct_tfidf_wrong.csv", index=False)
    focused_examples(disagreements, "tfidf_only_correct").to_csv(ERRORS / "tfidf_correct_qarib_wrong.csv", index=False)
    summary_json = disagreement_summary(disagreements)
    (METRICS / "tfidf_vs_qarib_disagreement_summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    wins_by_label(aligned).to_csv(METRICS / "model_wins_by_label.csv", index=False)
    error_counts_by_label(aligned).to_csv(METRICS / "error_counts_by_label.csv", index=False)
    (METRICS / "tfidf_vs_qarib_mcnemar.json").write_text(json.dumps(mcnemar_binary(aligned), indent=2), encoding="utf-8")

    if not args.skip_latency:
        count = min(args.latency_n, len(aligned))
        latency = latency_table(aligned.text_norm.astype(str).head(count).tolist())
        latency.to_csv(METRICS / "tfidf_vs_qarib_latency.csv", index=False)
        print(latency.to_string(index=False))
    print(summary.to_string(index=False))
    print(per_label.to_string(index=False))
    print(json.dumps(summary_json, indent=2))


if __name__ == "__main__":
    main()
