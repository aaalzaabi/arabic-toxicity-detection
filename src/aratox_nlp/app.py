import html
import json
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

from aratox_nlp.comparison import QARIB_NAME, SVM_NAME
from aratox_nlp.comparison_view import (
    EXAMPLE_SOURCES,
    FALSE_POSITIVE_INTERPRETATION,
    LATENCY_WARNING,
    OVERALL_INTERPRETATION,
    QARIB_SCORE_NOTE,
    SCORE_MAGNITUDE_NOTE,
    SOURCE_TO_GROUP,
    TFIDF_SCORE_NOTE,
    binary_toxic,
    correctness_mark,
    device_label,
    disagreement_stats,
    displayed_latency_ratio,
    example_label_view,
    filter_validation_examples,
    fewer_tfidf_false_positives,
    key_finding_text,
    missing_comparison_files,
    model_metrics,
    overall_improvement,
    per_label_interpretation,
    read_json,
    read_table,
    sorted_label_improvements,
    step_example_index,
    yes_no,
)
from aratox_nlp.config import ROOT, load_yaml
from aratox_nlp.constants import TOXIC_LABELS
from aratox_nlp.ethics import (
    ANNOTATION_BIAS,
    DIALECT_CAVEAT,
    DIALECT_NOT_IDENTITY,
    EXPLORER_NOTE,
    F1_NOT_FEWEST_FALSE_POSITIVES,
    FALSE_POSITIVE_CONSEQUENCE,
    HUMAN_REVIEW,
    POLICY_CHOICES,
    RESEARCH_CONCLUSION,
    RESPONSIBLE_SUMMARY,
    REVIEW_FIELDS,
    THRESHOLD_POLICY,
    nearest_threshold_row,
    operating_points,
)
from aratox_nlp.error_review import REVIEW_CATEGORIES, REVIEWER_WARNING, apply_review, review_summary
from aratox_nlp.model_selection import (
    EXPLORATORY_REGION_LABEL,
    JAIS_STATUS,
    LINGUISTIC_ROBUSTNESS,
    PRIMARY_RESEARCH_QUESTION,
    PROJECT_FRAMING,
    RESPONSIBLE_RESEARCH_QUESTION,
    SCENARIO_GUIDANCE,
    SELECTION_CONCLUSION,
    THRESHOLD_RESEARCH_QUESTION,
    load_model_selection,
    scorecard,
)
from aratox_nlp.operating_modes import MODE_NOTE, operating_mode_table
from aratox_nlp.presentation import without_sexual_examples
from aratox_nlp import inference
from aratox_nlp.dialect import ArabicDialectClassifier
from aratox_nlp.dialect_check import HUMAN_CHOICES, agreement, confusion, labelled
from aratox_nlp.explain import top_words, word_occlusion
from aratox_nlp.preprocessing import config_from_dict, normalize_arabic

REVIEWS = ROOT / "outputs" / "reviews" / "ratings.csv"

CATEGORY_LABELS = {
    "slang_or_dialect_expression": "Slang or a dialect expression",
    "sarcasm_or_irony": "Sarcasm or irony",
    "reclaimed_or_context_dependent_expression": "Reclaimed or context-dependent wording",
    "code_switching": "Mixed languages",
    "missing_conversation_context": "Missing conversation context",
    "ambiguous_wording": "Ambiguous wording",
    "possible_annotation_disagreement": "The label itself may be debatable",
    "lexical_trigger": "A single word triggered the flag",
    "other": "Other",
}
LABEL_TO_CATEGORY = {label: key for key, label in CATEGORY_LABELS.items()}

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;600&display=swap');
.block-container {padding-top: 2rem; max-width: 1240px;}
.hero {background: linear-gradient(135deg, #0f3d3a 0%, #172554 100%); border: 1px solid rgba(20,184,166,.35); border-radius: 16px; padding: 1.5rem 1.8rem; margin-bottom: 1rem;}
.hero h1 {color: #fff; font-size: 2rem; margin: 0 0 .35rem 0; padding: 0;}
.hero p {color: rgba(255,255,255,.88); margin: 0 0 .8rem 0;}
.hero .pill {display: inline-block; color: #fff; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.3); border-radius: 999px; padding: .12rem .7rem; margin: 0 .35rem .35rem 0; font-size: .82rem;}
div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); border-radius: 12px; padding: .75rem 1rem; background: #111a2b;}
.stTabs [data-baseweb="tab-list"] {gap: .25rem; border-bottom: 1px solid rgba(128,128,128,.22);}
.stTabs [data-baseweb="tab"] {padding: .55rem 1rem; font-weight: 600;}
.stTextArea textarea {unicode-bidi: plaintext; text-align: start; font-family: 'IBM Plex Sans Arabic', sans-serif; font-size: 1.05rem;}
.ar-text {font-family: 'IBM Plex Sans Arabic', sans-serif; font-size: 1.25rem; line-height: 1.9; text-align: start; padding: .9rem 1.1rem; margin: .3rem 0 .9rem 0; border-radius: 12px; border: 1px solid rgba(128,128,128,.22); border-inline-start: 4px solid #14b8a6; background: #111a2b;}
.verdict {border-radius: 14px; border: 1px solid; padding: 1rem 1.2rem; margin-bottom: .9rem;}
.verdict.toxic {background: rgba(220,38,38,.08); border-color: rgba(220,38,38,.45);}
.verdict.clean {background: rgba(22,163,74,.08); border-color: rgba(22,163,74,.45);}
.verdict .kicker {font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; opacity: .75;}
.verdict .value {font-size: 1.7rem; font-weight: 700; margin: .1rem 0 .3rem 0;}
.verdict.toxic .value {color: #f87171;}
.verdict.clean .value {color: #4ade80;}
.chip {display: inline-block; border-radius: 999px; padding: .1rem .65rem; margin: .2rem .3rem 0 0; font-size: .85rem; color: #fca5a5; background: rgba(220,38,38,.15); border: 1px solid rgba(220,38,38,.35);}
.muted {opacity: .7;}
.hl {border-radius: 4px; padding: 0 .15rem;}
.hero .notice {color: #fde68a; font-weight: 600; margin-top: .55rem; padding-top: .55rem; border-top: 1px solid rgba(255,255,255,.18);}
</style>
"""

st.set_page_config(
    page_title="Arabic Toxicity Across Dialects",
    page_icon=":material/shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(APP_CSS, unsafe_allow_html=True)


def arabic_block(text):
    body = html.escape(str(text)).replace("\n", "<br>")
    st.markdown(f'<div class="ar-text" dir="auto">{body}</div>', unsafe_allow_html=True)


if "hide_sexual_examples" not in st.session_state:
    st.session_state.hide_sexual_examples = True
with st.sidebar:
    st.markdown("### Display")
    hide_sexual_examples = st.toggle(
        "Hide sexual examples",
        key="hide_sexual_examples",
        help="Saved sentences tagged Sexual, or clearly sexual in the wording, stay off the screen. The Sexual score and F1 stay in the tables. Text you paste yourself is still shown.",
    )
    st.divider()
    st.markdown("### About")
    st.markdown(
        "**Dataset:** AraTox (`Alshargi/Aratox`)  \n"
        f"**Labels:** {', '.join(TOXIC_LABELS)}  \n"
        "**Toxicity models:** TF-IDF + Linear SVM, QARiB  \n"
        "**Dialect ID:** CAMeLBERT-Mix NADI (exploratory only)"
    )
    with st.expander("What this project is asking"):
        st.markdown(PROJECT_FRAMING)
        st.markdown(
            f"- {PRIMARY_RESEARCH_QUESTION}\n"
            f"- {RESPONSIBLE_RESEARCH_QUESTION}\n"
            f"- {THRESHOLD_RESEARCH_QUESTION}"
        )
    st.divider()
    st.caption(HUMAN_REVIEW)
if st.session_state.get("hide_sexual_seen") != hide_sexual_examples:
    st.session_state.hide_sexual_seen = hide_sexual_examples
    st.session_state.comparison_example_index = 0
    st.session_state.review_index = 0

st.markdown(
    '<div class="hero">'
    "<h1>Arabic Toxicity Detection Across Dialects</h1>"
    "<p>Start on <b>Analyze</b> to check one sentence. "
    "Use <b>Model Comparison</b> and <b>Responsible AI</b> when you need to choose a model or a cutoff. "
    "Use <b>Error Analysis</b> to record why a false alarm happened.</p>"
    '<span class="pill">AraTox</span><span class="pill">6 toxicity labels</span>'
    '<span class="pill">TF-IDF + Linear SVM</span><span class="pill">QARiB</span>'
    f'<div class="notice">{html.escape(HUMAN_REVIEW)}</div>'
    "</div>",
    unsafe_allow_html=True,
)


def headline_metrics():
    test_path = ROOT / "outputs" / "final" / "test_summary.csv"
    if test_path.is_file():
        test = read_table(test_path).set_index("key")
        primary = test[test.primary].iloc[0]
        cards = st.columns(4)
        cards[0].metric("Primary system · test Macro-F1", f"{primary.macro_f1:.3f}", f"{primary.macro_f1 - test.loc['svm_reported'].macro_f1:+.3f} vs TF-IDF")
        cards[1].metric("QARiB · test Macro-F1", f"{test.loc['qarib_reported'].macro_f1:.3f}")
        cards[2].metric("TF-IDF · test Macro-F1", f"{test.loc['svm_reported'].macro_f1:.3f}")
        cards[3].metric("Primary system · false positives", f"{int(primary.false_positives):,}", f"{int(primary.false_positives - test.loc['qarib_reported'].false_positives):+,} vs QARiB", delta_color="inverse")
        st.caption(f"Held-out AraTox test split (n={int(primary.n):,}), scored once after all choices were frozen on validation. Primary system: {primary.system}. Details on the Final Results tab.")
        return
    metrics = ROOT / "outputs" / "metrics"
    summary_path = metrics / "tfidf_vs_qarib_summary.csv"
    latency_path = metrics / "tfidf_vs_qarib_latency.csv"
    if not summary_path.is_file():
        return
    summary = read_table(summary_path)
    tfidf = model_metrics(summary, SVM_NAME)
    qarib = model_metrics(summary, QARIB_NAME)
    cards = st.columns(4)
    cards[0].metric("QARiB Macro-F1", f"{qarib.macro_f1:.3f}", f"{qarib.macro_f1 - tfidf.macro_f1:+.3f} vs TF-IDF")
    cards[1].metric("TF-IDF Macro-F1", f"{tfidf.macro_f1:.3f}")
    if latency_path.is_file():
        latency = read_table(latency_path)
        cards[2].metric("QARiB latency", f"{model_metrics(latency, QARIB_NAME).mean_ms_per_sample:.2f} ms")
        cards[3].metric("TF-IDF latency", f"{model_metrics(latency, SVM_NAME).mean_ms_per_sample:.2f} ms")
    st.caption("Validation split only. Latency is per sample on this machine.")


headline_metrics()

cfg = load_yaml("configs/data.yaml")
prep = config_from_dict(cfg["preprocessing"])


@st.cache_resource
def load_qarib():
    try:
        return inference.load_qarib()
    except FileNotFoundError:
        return None


@st.cache_resource
def load_svm():
    return inference.load_svm()


@st.cache_data
def load_calibration():
    return inference.load_calibration()


def run_system(texts_norm, system, policy):
    """Score normalized texts with the cached models. Raises FileNotFoundError if an artifact is missing."""
    qarib = None
    if inference.needs_qarib(system):
        qarib = load_qarib()
        if qarib is None:
            raise FileNotFoundError("Run scripts/train_qarib.py first.")
    svm = load_svm() if system != inference.QARIB_SYSTEM else None
    calibration = load_calibration() if inference.needs_calibration(system, policy) else None
    return inference.score_texts(texts_norm, system, policy, svm=svm, qarib=qarib, calibration=calibration)


@st.cache_resource
def load_dialect():
    cfg = load_yaml("configs/dialect.yaml")["dialect"]
    if not cfg.get("enabled", True):
        return None
    return ArabicDialectClassifier(
        model_name=cfg["model_name"],
        low_arabic_ratio_threshold=float(cfg["low_arabic_ratio_threshold"]),
        max_length=int(cfg["max_length"]),
        batch_size=int(cfg["batch_size"]),
    )


def dialect_for(text):
    try:
        classifier = load_dialect()
    except Exception as exc:
        return {"error": str(exc)}
    if classifier is None:
        return {"error": "Dialect classification is disabled in configs/dialect.yaml."}
    try:
        return classifier.predict(text)
    except Exception as exc:
        return {"error": str(exc)}


def pretty_labels(value):
    text = str(value or "").strip()
    if not text or text == "none":
        return "None"
    return text.replace("|", ", ")


def show_dialect(dialect):
    if not dialect or dialect.get("error"):
        st.error(dialect.get("error") if dialect else "Dialect prediction is unavailable.")
        return
    if dialect.get("dialect_status") == "empty_input":
        st.write("No text to classify.")
        return
    country, region = st.columns(2)
    country.metric("Predicted dialect", dialect["predicted_country_dialect"])
    region.metric("Region", dialect["predicted_region"])
    confidence = dialect.get("model_confidence")
    st.caption(
        f"Confidence {confidence:.3f}. " if confidence is not None else "Confidence unavailable. "
    )
    st.caption("This is a guess about writing style. It is not nationality, and it did not change the toxicity decision.")
    if dialect.get("dialect_status") == "low_confidence_input":
        st.info("Little Arabic script was found, so treat this dialect guess as weak.")


def save_rating(record):
    REVIEWS.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([record])
    if REVIEWS.exists():
        prev = pd.read_csv(REVIEWS)
        prev = prev[prev.review_id != record["review_id"]]
        row = pd.concat([prev, row], ignore_index=True)
    row.to_csv(REVIEWS, index=False)


def show_result(result):
    st.subheader("Result")
    flagged = [label for label, value in zip(TOXIC_LABELS, result["pred"]) if value]
    if flagged:
        chips = "".join(f'<span class="chip">{html.escape(label)}</span>' for label in flagged)
    else:
        chips = '<span class="muted">No toxicity label was flagged.</span>'
    tone = "toxic" if result["binary"] == "Toxic" else "clean"
    st.markdown(
        f'<div class="verdict {tone}">'
        f'<div class="kicker">Decision · {html.escape(result["model"])} · {html.escape(result.get("policy", inference.REPORTED_POLICY))}</div>'
        f'<div class="value">{html.escape(result["binary"])}</div>'
        f"<div>{chips}</div></div>",
        unsafe_allow_html=True,
    )
    details = pd.DataFrame(
        {
            "Label": list(TOXIC_LABELS),
            "Flagged": ["Yes" if value else "No" for value in result["pred"]],
            "Score": [float(score) for score in result["scores"]],
            "Cutoff": [float(value) for value in result.get("thresholds", [0.5] * len(TOXIC_LABELS))],
        }
    )
    st.dataframe(
        details,
        hide_index=True,
        width="stretch",
        column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0.0, max_value=1.0, format="%.3f"),
            "Cutoff": st.column_config.NumberColumn("Cutoff", format="%.2f"),
        },
    )
    st.caption(result["score_note"] + " Do not compare a TF-IDF score with a QARiB score as if they were the same kind of probability.")
    show_explanation(result)
    with st.expander("Normalized text"):
        arabic_block(result["text_norm"])
    with st.expander("Predicted dialect (separate from the toxicity decision)"):
        show_dialect(result.get("dialect"))
    st.write("Was this decision right?")
    vote = st.feedback("thumbs", key=f"vote_{result['review_id']}")
    if vote is None:
        return
    rating = "up" if vote == 1 else "down"
    if st.session_state.get("saved_vote") != (result["review_id"], rating):
        save_rating({
            "review_id": result["review_id"],
            "rated_at": datetime.now(timezone.utc).isoformat(),
            "model": result["model"],
            "text_raw": result["text_raw"],
            "text_norm": result["text_norm"],
            "binary_status": result["binary"],
            "rating": rating,
            "labels": json.dumps({label: int(value) for label, value in zip(TOXIC_LABELS, result["pred"])}, ensure_ascii=False),
            "scores": json.dumps({label: float(score) for label, score in zip(TOXIC_LABELS, result["scores"])}, ensure_ascii=False),
        })
        st.session_state.saved_vote = (result["review_id"], rating)
    st.success("Thumbs up saved." if rating == "up" else "Thumbs down saved for review.")


@st.cache_data(show_spinner="Measuring which words drive the score…")
def occlusion_for(text_norm, system, policy):
    words, base, deltas = word_occlusion(text_norm, lambda variants: run_system(variants, system, policy)[0])
    return words, base, deltas


def highlight_html(words, influence):
    peak = max(float(np.abs(influence).max()), 1e-6) if len(influence) else 1.0
    spans = []
    for word, value in zip(words, influence):
        strength = min(abs(float(value)) / peak, 1.0)
        color = f"rgba(239,68,68,{0.15 + 0.6 * strength:.2f})" if value > 0 else f"rgba(74,222,128,{0.1 + 0.35 * strength:.2f})"
        if abs(value) < 0.01:
            color = "transparent"
        spans.append(f'<span class="hl" style="background:{color}" title="{float(value):+.3f}">{html.escape(word)}</span>')
    return '<div class="ar-text" dir="auto">' + " ".join(spans) + "</div>"


def show_explanation(result):
    with st.expander("Why this decision? Words that move the score", expanded=result["binary"] == "Toxic"):
        try:
            words, _, deltas = occlusion_for(result["text_norm"], result["model"], result.get("policy", inference.REPORTED_POLICY))
        except FileNotFoundError as exc:
            st.error(str(exc))
            return
        if not words:
            st.write("No words to explain.")
            return
        scores = np.asarray(result["scores"])
        flagged = [i for i, value in enumerate(result["pred"]) if value]
        default = flagged[int(np.argmax(scores[flagged]))] if flagged else int(np.argmax(scores))
        label = st.selectbox("Label to explain", list(TOXIC_LABELS), index=default, key=f"explain_{result['review_id']}")
        index = TOXIC_LABELS.index(label)
        st.markdown(highlight_html(words, deltas[:, index]), unsafe_allow_html=True)
        strongest = top_words(words, deltas, index, k=3)
        if strongest:
            st.caption("Strongest push toward " + label + ": " + ", ".join(f"{word} ({delta:+.2f})" for word, delta in strongest))
        st.caption(
            "Red words raise this label's score; green words lower it. Each word is removed in turn and the score change is measured "
            "(word occlusion). This shows what the model is sensitive to. It is not the model's reasoning and not proof of intent."
        )


def score_pair(raw_text):
    norm = normalize_arabic(raw_text, prep)
    svm_scores, svm_pred, _ = run_system([norm], inference.SVM_SYSTEM, inference.REPORTED_POLICY)
    qarib_scores, qarib_pred, _ = run_system([norm], inference.QARIB_SYSTEM, inference.REPORTED_POLICY)
    return {
        "text_norm": norm,
        "svm": {"pred": svm_pred[0], "scores": svm_scores[0], "binary": bool(svm_pred[0].sum())},
        "qarib": {"pred": qarib_pred[0], "scores": qarib_scores[0], "binary": bool(qarib_pred[0].sum())},
    }


@st.cache_data
def load_saved_examples(path: str):
    return pd.read_csv(path)


def _prediction_column(column, title, predicted, gold, scores, score_note, differs):
    column = column.container(border=True)
    column.subheader(title)
    toxic = binary_toxic(predicted)
    gold_toxic = None if gold is None else binary_toxic(gold)
    binary_mark = correctness_mark(None if gold_toxic is None else toxic == gold_toxic)
    column.write(f"Binary: {'Toxic' if toxic else 'Non-toxic'}{(' ' + binary_mark) if binary_mark else ''}")
    for label in TOXIC_LABELS:
        yes = yes_no(predicted[label])
        mark = correctness_mark(None if gold is None else predicted[label] == gold[label])
        score_bit = f" ({float(scores[label]):.3f})" if scores is not None else ""
        line = f"{label}: {yes}{score_bit}{(' ' + mark) if mark else ''}"
        if label in differs:
            column.markdown(f"**{line}**")
            column.caption("The models disagree on this label.")
        else:
            column.write(line)
    if score_note:
        column.caption(score_note)


def comparison_tab_view():
    metrics = ROOT / "outputs" / "metrics"
    errors = ROOT / "outputs" / "errors"
    st.caption("Validation results only. QARiB is ahead on F1. TF-IDF is much faster and still competitive.")
    st.subheader("What TF-IDF and QARiB are")
    st.markdown(
        "TF-IDF + SVM learns which words and word combinations are associated with toxicity.\n\n"
        "QARiB uses a pretrained Arabic transformer and represents words in context.\n\n"
        "The comparison shows whether contextual language modeling improves toxicity classification "
        "and for which toxicity categories."
    )
    missing = missing_comparison_files(metrics, errors)
    required = {"tfidf_vs_qarib_summary.csv", "tfidf_vs_qarib_per_label.csv"}
    if required & set(missing):
        st.info("Run `uv run python scripts/compare_models.py --skip-latency` to create the validation comparison. Missing: " + ", ".join(missing))
        return
    summary = read_table(metrics / "tfidf_vs_qarib_summary.csv")
    per_label = read_table(metrics / "tfidf_vs_qarib_per_label.csv")
    tfidf = model_metrics(summary, SVM_NAME)
    qarib = model_metrics(summary, QARIB_NAME)
    ranked = sorted_label_improvements(per_label)
    gains = overall_improvement(tfidf, qarib)

    with st.container(border=True):
        st.markdown("**Key Finding**")
        st.markdown(key_finding_text(float(qarib.macro_f1), float(tfidf.macro_f1), ranked.label.tolist()))
        st.caption("Validation split only. These figures are not a claim about other datasets or machines.")

    st.subheader("Overall Performance")
    st.caption("Validation split only. Scores are read from the saved comparison files.")
    overall = st.columns(4)
    overall[0].metric(f"{SVM_NAME} · Macro-F1", f"{tfidf.macro_f1:.3f}")
    overall[1].metric(f"{QARIB_NAME} · Macro-F1", f"{qarib.macro_f1:.3f}", f"{gains['macro_f1']:+.3f}")
    overall[2].metric(f"{SVM_NAME} · Micro-F1", f"{tfidf.micro_f1:.3f}")
    overall[3].metric(f"{QARIB_NAME} · Micro-F1", f"{qarib.micro_f1:.3f}", f"{gains['micro_f1']:+.3f}")
    st.write(OVERALL_INTERPRETATION)

    st.subheader("Exact-Match Disagreement")
    summary_path = metrics / "tfidf_vs_qarib_disagreement_summary.json"
    disagreement = None
    if summary_path.is_file():
        disagreement = disagreement_stats(read_json(summary_path))
        cards = st.columns(4)
        cards[0].metric("Both correct", f"{disagreement['both_correct']:,}")
        cards[1].metric("QARiB only correct", f"{disagreement['qarib_only_correct']:,}")
        cards[2].metric("TF-IDF only correct", f"{disagreement['tfidf_only_correct']:,}")
        cards[3].metric("Both wrong", f"{disagreement['both_wrong']:,}")
        st.write(
            f"Models disagree on {disagreement['disagree']:,} of {disagreement['n']:,} "
            f"validation examples ({disagreement['disagree_percent']:.1f}%)."
        )
        st.write(
            "Among examples where only one model is exactly correct, "
            f"QARiB solves {disagreement['qarib_exact_advantage']:,} more examples than TF-IDF."
        )
        st.caption("Exact-match means all six toxicity labels match. This count is not a significance test.")
    else:
        st.info("The disagreement summary file is not available.")

    st.subheader("Per-Label F1")
    chart = ROOT / "outputs" / "figures" / "tfidf_vs_qarib_per_label_f1.png"
    if chart.exists():
        st.image(str(chart))
    st.write(per_label_interpretation(ranked))
    top = ranked.head(2)
    gain_cols = st.columns(2)
    for column, row in zip(gain_cols, top.itertuples()):
        column.metric(f"Largest gain: {row.label}", f"{row.f1_difference:+.3f}")
    shown = ranked[["label", "tfidf_f1", "qarib_f1", "f1_difference"]].rename(columns={
        "label": "Label",
        "tfidf_f1": "TF-IDF F1",
        "qarib_f1": "QARiB F1",
        "f1_difference": "Difference",
    })
    shown["TF-IDF F1"] = shown["TF-IDF F1"].map(lambda value: f"{value:.3f}")
    shown["QARiB F1"] = shown["QARiB F1"].map(lambda value: f"{value:.3f}")
    shown["Difference"] = shown["Difference"].map(lambda value: f"{value:+.3f}")
    st.dataframe(shown, hide_index=True, width="stretch")

    st.subheader("False-Positive / Error Trade-offs")
    error_path = metrics / "error_counts_by_label.csv"
    if error_path.is_file():
        fewer = fewer_tfidf_false_positives(read_table(error_path))
        if fewer.empty:
            st.write("TF-IDF did not produce fewer false positives on any saved label.")
        else:
            st.markdown("**Precision / False-Positive Trade-off**")
            for row in fewer.itertuples():
                st.write(
                    f"{row.label}: TF-IDF {int(row.tfidf_false_positives)} vs QARiB {int(row.qarib_false_positives)}"
                )
        st.write(FALSE_POSITIVE_INTERPRETATION)
        with st.expander("False-positive and false-negative counts"):
            st.dataframe(read_table(error_path), hide_index=True, width="stretch")
    else:
        st.info("The error-count file is not available.")

    st.subheader("Latency and Deployment Trade-off")
    latency_path = metrics / "tfidf_vs_qarib_latency.csv"
    if latency_path.is_file():
        latency = read_table(latency_path)
        tfidf_time = model_metrics(latency, SVM_NAME)
        qarib_time = model_metrics(latency, QARIB_NAME)
        time_cols = st.columns(2)
        for column, row in ((time_cols[0], tfidf_time), (time_cols[1], qarib_time)):
            box = column.container(border=True)
            box.markdown(f"**{row.model}**")
            speed, throughput = box.columns(2)
            speed.metric("ms / sample", f"{row.mean_ms_per_sample:.2f}")
            throughput.metric("samples / second", f"{row.samples_per_second:,.0f}")
            box.caption(f"{device_label(row.device)} · n={int(row.n)}")
        ratio = displayed_latency_ratio(float(qarib_time.mean_ms_per_sample), float(tfidf_time.mean_ms_per_sample))
        st.write(f"Observed latency ratio on this machine: {ratio:.1f}×")
        st.warning(LATENCY_WARNING)
    else:
        st.info("Latency has not been measured yet.")

    st.subheader("Side-by-Side Prediction")
    source = st.selectbox("Example source", EXAMPLE_SOURCES, key="comparison_example_source")
    if st.session_state.get("comparison_source_seen") != source:
        st.session_state.comparison_source_seen = source
        st.session_state.comparison_example_index = 0
    if source == "Custom text":
        sentence = st.text_area("Arabic sentence", key="comparison_text")
        if st.button("Compare models", key="compare_models"):
            if not sentence.strip():
                st.session_state.comparison = None
                st.warning("Enter Arabic text first.")
            else:
                try:
                    st.session_state.comparison = score_pair(sentence)
                except FileNotFoundError as exc:
                    st.session_state.comparison = None
                    st.error(str(exc))
        compared = st.session_state.get("comparison")
        if compared:
            tfidf_flags = {label: bool(value) for label, value in zip(TOXIC_LABELS, compared["svm"]["pred"])}
            qarib_flags = {label: bool(value) for label, value in zip(TOXIC_LABELS, compared["qarib"]["pred"])}
            differs = {label for label in TOXIC_LABELS if tfidf_flags[label] != qarib_flags[label]}
            svm_col, qarib_col = st.columns(2)
            _prediction_column(svm_col, "TF-IDF + SVM", tfidf_flags, None, compared["svm"]["scores"], TFIDF_SCORE_NOTE, differs)
            _prediction_column(qarib_col, "QARiB", qarib_flags, None, compared["qarib"]["scores"], QARIB_SCORE_NOTE, differs)
    else:
        example_path = errors / "tfidf_vs_qarib_disagreements.csv"
        if not example_path.is_file():
            st.info("The disagreement example file is not available.")
        else:
            chosen = filter_validation_examples(load_saved_examples(str(example_path)), SOURCE_TO_GROUP[source])
            full_count = len(chosen)
            chosen = without_sexual_examples(chosen, hide_sexual_examples)
            count = len(chosen)
            if hide_sexual_examples and full_count > count:
                st.caption(f"Hiding {full_count - count:,} sexual examples in this group. The Sexual label is still included in the scores above.")
            if count == 0:
                st.info("No saved validation examples are in this group.")
            else:
                index = int(st.session_state.get("comparison_example_index", 0)) % count
                previous, following, random_choice = st.columns(3)
                if previous.button("Previous example"):
                    index = step_example_index(index, count, -1)
                if following.button("Next example"):
                    index = step_example_index(index, count, 1)
                if random_choice.button("Random example"):
                    index = random.randrange(count)
                st.session_state.comparison_example_index = index
                row = chosen.iloc[index]
                view = example_label_view(row)
                st.caption(f"Example {index + 1} of {count:,}")
                arabic_block(row.text)
                st.markdown("**Ground Truth**")
                truth_left, truth_right = st.columns(2)
                for position, item in enumerate(view):
                    target = truth_left if position < 3 else truth_right
                    target.write(f"{item['label']}: {yes_no(item['gold'])}")
                gold = {item["label"]: item["gold"] for item in view}
                tfidf_flags = {item["label"]: item["tfidf"] for item in view}
                qarib_flags = {item["label"]: item["qarib"] for item in view}
                differs = {item["label"] for item in view if item["models_differ"]}
                svm_col, qarib_col = st.columns(2)
                _prediction_column(svm_col, "TF-IDF + SVM", tfidf_flags, gold, None, None, differs)
                _prediction_column(qarib_col, "QARiB", qarib_flags, gold, None, None, differs)
    st.caption(SCORE_MAGNITUDE_NOTE)

    st.subheader("Disagreement Examples")
    if disagreement is None:
        st.info("Disagreement examples need the saved validation comparison files.")
    else:
        st.write(
            f"QARiB alone is exactly correct on {disagreement['qarib_only_correct']:,} examples, "
            f"TF-IDF alone on {disagreement['tfidf_only_correct']:,}, "
            f"and both are wrong on {disagreement['both_wrong']:,}."
        )
        st.caption("Use Side-by-Side Prediction to open one saved example with its ground truth. The table below is a short preview.")
        preview_group = SOURCE_TO_GROUP.get(source, "qarib_only_correct")
        preview_path = errors / "tfidf_vs_qarib_disagreements.csv"
        if preview_path.is_file():
            preview = without_sexual_examples(
                filter_validation_examples(load_saved_examples(str(preview_path)), preview_group),
                hide_sexual_examples,
            )
            with st.expander(f"Preview: {source if source != 'Custom text' else 'QARiB correct / TF-IDF wrong'}"):
                st.dataframe(
                    preview[["text", "true_labels", "tfidf_predicted_labels", "qarib_predicted_labels"]].head(15),
                    hide_index=True,
                    width="stretch",
                )

    st.subheader("What Does This Comparison Show?")
    st.markdown(
        "TF-IDF + SVM learns sparse word and phrase patterns from the training data.\n\n"
        "QARiB uses contextual Arabic representations from a pretrained transformer.\n\n"
        "On this validation set, QARiB achieves higher F1 across all six toxicity labels, "
        "but the gains are relatively small and TF-IDF remains competitive.\n\n"
        "This creates a practical trade-off between predictive performance, inference cost, and model complexity."
    )
    with st.container(border=True):
        st.markdown("**Research Conclusion**")
        st.markdown(
            "The results support the hypothesis that contextual Arabic representations improve "
            "multi-label toxicity detection over a classical TF-IDF baseline.\n\n"
            "However, the improvement is incremental rather than dramatic, suggesting that classical "
            "sparse features remain a strong baseline for this dataset.\n\n"
            "The main advantage of QARiB appears in categories such as Racial, Violence, Appearance, and Hatred."
        )
        st.caption("Validation split only. This conclusion does not extend to the held-out test set or to other datasets.")


BATCH_LIMIT = 5000


def batch_view():
    st.subheader("Score a file")
    st.caption(f"Upload a CSV with one comment per row (up to {BATCH_LIMIT:,} rows). The model and cutoff chosen above are used. Nothing is stored.")
    upload = st.file_uploader("CSV file", type=["csv"], key="batch_file")
    if upload is None:
        return
    try:
        frame = pd.read_csv(upload, encoding="utf-8-sig")
    except Exception as exc:
        st.error(f"Could not read the file as CSV: {exc}")
        return
    text_columns = [column for column in frame.columns if frame[column].dtype == object] or list(frame.columns)
    column = st.selectbox("Text column", text_columns, key="batch_column")
    if len(frame) > BATCH_LIMIT:
        st.warning(f"Only the first {BATCH_LIMIT:,} of {len(frame):,} rows will be scored.")
        frame = frame.head(BATCH_LIMIT)
    if not st.button(f"Score {len(frame):,} rows with {choice} · {policy}", key="batch_run"):
        return
    texts = frame[column].fillna("").astype(str).tolist()
    norms = [normalize_arabic(text, prep) for text in texts]
    try:
        with st.spinner("Scoring…"):
            scores, pred, note = run_system(norms, choice, policy)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    out = pd.DataFrame({"text": texts, "decision": np.where(pred.sum(1) > 0, "Toxic", "Non-toxic")})
    out["flagged_labels"] = [", ".join(label for label, value in zip(TOXIC_LABELS, row) if value) or "None" for row in pred]
    for i, label in enumerate(TOXIC_LABELS):
        out[f"score_{label}"] = scores[:, i].round(4)
    toxic = int((out.decision == "Toxic").sum())
    cards = st.columns(3)
    cards[0].metric("Rows scored", f"{len(out):,}")
    cards[1].metric("Flagged toxic", f"{toxic:,}", f"{toxic / max(len(out), 1):.0%} of rows", delta_color="off")
    cards[2].metric("Most frequent label", pd.Series([l for row in pred for l, v in zip(TOXIC_LABELS, row) if v]).mode().iat[0] if pred.sum() else "None")
    st.dataframe(out, hide_index=True, width="stretch")
    st.caption(note + " Every flag still needs human review.")
    st.download_button(
        "Download results CSV",
        out.to_csv(index=False).encode("utf-8-sig"),
        file_name="toxicity_scores.csv",
        mime="text/csv",
        type="primary",
    )


def _ci_text(row, prefix="macro_f1"):
    return f"{row[prefix]:.3f} [{row[prefix + '_ci_low']:.3f}, {row[prefix + '_ci_high']:.3f}]"


def jais_view(final, metrics):
    st.subheader("Jais-2 (prompted LLM, no training)")
    val_path = metrics / "jais_validation_summary.csv"
    test_path = final / "jais_test_results.json"
    if not val_path.is_file():
        st.info("Jais-2 has not been evaluated yet. See the README for the gated-model login and `just jais-validation`.")
        return
    val = read_table(val_path)
    st.dataframe(
        val[["mode", "macro_f1", "macro_f1_parsed_rows_only", "parse_failure_rate", "median_latency_seconds", "n"]].rename(columns={
            "mode": "Prompt", "macro_f1": "Macro-F1", "macro_f1_parsed_rows_only": "Macro-F1 (parsed rows only)",
            "parse_failure_rate": "Unparseable answers", "median_latency_seconds": "Median seconds / comment", "n": "Validation rows",
        }),
        hide_index=True,
        width="stretch",
        column_config={
            "Macro-F1": st.column_config.NumberColumn(format="%.3f"),
            "Macro-F1 (parsed rows only)": st.column_config.NumberColumn(format="%.3f"),
            "Unparseable answers": st.column_config.NumberColumn(format="percent"),
            "Median seconds / comment": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.caption("Validation, first 500 rows. Unparseable answers count as no label. The prompt mode with the higher Macro-F1 was frozen before the test run.")
    if not test_path.is_file():
        st.info("The frozen Jais prompt has not been scored on test yet.")
        return
    results = read_json(test_path)
    jais = results["jais"]
    comparison = pd.DataFrame(results["comparison_same_rows"]).sort_values("macro_f1", ascending=False)
    cards = st.columns(3)
    cards[0].metric(f"{results['mode']} · test Macro-F1", f"{jais['macro_f1']:.3f}")
    cards[1].metric("Unparseable answers", f"{jais['parse_failure_rate']:.1%}")
    cards[2].metric("Median seconds / comment", f"{jais['median_latency_seconds']:.2f}")
    comparison["Macro-F1 [95% CI]"] = [_ci_text(row) for _, row in comparison.iterrows()]
    st.dataframe(comparison[["system", "Macro-F1 [95% CI]"]].rename(columns={"system": "System"}), hide_index=True, width="stretch")
    st.caption(results["subset"] + " Brackets are 95% paired-bootstrap intervals on those rows.")


def final_results_view():
    final = ROOT / "outputs" / "final"
    metrics = ROOT / "outputs" / "metrics"
    figures = ROOT / "outputs" / "figures"
    st.caption("Everything on this page was produced by the scripts in `scripts/`. Brackets are 95% paired-bootstrap confidence intervals (2,000 resamples).")

    st.subheader("Held-out test results")
    results_path = final / "test_results.json"
    if not results_path.is_file():
        st.info("The test split has not been scored yet. Run `scripts/fit_calibration.py`, `scripts/freeze_decisions.py`, then `scripts/final_test_evaluation.py` once.")
    else:
        results = read_json(results_path)
        summary = read_table(final / "test_summary.csv")
        boot = read_table(final / "test_bootstrap.csv").set_index("system")
        primary = summary[summary.primary].iloc[0]
        vs_qarib = results["primary_vs_qarib"][0]
        vs_tfidf = boot.loc[primary.system]
        with st.container(border=True):
            st.markdown(f"**Primary system (chosen on validation before the test run): {primary.system}**")
            cards = st.columns(4)
            cards[0].metric("Test Macro-F1", f"{primary.macro_f1:.3f}")
            cards[1].metric("vs TF-IDF", f"{vs_tfidf.macro_f1_diff_vs_reference:+.3f}", f"95% CI {vs_tfidf.diff_ci_low:+.3f} to {vs_tfidf.diff_ci_high:+.3f}", delta_color="off")
            cards[2].metric("vs QARiB", f"{vs_qarib['macro_f1_diff_vs_reference']:+.3f}", f"95% CI {vs_qarib['diff_ci_low']:+.3f} to {vs_qarib['diff_ci_high']:+.3f}", delta_color="off")
            cards[3].metric("False positives", f"{int(primary.false_positives):,}", f"of {int(primary.n):,} comments × 6 labels", delta_color="off")
            st.caption(f"Scored once on {results['scored_at'][:10]} (n={results['n']:,}). Decisions were frozen on {results['frozen_at'][:10]}; the fingerprints are in `outputs/final/frozen_decisions.json`.")
        table = summary.assign(
            **{"Macro-F1 [95% CI]": [_ci_text(boot.loc[name]) for name in summary.system],
               "Micro-F1 [95% CI]": [_ci_text(boot.loc[name], "micro_f1") for name in summary.system]}
        )[["system", "Macro-F1 [95% CI]", "Micro-F1 [95% CI]", "subset_accuracy", "binary_macro_f1", "false_positives", "false_negatives"]]
        st.dataframe(
            table.rename(columns={"system": "System", "subset_accuracy": "Subset accuracy", "binary_macro_f1": "Binary Macro-F1", "false_positives": "False positives", "false_negatives": "False negatives"}),
            hide_index=True,
            width="stretch",
            column_config={"Subset accuracy": st.column_config.NumberColumn(format="%.3f"), "Binary Macro-F1": st.column_config.NumberColumn(format="%.3f")},
        )
        chart = figures / "final_test_macro_f1.png"
        if chart.exists():
            st.image(str(chart))
        st.markdown(
            "- The ensemble's gain over each single model is small but its confidence interval excludes zero.\n"
            "- Tuning cutoffs per label made TF-IDF slightly worse on both validation and test. Treat tuned cutoffs as a policy choice, not a free gain.\n"
            "- int8 QARiB stays within about 0.006 Macro-F1 of the full model at half the size."
        )
        st.caption(results["jais"])

    jais_view(final, metrics)

    st.subheader("Is QARiB's lead real? (validation)")
    boot_val = metrics / "bootstrap_validation.csv"
    if boot_val.is_file():
        val = read_table(boot_val)
        shown = val.assign(**{"Macro-F1 [95% CI]": [_ci_text(row) for _, row in val.iterrows()]})
        shown["Gap vs TF-IDF [95% CI]"] = [
            "reference" if pd.isna(row.get("diff_ci_low")) else f"{row.macro_f1_diff_vs_reference:+.3f} [{row.diff_ci_low:+.3f}, {row.diff_ci_high:+.3f}]"
            for _, row in val.iterrows()
        ]
        st.dataframe(shown[["system", "estimate", "Macro-F1 [95% CI]", "Gap vs TF-IDF [95% CI]"]].rename(columns={"system": "System", "estimate": "Estimate"}), hide_index=True, width="stretch")
        st.caption("Tuned systems use 5-fold cross-fitting on validation, so a cutoff is never scored on the rows it was tuned on. This is how the primary system was chosen.")
    else:
        st.info("Run `uv run python scripts/fit_calibration.py`.")

    st.subheader("Calibration")
    ece_path = metrics / "calibration_validation.csv"
    if ece_path.is_file():
        ece = read_table(ece_path)
        cols = st.columns(len(ece))
        for column, row in zip(cols, ece.itertuples()):
            before = "" if pd.isna(row.ece_before) else f"from {row.ece_before:.3f}"
            column.metric(f"{row.model} · calibration error", f"{row.ece_after_platt:.3f}", before, delta_color="off")
        if (figures / "reliability_validation.png").exists():
            st.image(str(figures / "reliability_validation.png"))
        st.caption(
            "Expected calibration error: the average gap between a score and how often comments with that score are truly toxic (lower is better). "
            "Platt scaling was fitted on validation. With calibrated scores, 0.80 means roughly 80% of such comments carry the label."
        )

    st.subheader("Robustness stress test")
    rob_path = metrics / "robustness_validation.csv"
    if rob_path.is_file():
        rob = read_table(rob_path)
        rob = rob[rob.perturbation != "Original"]
        pivot = rob.pivot(index="perturbation", columns="system", values="macro_f1_change")
        st.dataframe(pivot.style.format("{:+.3f}"), width="stretch")
        st.caption("Change in Macro-F1 after the perturbation (1,000 validation comments; labels unchanged).")
        if (figures / "robustness_flip_rate.png").exists():
            st.image(str(figures / "robustness_flip_rate.png"))
        st.markdown(
            "- **Arabizi** (Arabic typed in Latin letters) breaks every model: none saw it in training.\n"
            "- **Elongated letters** and **typos** cost about 0.06–0.10 Macro-F1. Users who stretch or misspell insults can evade the filter.\n"
            "- **Diacritics** and **tatweel** have no effect because preprocessing removes them.\n"
            "- **Emoji** and a few **English words** barely matter."
        )

    st.subheader("Smaller QARiB (int8)")
    q_path = metrics / "qarib_quantization_validation.csv"
    if q_path.is_file():
        q = read_table(q_path)
        st.dataframe(
            q[["variant", "macro_f1", "label_agreement_with_fp32", "size_mb", "ms_per_sample_batch1", "ms_per_sample_batch16"]].rename(columns={
                "variant": "Variant", "macro_f1": "Macro-F1", "label_agreement_with_fp32": "Agreement with fp32",
                "size_mb": "Size (MB)", "ms_per_sample_batch1": "ms / sentence (one at a time)", "ms_per_sample_batch16": "ms / sentence (batches of 16)",
            }),
            hide_index=True,
            width="stretch",
            column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ("Macro-F1", "Agreement with fp32")} | {
                c: st.column_config.NumberColumn(format="%.1f") for c in ("Size (MB)", "ms / sentence (one at a time)", "ms / sentence (batches of 16)")
            },
        )
        st.caption("Dynamic int8 quantization of the Linear layers, no retraining. It helps one-sentence CPU use (no GPU) and halves the file size; batch throughput on Apple Silicon is better on the GPU in fp32.")

    frozen_path = final / "frozen_decisions.json"
    if frozen_path.is_file():
        with st.expander("Frozen decisions and file fingerprints"):
            st.json(read_json(frozen_path))


def dialect_check_view():
    path = ROOT / "outputs" / "ethics" / "dialect_check_sample.csv"
    st.subheader("Check the dialect model by hand")
    st.caption("Label each comment's region yourself. The model's guess is hidden until you save, so it cannot anchor your answer.")
    if not path.is_file():
        st.info("Run `uv run python scripts/build_dialect_check_sample.py` to create the 100-comment check sheet.")
        return
    sheet = pd.read_csv(path, dtype=str).fillna("")
    stats = agreement(sheet)
    st.progress(stats["labelled"] / max(stats["total"], 1))
    cards = st.columns(3)
    cards[0].metric("Labelled", f"{stats['labelled']} / {stats['total']}")
    cards[1].metric("Model agrees with you", "—" if stats["accuracy"] is None else f"{stats['accuracy']:.0%}", f"on {stats['scored']} scored rows", delta_color="off")
    cards[2].metric("MSA / can't tell", f"{stats['msa']} / {stats['cant_tell']}")
    pending = sheet[sheet.human_region.str.strip() == ""]
    if pending.empty:
        st.success("Every comment is labelled.")
    else:
        row = pending.iloc[0]
        with st.container(border=True):
            st.caption(f"Comment {int(row.check_id)} of {len(sheet)}")
            arabic_block(row.text)
            answer = st.radio("Which region does this read as?", HUMAN_CHOICES, index=None, horizontal=True, key=f"dialect_check_{row.check_id}")
            if st.button("Save and show the model's guess", type="primary", disabled=answer is None, key="dialect_check_save"):
                sheet.loc[sheet.check_id == row.check_id, "human_region"] = answer
                sheet.to_csv(path, index=False)
                st.session_state.dialect_check_last = (row.check_id, answer, row.predicted_country_dialect, row.predicted_region)
                st.rerun()
    last = st.session_state.get("dialect_check_last")
    if last:
        check_id, answer, country, region = last
        verdict = "agrees" if answer == region else ("not scored" if answer in ("MSA (formal Arabic)", "Can't tell") else "disagrees")
        st.info(f"Comment {check_id}: you said **{answer}**; the model guessed **{region}** ({country}) — {verdict}.")
    if stats["labelled"]:
        with st.expander("Agreement by predicted region and confusion table"):
            if stats["per_region"]:
                st.dataframe(pd.DataFrame(stats["per_region"]), hide_index=True, width="stretch")
            st.dataframe(confusion(sheet), width="stretch")
        st.download_button("Download check sheet", path.read_bytes(), file_name=path.name, mime="text/csv")
    st.caption("Your labels judge writing style only. They are not nationality, and they never change a toxicity score.")


def responsible_ai_view():
    ethics_dir = ROOT / "outputs" / "ethics"
    metrics = ROOT / "outputs" / "metrics"
    st.caption("Read the scorecard, then the three cutoff cards. Nothing on this page retrains a model or changes the reported 0.50 result.")
    st.subheader("Why ethics matters")
    st.markdown(
        "A wrong flag can hide a legitimate comment. A missed flag can leave toxic content up. "
        "This page shows those trade-offs on the validation set. It does not change either model."
    )
    with st.container(border=True):
        st.markdown("**Responsible AI Summary**")
        st.markdown(RESPONSIBLE_SUMMARY)

    st.subheader("Responsible Model Selection")
    try:
        selection = load_model_selection(metrics)
    except (FileNotFoundError, ValueError):
        selection = None
    if selection is None:
        st.info("Saved validation comparison files are required for the scorecard.")
    else:
        st.table(scorecard(selection).set_index("Dimension"))
        st.caption(JAIS_STATUS)
        st.caption("Latency cells are the measured validation subset on this machine. They are not a hardware-normalized benchmark.")
    st.subheader("What should we choose?")
    for block in (part for part in SCENARIO_GUIDANCE.split("\n\n") if part.strip()):
        title, _, body = block.partition("\n")
        with st.container(border=True):
            st.markdown(f"**{title.rstrip(':')}**")
            st.write(body)
    st.caption("This is an interpretation of the current validation project, not a universal recommendation.")

    st.subheader("Operating Policy Scenarios")
    st.caption("Same trained QARiB model, three cutoffs. Pick the card that matches the cost of a false alarm versus a miss.")
    modes_path = ethics_dir / "qarib_operating_modes.csv"
    tradeoff_path = ethics_dir / "qarib_threshold_tradeoff.csv"
    if tradeoff_path.is_file():
        modes = operating_mode_table(read_table(tradeoff_path)) if not modes_path.is_file() else read_table(modes_path)
        mode_columns = st.columns(3)
        for column, row in zip(mode_columns, modes.itertuples()):
            box = column.container(border=True)
            box.markdown(f"**{row.mode}**")
            box.metric("Threshold", f"{float(row.threshold):.2f}")
            box.markdown(
                f"Macro-F1 **{float(row.macro_f1):.3f}** · "
                f"Precision **{float(row.macro_precision):.3f}** · "
                f"Recall **{float(row.macro_recall):.3f}**  \n"
                f"False positives **{int(row.false_positives):,}** · "
                f"False negatives **{int(row.false_negatives):,}**"
            )
            box.caption(row.interpretation)
        mode_chart = ROOT / "outputs" / "figures" / "qarib_operating_modes.png"
        if mode_chart.exists():
            st.image(str(mode_chart))
        st.info(MODE_NOTE)
        st.caption("The reported result remains the 0.50 row. The 0.65 and 0.80 rows are exploratory policies.")
    else:
        st.info("Run `uv run python scripts/analyze_threshold_tradeoff.py` before building operating modes.")

    st.subheader("False-positive risk")
    st.markdown(
        "A false positive occurs when a non-toxic comment is incorrectly classified as toxic.\n\n"
        "In a deployed moderation workflow, that kind of mistake could mean legitimate content being flagged, "
        "unnecessary moderator review, reduced visibility, account penalties if used carelessly, or restriction of legitimate speech."
    )
    st.info(FALSE_POSITIVE_CONSEQUENCE)
    error_path = metrics / "error_counts_by_label.csv"
    if error_path.is_file():
        fewer = fewer_tfidf_false_positives(read_table(error_path))
        if not fewer.empty:
            st.write(F1_NOT_FEWEST_FALSE_POSITIVES)
            for row in fewer.itertuples():
                st.write(
                    f"{row.label}: TF-IDF false positives {int(row.tfidf_false_positives)}, "
                    f"QARiB false positives {int(row.qarib_false_positives)}"
                )

    st.subheader("False-positive comparison by model")
    summary_path = ethics_dir / "false_positive_summary.csv"
    chart = ROOT / "outputs" / "figures" / "false_positive_rate_by_label.png"
    if summary_path.is_file():
        if chart.exists():
            st.image(str(chart))
        rates = read_table(summary_path)
        shown = rates.copy()
        shown["false_positive_rate"] = shown["false_positive_rate"].map(lambda value: f"{value:.4f}")
        with st.expander("False-positive rates and raw counts"):
            st.dataframe(shown, hide_index=True, width="stretch")
        st.caption("False-positive rate is false positives divided by all comments that are truly negative for that label.")
    else:
        st.info("Run `uv run python scripts/analyze_false_positive_risk.py` to create the false-positive table.")

    st.subheader("Threshold trade-off")
    tradeoff_path = ethics_dir / "qarib_threshold_tradeoff.csv"
    precision_chart = ROOT / "outputs" / "figures" / "qarib_threshold_precision_recall.png"
    if tradeoff_path.is_file():
        tradeoff = read_table(tradeoff_path)
        if precision_chart.exists():
            st.image(str(precision_chart))
        fp_chart = ROOT / "outputs" / "figures" / "qarib_threshold_false_positives.png"
        if fp_chart.exists():
            st.image(str(fp_chart))
        points = operating_points(tradeoff)
        point_cards = st.columns(4)
        point_cards[0].metric("Highest Macro-F1", f"{points['highest_macro_f1']['threshold']:.2f}")
        point_cards[1].metric("Highest Macro Precision", f"{points['highest_macro_precision']['threshold']:.2f}")
        point_cards[2].metric("Fewest false positives", f"{points['lowest_false_positives']['threshold']:.2f}")
        point_cards[3].metric("Current threshold", f"{points['current']['threshold']:.2f}")
        st.write(POLICY_CHOICES)
        st.caption(THRESHOLD_POLICY)
        chosen_threshold = st.slider(
            "QARiB decision threshold", min_value=0.30, max_value=0.80, value=0.50, step=0.05, key="ethics_threshold"
        )
        selected = nearest_threshold_row(tradeoff, float(chosen_threshold))
        st.caption(EXPLORER_NOTE)
        cards = st.columns(5)
        cards[0].metric("Macro precision", f"{selected.macro_precision:.3f}")
        cards[1].metric("Macro recall", f"{selected.macro_recall:.3f}")
        cards[2].metric("Macro-F1", f"{selected.macro_f1:.3f}")
        cards[3].metric("False positives", f"{int(selected.total_false_positives):,}")
        cards[4].metric("False negatives", f"{int(selected.total_false_negatives):,}")
    else:
        st.info("Run `uv run python scripts/analyze_threshold_tradeoff.py` to create the threshold table.")

    st.subheader("Manual False-Positive Review")
    st.warning(REVIEWER_WARNING)
    review_path = ethics_dir / "manual_false_positive_review.csv"
    if review_path.is_file():
        preview = read_table(review_path)
        reviewed = review_summary(preview.fillna(""))
        st.write(f"Review sample: {len(preview):,} false-positive examples. Reviewed so far: {int((preview.get('reviewed', pd.Series(dtype=str)).fillna('').astype(str) == 'yes').sum()):,}.")
        st.caption("Record interpretations on the Error Analysis tab. Saving there does not change model predictions.")
        if not reviewed.empty:
            st.table(reviewed)
    else:
        st.info("Run `uv run python scripts/build_false_positive_review_sample.py` to create the review sample.")

    st.subheader("Exploratory Linguistic Robustness")
    st.markdown(
        "Toxicity classification may behave differently across dialects, slang, code-switching, sarcasm, "
        "culturally specific expressions, and reclaimed or context-dependent insults."
    )
    st.warning(DIALECT_CAVEAT)
    st.write(LINGUISTIC_ROBUSTNESS)
    st.write(DIALECT_NOT_IDENTITY)
    region_path = ethics_dir / "false_positive_by_predicted_region.csv"
    if region_path.is_file():
        st.caption(EXPLORATORY_REGION_LABEL)
        st.table(read_table(region_path))

    st.subheader("Annotation bias")
    st.write(ANNOTATION_BIAS)
    st.caption("This is a general limitation of supervised toxicity classification, not a finding about the AraTox annotators.")

    st.subheader("Potential misuse")
    st.markdown(
        "This demo should not automatically be used for censorship, surveillance, law-enforcement action, "
        "account suspension, restricting political or legitimate speech, or profiling people based on language style or predicted dialect."
    )
    st.write(DIALECT_NOT_IDENTITY)

    st.subheader("Human review requirement")
    st.error(HUMAN_REVIEW)
    st.write(RESEARCH_CONCLUSION)
    st.subheader("Model Selection Conclusion")
    st.markdown(SELECTION_CONCLUSION)
    with st.expander("Category names stored for manual review"):
        st.caption("These stay blank until a person fills them in: " + ", ".join(REVIEW_FIELDS) + ".")


def error_analysis_view():
    path = ROOT / "outputs" / "ethics" / "manual_false_positive_review.csv"
    st.subheader("Manual False-Positive Review")
    st.caption("Each card is a comment the model flagged and the annotation did not. Say why you think that happened.")
    st.warning(REVIEWER_WARNING)
    if not path.is_file():
        st.info("The review sample has not been built yet.")
        return
    full_sample = pd.read_csv(path, dtype=str).fillna("")
    full_count = len(full_sample)
    sample = without_sexual_examples(full_sample, hide_sexual_examples)
    count = len(sample)
    if hide_sexual_examples and full_count > count:
        st.caption(f"Hiding {full_count - count:,} sexual examples. {count:,} of {full_count:,} remain. The Sexual label is still in the score tables.")
    if count == 0:
        st.info("The review sample is empty.")
        return
    reviewed_count = int(sample.reviewed.astype(str).str.lower().eq("yes").sum())
    st.progress(reviewed_count / count)
    st.caption(f"{reviewed_count} of {count} examples reviewed")
    index = int(st.session_state.get("review_index", 0)) % count
    previous, position, following = st.columns([1, 2, 1])
    step = 0
    if previous.button("Previous", key="review_previous", width="stretch"):
        step = -1
    if following.button("Next", key="review_next", width="stretch"):
        step = 1
    if step:
        index = step_example_index(index, count, step)
    st.session_state.review_index = index
    position.markdown(f"<div style='text-align:center'><b>Example {index + 1} of {count}</b></div>", unsafe_allow_html=True)
    row = sample.iloc[index]
    arabic_block(row.text_raw)
    truth, tfidf_col, qarib_col = st.columns(3)
    truth.metric("Annotated labels", pretty_labels(row.true_labels))
    tfidf_col.metric("TF-IDF flagged", pretty_labels(row.tfidf_predicted_labels))
    qarib_col.metric("QARiB flagged", pretty_labels(row.qarib_predicted_labels))
    st.info(f"{row.trigger_model} flagged {row.trigger_label}, and the annotation does not include it.")
    with st.expander("Predicted dialect (exploratory)"):
        st.write(f"Dialect: {row.predicted_dialect or 'unavailable'}")
        st.write(f"Region: {row.predicted_region or 'unavailable'}")
        st.caption(DIALECT_CAVEAT)
    saved_categories = [item for item in str(row.review_category).split("|") if item in REVIEW_CATEGORIES]
    categories = st.multiselect(
        "Why do you think this was flagged?",
        list(CATEGORY_LABELS.values()),
        default=[CATEGORY_LABELS[item] for item in saved_categories],
        key=f"review_categories_{row.row_id}",
        help="Your interpretation for error analysis. It is not a model explanation and it is not a new ground-truth label.",
    )
    notes = st.text_area(
        "Notes",
        value=row.review_notes,
        key=f"review_notes_{row.row_id}",
        placeholder="Optional. What in the sentence makes this a false alarm?",
    )
    if st.button("Save this review", key="save_review", type="primary"):
        updated = apply_review(full_sample, row.row_id, [LABEL_TO_CATEGORY[item] for item in categories], notes)
        updated.to_csv(path, index=False)
        summary = review_summary(updated)
        summary_path = path.with_name("manual_false_positive_review_summary.csv")
        if summary.empty:
            if summary_path.exists():
                summary_path.unlink()
        else:
            summary.to_csv(summary_path, index=False)
        st.success("Review saved. Model predictions were not changed.")
    summary_path = path.with_name("manual_false_positive_review_summary.csv")
    if summary_path.is_file():
        st.table(read_table(summary_path))
    st.download_button("Download review CSV", path.read_bytes(), file_name=path.name, mime="text/csv")


def linguistic_view():
    st.subheader("Exploratory Linguistic Robustness")
    st.caption("This page groups mistakes by a predicted region. It does not measure fairness for real dialect groups.")
    st.warning(DIALECT_CAVEAT)
    st.write(LINGUISTIC_ROBUSTNESS)
    st.write(DIALECT_NOT_IDENTITY)
    st.caption(EXPLORATORY_REGION_LABEL)
    region_path = ROOT / "outputs" / "ethics" / "false_positive_by_predicted_region.csv"
    if region_path.is_file():
        st.table(read_table(region_path))
    else:
        st.info("Run `uv run python scripts/build_false_positive_review_sample.py` to summarize predicted regions.")
    st.caption("Predicted dialect does not change the toxicity score, the threshold, or the model.")


analyze_tab, comparison_tab, final_tab, ethics_tab, error_tab, dialect_tab = st.tabs(
    [
        ":material/search: Analyze",
        ":material/bar_chart: Model Comparison",
        ":material/emoji_events: Final Results",
        ":material/verified_user: Responsible AI",
        ":material/rate_review: Error Analysis",
        ":material/public: Dialect Analysis",
    ]
)
with analyze_tab:
    input_col, result_col = st.columns(2, gap="large")
with input_col:
    st.subheader("Check a sentence")
    choice = st.radio(
        "Model",
        list(inference.SYSTEMS),
        horizontal=True,
        help="TF-IDF is fast and simple. QARiB reads words in context. The ensemble averages their calibrated scores and was the most accurate on the held-out test set.",
    )
    policy = st.radio(
        "Cutoff",
        list(inference.POLICIES),
        horizontal=True,
        help="Reported cutoff: 0.50 (TF-IDF: decision score 0). Tuned per label: calibrated scores with a cutoff per label chosen on validation.",
    )
    text = st.text_area("Arabic text", height=180, placeholder="الصق الجملة هنا")
    if st.button("Analyze", type="primary", width="stretch"):
        if not text.strip():
            st.session_state.result = None
            st.warning("Enter Arabic text first.")
        else:
            norm = normalize_arabic(text, prep)
            review_id = f"{choice}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
            try:
                scores, pred, note = run_system([norm], choice, policy)
            except FileNotFoundError as exc:
                st.session_state.result = None
                st.error(str(exc))
            else:
                st.session_state.result = {
                    "review_id": review_id,
                    "model": choice,
                    "policy": policy,
                    "text_raw": text,
                    "text_norm": norm,
                    "binary": "Toxic" if pred[0].sum() else "Non-toxic",
                    "pred": pred[0].tolist(),
                    "scores": scores[0].tolist(),
                    "thresholds": inference.threshold_vector(choice, policy, load_calibration()).tolist(),
                    "score_note": note,
                    "dialect": dialect_for(text),
                }
with result_col:
    if st.session_state.get("result"):
        show_result(st.session_state.result)
    else:
        st.info("Paste an Arabic sentence and press **Analyze**. The decision, the score for each label, the words that drove it and a dialect guess appear here.")
with analyze_tab:
    st.divider()
    batch_view()

with comparison_tab:
    comparison_tab_view()

with final_tab:
    final_results_view()

with ethics_tab:
    responsible_ai_view()

with error_tab:
    error_analysis_view()

with dialect_tab:
    linguistic_view()
    st.divider()
    dialect_check_view()
