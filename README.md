# Arabic Toxicity Detection Across Dialects

**Comparing Classical NLP, QARiB, and Jais-2**

Complete Applied NLP project implementing the proposal:

- **Dataset:** AraTox (`Alshargi/Aratox`)
- **Binary task:** toxic vs non-toxic
- **Multi-label task:** Appearance, Cussing, Hatred, Racial, Sexual, Violence
- **Tier 1:** TF-IDF + One-vs-Rest Linear SVM
- **Tier 2:** QARiB (`ahmedabdelali/bert-base-qarib`)
- **Tier 3:** Jais-2-8B-Chat (`inception42/Jais-2-8B-Chat`)
- **Demo:** Streamlit

This project does not only ask which Arabic toxicity classifier achieves the highest F1. It asks which model and operating policy are most appropriate when predictive quality, inference efficiency, false-positive risk, language variation, and human-review requirements are considered together.

## Quick start

```bash
uv sync --extra dev
uv run python scripts/prepare_data.py
uv run python scripts/train_svm.py
uv run python scripts/train_qarib.py
uv run python scripts/predict_qarib.py --split validation
uv run python scripts/tune_thresholds.py --predictions outputs/predictions/qarib_validation.csv
uv run python scripts/run_jais.py --backend transformers --split validation --max-items 100
uv run python scripts/evaluate_models.py --split validation
uv run python scripts/export_errors.py --predictions outputs/predictions/qarib_validation.csv
uv run streamlit run src/aratox_nlp/app.py
```

Or use `just prepare`, `just svm`, `just qarib`, etc.

## Scientific protocol

- The public AraTox **test** split is held out.
- A validation split is created reproducibly from the public training split.
- Thresholds and model choices are made on validation only.
- The test split should be scored **once**, after decisions are frozen.
- Binary toxicity is derived as `1` when any of the six toxicity labels is positive.
- `NOT` is retained as a consistency check rather than used as a seventh toxicity target.
- Missing dialect ground truth is never inferred or fabricated.

## Data preparation

`prepare_data.py` validates AraTox label columns, normalizes Arabic text with configurable rules, hashes normalized text, reports within-split duplicates and cross-split overlap, derives binary toxicity, and saves:

- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`
- `data/processed/data_report.json`

If the downloadable AraTox table you use includes an official dialect column, set its exact name in `configs/data.yaml`. If it does not, leave the value as `null`; dialect-level evaluation will stop rather than invent labels.

## Classical model

The classical model uses:

- word TF-IDF: 1–3 grams
- character TF-IDF: 2–5 grams
- One-vs-Rest Linear SVM for the six toxicity labels
- separate binary Linear SVM for toxic/non-toxic

Run:

```bash
uv run python scripts/train_svm.py
```

## QARiB

QARiB is fine-tuned with six output logits and `BCEWithLogitsLoss` through Transformers' multi-label sequence-classification model.

```bash
uv run python scripts/train_qarib.py
uv run python scripts/predict_qarib.py --split validation
```

The training script supports CUDA, Apple MPS, or CPU.

## Jais-2

Zero-shot:

```bash
uv run python scripts/run_jais.py --backend transformers --split validation --max-items 100
```

Few-shot:

```bash
uv run python scripts/run_jais.py --backend transformers --split validation --max-items 100 --few-shot
```

For a vLLM/OpenAI-compatible endpoint:

```bash
uv sync --extra jais-api
uv run python scripts/run_jais.py \
  --backend openai \
  --base-url http://localhost:8000/v1 \
  --model inception42/Jais-2-8B-Chat \
  --split validation \
  --max-items 100
```

Invalid Jais JSON is logged; it is not silently corrected.

## TF-IDF vs QARiB Comparison

### TF-IDF + Linear SVM

- classical NLP
- sparse word and character features
- fast and lightweight
- does not directly model deep context

### QARiB

- pretrained Arabic transformer
- contextual representations
- larger and slower
- can capture contextual language patterns

Research question: does a contextual Arabic transformer improve multi-label toxicity detection enough to justify its additional computational cost over a classical TF-IDF + Linear SVM baseline?

The comparison uses the existing validation predictions. It does not retrain either model, change thresholds, or score the held-out test set. Exact-match disagreement means all six toxicity labels match. Binary McNemar uses the toxic/non-toxic decision only.

```bash
uv run python scripts/compare_models.py
```

## Evaluation

The project reports:

- Macro-F1
- Micro-F1
- per-label precision / recall / F1
- Hamming Loss
- Subset Accuracy
- binary accuracy / Macro-F1 / toxic-class F1
- inference latency for Jais
- error exports for manual analysis

```bash
uv run python scripts/evaluate_models.py --split validation
```

## Dialect Identification

AraTox is multi-dialect in composition: the paper describes comments from Gulf, Levantine, Nile Basin, North African, Yemeni, and Modern Standard Arabic. The public rows are not individually tagged with dialect. They contain the text, six toxicity labels, `NOT`, and word-count fields.

Dialect in this project is predicted by a separate classifier, `CAMeL-Lab/bert-base-arabic-camelbert-mix-did-nadi` (CAMeLBERT-Mix fine-tuned on NADI country-level labels; Apache-2.0). Predicted dialect is not ground truth, and it does not identify nationality. It is a guess about linguistic style. Short text and code-switching are difficult. The NADI model has no MSA class, so formal Arabic can still be assigned a country label.

The toxicity models and the dialect model stay independent. Dialect predictions are not inputs to QARiB, Jais, or the SVM, and they are not used to retrain or select a toxicity model.

Exploratory question: does toxicity detection performance vary across dialect categories predicted by an independent Arabic dialect-identification model? Tables that answer this are toxicity performance grouped by predicted dialect, not performance by true dialect.

```bash
uv run python scripts/run_dialect_inference.py \
  --input data/processed/validation.csv \
  --text-column text_raw
uv run python scripts/analyze_by_predicted_dialect.py
```

`scripts/dialect_analysis.py` still runs only when an official dialect column is configured. It does not invent AraTox dialect labels, and topic or location words are not dialect labels.

Country labels, in the model's config order: Algerian, Bahraini, Djiboutian, Egyptian, Iraqi, Jordanian, Kuwaiti, Lebanese, Libyan, Mauritanian, Moroccan, Omani, Palestinian, Qatari, Saudi, Somali, Sudanese, Syrian, Tunisian, Emirati, Yemeni.

Regional families: Gulf (Emirati, Saudi, Kuwaiti, Qatari, Bahraini, Omani), Egyptian, Levantine (Lebanese, Syrian, Jordanian, Palestinian), Iraqi, Maghrebi (Moroccan, Algerian, Tunisian, Libyan), and Other (Yemeni, Sudanese, Djiboutian, Mauritanian, Somali). Yemen, Sudan, Djibouti, Mauritania, and Somalia are not folded into Gulf, Egyptian, or Maghrebi.

## Responsible AI and Ethical Considerations

This project does not only ask which Arabic toxicity classifier achieves the highest F1. It asks which model and operating policy are most appropriate when predictive quality, inference efficiency, false-positive risk, language variation, and human-review requirements are considered together.

Toxicity classification can affect people when a mistake is acted on. A false positive labels a non-toxic comment as toxic. In a real moderation system, that could flag legitimate content, add unnecessary review, reduce visibility, or restrict speech if no person checks the decision. A false negative misses toxic content. This demo does not apply either outcome to accounts.

AraTox labels come from human annotation. Toxicity is partly context-dependent. Annotators may differ on insults, slang, sarcasm, cultural references, or acceptable speech, and a supervised model can reproduce those inconsistencies. That is a general limitation of this kind of task.

AraTox does not label dialect on each row. Dialect, slang, code-switching, sarcasm, and reclaimed expressions can change how a comment is read. The separate CAMeLBERT/NADI classifier can support an exploratory analysis by predicted dialect. Those categories are not AraTox ground truth, and they are not used to change a toxicity score, threshold, or model. Predicted dialect describes linguistic style, not nationality, identity, or origin.

The system should not be used for censorship, surveillance, law-enforcement action, account suspension, restricting legitimate speech, or profiling people by language style.

QARiB’s saved validation scores can be re-thresholded to see how precision, recall, and false positives move together. Raising or lowering the threshold is a policy choice about the cost of each error. The reported comparison still uses the existing 0.5 threshold. This phase does not replace it.

Research questions:

- Does a contextual Arabic transformer improve multi-label toxicity detection enough to justify its additional computational cost over a classical TF-IDF + Linear SVM baseline?
- Does the model with the highest predictive performance also provide acceptable error behavior for a human-reviewed moderation setting?
- How do different decision thresholds change the trade-off between false positives and missed toxic content?

Three QARiB analysis policies are read from the saved threshold table: reported / balanced at 0.50, an exploratory F1-focused point at 0.65, and a precision-focused review point at 0.80. They do not change the trained model or replace the reported 0.50 result. Jais-2-8B-Chat is integrated, and a full validation evaluation is still pending, so it is not in the main scorecard.

Predicted dialect supports an exploratory false-positive distribution by predicted region. That table is not a dialect fairness audit.

```bash
uv run python scripts/build_false_positive_review_sample.py
```

On this validation set, QARiB has higher F1 than TF-IDF + SVM, while false-positive counts still favor TF-IDF on some labels. Neither result makes one model ethically superior. Model selection for moderation should consider precision, recall, false-positive risk, computational cost, and the intended consequences of classification. Human review is required for consequential decisions.

```bash
uv run python scripts/analyze_false_positive_risk.py
uv run python scripts/analyze_threshold_tradeoff.py
```

## Streamlit

```bash
PYTHONPATH=src uv run streamlit run src/aratox_nlp/app.py
```

The UI is a research/decision-support demo, not an autonomous moderation authority. Tabs are Analyze, Model Comparison, Responsible AI, Error Analysis, and Dialect Analysis. Responsible AI presents the scorecard, operating policies, false-positive risk, threshold trade-off, exploratory linguistic robustness, and the model-selection conclusion. Error Analysis is where a person can annotate the false-positive sample. Predicted dialect stays exploratory.

## Final evaluation and enhancements

All choices below were made on validation only. The held-out test split was then scored **once**.

```bash
just calibrate      # Platt calibration, per-label cutoffs, ensemble, bootstrap CIs (validation)
just robustness     # perturbation stress test (validation sample, n=1,000)
just quantize       # int8 QARiB vs fp32 (validation)
just dialect-check  # blank 102-comment sheet for a human dialect spot-check
just freeze         # fingerprints of models, calibration and configs
just final-dry-run  # whole test pipeline on validation, never touches test
just final-test     # scores test once; refuses to run twice or if a frozen file changed
```

**Held-out test (n=3,670), Macro-F1 with 95% paired-bootstrap CIs:**

| System | Macro-F1 | False positives |
|---|---|---:|
| **Ensemble (calibrated TF-IDF + QARiB, tuned cutoffs) — primary, chosen on validation** | **0.921 [0.913, 0.928]** | **234** |
| QARiB, tuned cutoffs | 0.915 [0.908, 0.924] | 267 |
| QARiB, reported 0.50 | 0.914 [0.906, 0.922] | 315 |
| QARiB int8 (CPU), 0.50 | 0.908 [0.900, 0.916] | 264 |
| TF-IDF + Linear SVM, reported cutoff | 0.904 [0.895, 0.912] | 294 |
| TF-IDF + Linear SVM, tuned cutoffs | 0.899 [0.890, 0.907] | 327 |

- QARiB vs TF-IDF: +0.010 [+0.003, +0.018] on test (+0.013 [+0.006, +0.020] on validation). The gain is small but not noise.
- Ensemble vs QARiB: +0.007 [+0.002, +0.012], with the fewest false positives.
- Per-label cutoffs did not help TF-IDF on test.
- Calibration error: TF-IDF 0.209 → 0.009, QARiB 0.016 → 0.010 (test).
- Robustness: Arabizi breaks every model (Macro-F1 −0.88 to −0.91); elongated letters −0.08 to −0.10; typos −0.06 to −0.07; diacritics and tatweel have no effect because preprocessing removes them.
- int8 QARiB: half the size (284 MB vs 541 MB), −0.007 Macro-F1 on validation, about 2× faster for one sentence on CPU.

The Streamlit app adds a **Final Results** tab, an ensemble option, a calibrated "Tuned per label" cutoff, word-occlusion highlighting ("Why this decision?"), CSV batch scoring, and a blind dialect spot-check on the Dialect Analysis tab. Word highlights show sensitivity, not the model's reasoning.

**Jais-2** is gated on Hugging Face. Log in (`hf auth login`), accept the model terms on its page, then run `just jais-validation`. The loader now uses bfloat16 on Apple GPUs so the 8B model fits in memory. Jais is not part of the frozen test run; choose zero- or few-shot on validation, then score test once for the chosen mode.

## Data and acknowledgements

- **AraTox** (`Alshargi/Aratox` on Hugging Face), licensed CC-BY-4.0. The dataset is downloaded by `scripts/prepare_data.py` and is not redistributed in this repository; nor are prediction files that contain comment text.
- **QARiB** (`ahmedabdelali/bert-base-qarib`), **Jais-2-8B-Chat** (`inception42/Jais-2-8B-Chat`, gated) and **CAMeLBERT-Mix NADI** (`CAMeL-Lab/bert-base-arabic-camelbert-mix-did-nadi`) are used under their own licenses. Trained model weights are not included; retrain with the commands above.

## Tests

```bash
uv run pytest -q
```

## Recommended final workflow

1. Prepare data and inspect `data_report.json`.
2. Train SVM.
3. Train QARiB.
4. Tune QARiB thresholds on validation.
5. Run Jais zero-shot/few-shot.
6. Compare validation performance and latency.
7. Freeze all decisions.
8. Run final test predictions once.
9. Export final tables, dialect analysis if metadata exists, and errors.
10. Present results in Streamlit and the final report.
