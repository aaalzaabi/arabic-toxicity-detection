# Execution plan

1. **Data:** prepare AraTox, validate labels, create validation from public train, keep public test held out, inspect duplicate overlap.
2. **Classical baseline:** word/character TF-IDF + One-vs-Rest Linear SVM for six toxicity labels; separate binary SVM.
3. **QARiB:** fine-tune six-output classifier with BCEWithLogitsLoss; select checkpoint using validation Macro-F1.
4. **Thresholds:** tune only on validation.
5. **Jais-2:** compare zero-shot and few-shot structured prompting; measure parse-failure rate and latency.
6. **Evaluation:** Macro/Micro F1, per-label F1, Hamming Loss, Subset Accuracy, binary Macro-F1/toxic-F1, latency, manual error review.
7. **Dialect analysis:** only when official dialect metadata is present; always report group sample sizes.
8. **Final test:** freeze preprocessing, checkpoint, thresholds, prompts, and few-shot examples, then evaluate the held-out test once.
9. **Demo:** Streamlit decision-support UI.
