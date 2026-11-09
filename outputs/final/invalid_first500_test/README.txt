Invalid Jais-2 test run (2026-11-09, 14:18-14:44).

The pipeline scored the FIRST 500 rows of the official AraTox test file. That file is
ordered by label, so those rows were 99.8% toxic with no Hatred, Racial or Violence
examples. Macro-F1 on that slice is meaningless for every system (QARiB 0.38, TF-IDF 0.33).

No decision was changed because of it: the Jais prompt mode (zero-shot) was chosen on
validation before this run and stays frozen. Jais was re-scored on the full test split.
