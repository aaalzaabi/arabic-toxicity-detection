set shell := ["bash", "-uc"]

sync:
    uv sync --extra dev
prepare:
    uv run python scripts/prepare_data.py
svm:
    uv run python scripts/train_svm.py
qarib:
    uv run python scripts/train_qarib.py
qarib-predict:
    uv run python scripts/predict_qarib.py --split validation
thresholds:
    uv run python scripts/tune_thresholds.py --predictions outputs/predictions/qarib_validation.csv
jais:
    uv run python scripts/run_jais.py --backend transformers --split validation --max-items 100
evaluate:
    uv run python scripts/evaluate_models.py --split validation
errors:
    uv run python scripts/export_errors.py --predictions outputs/predictions/qarib_validation.csv
dialect:
    uv run python scripts/run_dialect_inference.py --input data/processed/validation.csv --text-column text_raw
dialect-analyze:
    uv run python scripts/analyze_by_predicted_dialect.py

compare:
    uv run python scripts/compare_models.py

ethics:
    uv run python scripts/analyze_false_positive_risk.py
    uv run python scripts/analyze_threshold_tradeoff.py
    uv run python scripts/build_false_positive_review_sample.py
app:
    PYTHONPATH=src uv run streamlit run src/aratox_nlp/app.py
test:
    uv run pytest -q

calibrate:
    uv run python scripts/fit_calibration.py
robustness:
    uv run python scripts/run_robustness.py
quantize:
    uv run python scripts/quantize_qarib.py
dialect-check:
    uv run python scripts/build_dialect_check_sample.py
freeze:
    uv run python scripts/freeze_decisions.py
final-dry-run:
    uv run python scripts/final_test_evaluation.py --dry-run-on-validation
final-test:
    uv run python scripts/final_test_evaluation.py
jais-validation:
    uv run python scripts/run_jais.py --backend transformers --split validation --max-items 500
    uv run python scripts/run_jais.py --backend transformers --split validation --max-items 500 --few-shot
    uv run python scripts/evaluate_models.py --split validation
