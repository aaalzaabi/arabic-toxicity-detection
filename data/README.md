# Data

Run `uv run python scripts/prepare_data.py` to load `Alshargi/Aratox` through Hugging Face and create local processed splits.

The project never invents dialect labels. If the distributed table does not expose official dialect metadata, leave `dialect_column: null` in `configs/data.yaml` and skip dialect-stratified metrics until official metadata is available.
