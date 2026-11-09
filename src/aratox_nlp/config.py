from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]

def load_yaml(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with p.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def ensure_output_dirs():
    for rel in ['data/processed','outputs/models','outputs/metrics','outputs/predictions','outputs/figures','outputs/errors','outputs/reviews','outputs/ethics']:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)
