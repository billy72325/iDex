"""Label file access shared by prepare.py (no torch) and model.py."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABELS = HERE / 'pokemon_labels_national.json'


def load_labels(path=LABELS):
    labels = json.loads(Path(path).read_text())
    ids = [x['id'] for x in labels]
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        raise ValueError('labels must be unique and sorted by National Dex id')
    return labels
