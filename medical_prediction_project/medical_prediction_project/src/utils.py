import json
from pathlib import Path

def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    path = Path(path)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
