import json
from pathlib import Path

splits_dir = Path('model_training/artifacts/railvivid_v1/splits')
for d in splits_dir.iterdir():
    if d.is_dir():
        ready = d / 'READY.json'
        if ready.exists():
            data = json.load(open(ready))
            print(f"=== Manifest: {d.name} ===")
            print("Cohort:", data.get('cohort'))
            print("Summary keys:", [list(p.keys()) for p in data.get('summary', [])])
            print("Summary content:", json.dumps(data.get('summary', []), indent=2))
