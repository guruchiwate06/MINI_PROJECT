import os

for root, dirs, files in os.walk('model_training'):
    if '.venv' in root:
        continue
    for f in files:
        if f.endswith('.json') or f.endswith('.csv') or f.endswith('.txt') or f.endswith('.py') or f.endswith('.md'):
            p = os.path.join(root, f)
            try:
                with open(p, 'r', encoding='utf-8', errors='ignore') as fh:
                    txt = fh.read()
                    if '386' in txt and '44' in txt and '14' in txt:
                        print(f"MATCH (14, 44, 386) in: {p}")
                    if '0.3500' in txt:
                        print(f"MATCH (0.3500) in: {p}")
            except Exception:
                pass
