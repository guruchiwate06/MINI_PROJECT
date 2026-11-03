import os

for root, dirs, files in os.walk('.'):
    if '.venv' in root or 'node_modules' in root or '.git' in root:
        continue
    for f in files:
        if f.endswith(('.md', '.json')):
            p = os.path.join(root, f)
            try:
                c = open(p, encoding='utf-8', errors='ignore').read()
                if '0.3500' in c or '0.6845' in c or '0.5714' in c:
                    for line in c.splitlines():
                        if any(term in line for term in ['0.3500', '0.6845', '0.5714', 'Site 6', 'Site 9', '24.1', '63.6', '40.9', '94.7']):
                            print(f"{p}: {line.strip()[:160]}")
            except Exception:
                pass

