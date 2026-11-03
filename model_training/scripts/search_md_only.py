import glob

for f in glob.glob('**/*.md', recursive=True):
    if '.venv' in f or 'node_modules' in f: continue
    text = open(f, encoding='utf-8', errors='ignore').read()
    for phrase in ['24.1', '63.6', '0.3500', '40.9', '94.7', '0.5714', '0.6845']:
        if phrase in text:
            print(f"Matched '{phrase}' in {f}")

