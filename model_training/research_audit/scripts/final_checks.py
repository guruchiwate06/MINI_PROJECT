from pathlib import Path
import sys,json,hashlib
import requests
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'research_audit/evidence'; DATA=Path('D:/Rail-VIVID')
REPO = ROOT.parent if ROOT.name == 'model_training' else ROOT / 'mini_project_rajguru'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def main():
    rows=[]
    for f in sorted(DATA.glob('*to*_*_*/*.csv')):
        rows.append({'path':str(f),'bytes':f.stat().st_size,'sha256':sha(f)})
    pd.DataFrame(rows).to_csv(OUT/'raw_csv_sha256.csv',index=False)
    upstream=[]
    for run in ['AtoB_22.5_3','AtoB_22.5_4']:
        folder=next((DATA/run).glob('Frames*'))
        url=f'https://huggingface.co/api/datasets/saluslab/Rail-VIVID/tree/5fbbc49ca795296c7fb3fd6ee55a3cae2bb01410/{run}/{folder.name}'
        try:
            r=requests.get(url,params={'limit':2},timeout=30);r.raise_for_status()
            for item in r.json():
                if item.get('type')!='file':continue
                p=DATA/item['path']; local=sha(p)
                upstream.append({'upstream_path':item['path'],'lfs_oid':item.get('lfs',{}).get('oid'),
                    'local_sha256':local,'local_matches_upstream_lfs':local==item.get('lfs',{}).get('oid')})
        except Exception as e: upstream.append({'run':run,'error':str(e)})
    (OUT/'upstream_duplicate_spotcheck.json').write_text(json.dumps(upstream,indent=2))
    # Cache/report copies must be distinguished from the data folder's Git checkout.
    files=['src/evaluation/splitters.py','run_benchmarks.py','reports/baseline_results.json','configs/baseline_config.json']
    (OUT/'checkout_comparison.json').write_text(json.dumps({f:{
        'byte_equal':sha(REPO/f)==sha(DATA/f),
        'text_equal_normalized_newlines':(REPO/f).read_text(encoding='utf-8')==(DATA/f).read_text(encoding='utf-8')
    } for f in files},indent=2))
    print('final checks done',flush=True)

if __name__=='__main__': main()
