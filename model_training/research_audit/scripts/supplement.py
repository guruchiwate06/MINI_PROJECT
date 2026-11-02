"""Provenance, DR-Train samples, and alternative fold feasibility."""
from pathlib import Path
import sys,json,hashlib,subprocess,importlib.util
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.neighbors import NearestNeighbors
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'research_audit/evidence'
REPO = ROOT.parent if ROOT.name == 'model_training' else ROOT / 'mini_project_rajguru'
sys.path.insert(0,str(REPO))
from src.evaluation.splitters import SpatialHeldOutSplitter,CombinedRunAndSpatialSplitter

def main():
    dr=Path('D:/DR_Train'); records=[]; samples=[]
    for v in ['LRV4306','LRV4313']:
        for kind in ['accelerometer_data','gps_data']:
            files=sorted((dr/v/kind).rglob('*.mat'))
            records.append({'vehicle':v,'kind':kind,'files':len(files),'bytes':sum(f.stat().st_size for f in files)})
            if files:
                f=files[0]
                try:
                    d=loadmat(f,squeeze_me=True); a=np.asarray(d['save_var'])
                    samples.append({'file':str(f),'shape':a.shape,'dtype':str(a.dtype),'finite':bool(np.isfinite(a).all()),'min':float(a.min()),'max':float(a.max())})
                except Exception as e: samples.append({'file':str(f),'error':str(e)})
    pd.DataFrame(records).to_csv(OUT/'dr_train_actual_inventory.csv',index=False)
    (OUT/'dr_train_sample_records.json').write_text(json.dumps(samples,indent=2,default=str))
    # Independently check coordinate units on actual route layout files.
    route=[]
    for f in (dr/'ground_truth_rail_layout_GPS').glob('*.csv'):
        a=pd.read_csv(f,header=None).to_numpy(dtype=float)
        if a.shape[1]<3: continue
        ds=np.diff(a[:,0]); lat=np.deg2rad(a[:,1]); lon=np.deg2rad(a[:,2])
        hav=np.sin(np.diff(lat)/2)**2+np.cos(lat[:-1])*np.cos(lat[1:])*np.sin(np.diff(lon)/2)**2
        dist=6371000*2*np.arctan2(np.sqrt(hav),np.sqrt(1-hav)); m=ds>0
        route.append({'file':f.name,'rows':len(a),'coordinate_span_raw':float(np.ptp(a[:,0])),
            'median_geodesic_per_raw_step':float(np.median(dist[m]/ds[m]))})
    pd.DataFrame(route).to_csv(OUT/'dr_train_route_unit_check.csv',index=False)
    d=pd.read_csv('D:/Rail-VIVID/data/cached_window_features.csv'); folds=[]; manifests={}
    for k,(test,val) in enumerate([([1,2,3],[4]),([4,5,6],[7]),([7,8,9],[1])],1):
        for kind,sp in [('spatial',SpatialHeldOutSplitter(test_anomaly_ids=test,val_anomaly_ids=val)),
            ('combined',CombinedRunAndSpatialSplitter(test_anomaly_ids=test,val_anomaly_ids=val))]:
            parts=sp.split(d)
            manifests[f'{kind}_fold{k}']={p:x.index.tolist() for p,x in zip(['train','validation','test'],parts)}
            for part,x in zip(['train','validation','test'],parts):
                folds.append({'fold':k,'protocol':kind,'partition':part,'windows':len(x),'positive':int((x.label==1).sum()),
                    'negative':int((x.label==0).sum()),'runs':x.run_id.nunique(),'sites':','.join(map(str,sorted(x.loc[x.label==1,'nearest_anomaly_id'].unique())))})
    pd.DataFrame(folds).to_csv(OUT/'candidate_fold_counts.csv',index=False)
    (OUT/'candidate_fold_indices.json').write_text(json.dumps(manifests))
    env={'available_modules':{m:bool(importlib.util.find_spec(m)) for m in ['transformers','timm','torchvision','h5py','psutil']}}
    try:
        import torchvision
        env['torchvision']=torchvision.__version__
    except Exception as e: env['torchvision_error']=str(e)
    try:
        import psutil
        env['ram_total_bytes']=psutil.virtual_memory().total; env['ram_available_bytes']=psutil.virtual_memory().available
    except ImportError: pass
    env['local_checkpoints']=[str(p) for folder in [REPO/'outputs',Path('D:/Rail-VIVID/outputs'),Path.home()/'.cache/torch/hub/checkpoints',Path.home()/'.cache/huggingface/hub'] if folder.exists() for p in folder.rglob('*') if p.is_file() and p.suffix in ['.pt','.pth','.ckpt','.safetensors','.bin']]
    (OUT/'supplement_environment.json').write_text(json.dumps(env,indent=2))
    # Metadata only. Never download dataset payloads.
    try:
        import requests
        r=requests.get('https://huggingface.co/api/datasets/saluslab/Rail-VIVID',timeout=45); r.raise_for_status(); meta=r.json()
        (OUT/'upstream_metadata.json').write_text(json.dumps(meta,indent=2))
        files=[x['rfilename'] for x in meta.get('siblings',[])]
        pd.DataFrame({'upstream_path':files,'exists_locally':[(Path('D:/Rail-VIVID')/p).is_file() for p in files]}).to_csv(OUT/'upstream_presence.csv',index=False)
        print('Upstream revision',meta.get('sha'),'files',len(files),'local',sum((Path('D:/Rail-VIVID')/p).is_file() for p in files),flush=True)
    except Exception as e: (OUT/'upstream_metadata_error.txt').write_text(str(e))
    print('supplement done',flush=True)

if __name__=='__main__': main()
