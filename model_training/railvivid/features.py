"""Optional statistical feature preparation reusing the historical extractor."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.features.extractor import compute_window_features
from .common import fingerprint,save_array,write_csv,write_json,file_hash,finite
from .prepare import load_prepared,CHANNELS
from .errors import ArtifactError

def extract_statistics(cfg):
    root,meta=load_prepared(cfg);target=root/'statistics'
    if target.exists():raise ArtifactError(f'Statistical feature destination already exists: {target}')
    target.mkdir();windows=pd.read_csv(root/'windows.csv');rows=[];columns=None
    for run,g in windows.groupby('run_id',sort=False):
        raw=np.load(root/'runs'/run/'acceleration.npy',mmap_mode='r')
        for row in g.itertuples():
            values={}
            for i,ch in enumerate(CHANNELS):
                f=compute_window_features(np.asarray(raw[row.start_idx:row.end_idx,i],dtype='float64'),fs=cfg['data']['sampling_hz'])
                values.update({ch+'_'+k:v for k,v in f.items()})
            columns=list(values);rows.append([row.window_id,*values.values()])
    table=pd.DataFrame(rows,columns=['window_id',*columns]);finite(table[columns].to_numpy(),'statistical features')
    write_csv(target/'features.csv',table)
    result={'data_fingerprint':meta['data_fingerprint'],'columns':columns,'rows':len(table),
        'extractor_sha256':file_hash(Path(__file__).parents[1]/'src/features/extractor.py'),'features_sha256':file_hash(target/'features.csv')}
    write_json(target/'READY.json',result);return result
