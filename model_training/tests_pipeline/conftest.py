"""All tests are fitting-free. Optimizer steps and forest fitting are forbidden."""
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from railvivid.config import DEFAULTS
from railvivid.common import file_hash,save_array,write_csv,write_json

@pytest.fixture(autouse=True)
def prohibit_training(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Tests must not train models on this laptop')
    for optimizer in [torch.optim.Adam,torch.optim.AdamW,torch.optim.SGD]:monkeypatch.setattr(optimizer,'step',forbidden)
    from sklearn.ensemble import RandomForestClassifier
    monkeypatch.setattr(RandomForestClassifier,'fit',forbidden)

@pytest.fixture
def cfg(tmp_path):
    c=copy.deepcopy(DEFAULTS);c['data'].update({'root':str(tmp_path/'raw'),'artifacts':str(tmp_path/'prepared'),
        'window_samples':128,'stride_samples':64,'reference_run':'AtoB_20_1'})
    c['split'].update({'test_runs':['AtoB_20_3'],'validation_runs':['AtoB_20_2']})
    c['alignment'].update({'candidate_offsets_m':[-2.,0.,2.],'uncertainty_m':0.,'camera_footprint_m':0.})
    c['model'].update({'width':8,'depth':2,'embedding_dim':16,'vision_dim':12,'dropout':0.})
    c['training'].update({'batch_size':4,'workers':0})
    c['experiment'].update({'name':'fixture','output':str(tmp_path/'experiments')})
    c['evaluation']['bootstrap_repeats']=20
    return c

@pytest.fixture
def prepared(cfg):
    root=Path(cfg['data']['artifacts']);root.mkdir();rows=[]
    rng=np.random.default_rng(17)
    for run in ['AtoB_20_1','AtoB_20_2','AtoB_20_3']:
        folder=root/'runs'/run;folder.mkdir(parents=True)
        save_array(folder/'acceleration.npy',rng.normal(size=(18*128,6)).astype('float32'))
        for site in range(1,10):
            for normal in [False,True]:
                start=((site-1)*2+int(normal))*128;position=site*100+(25 if normal else 0)
                rows.append({'window_id':f'{run}:{start}:{start+128}','run_id':run,'start_idx':start,'end_idx':start+128,
                    'timestamp_ns':start*500000,'chainage_m':float(position),'support_min_m':position-.5,'support_max_m':position+.5,
                    'reference_distance_m':0.,'latitude':40.,'longitude':-77.,'label':0 if normal else 1,'site_id':site,
                    'site_distance_m':25 if normal else 0,'direction':'AtoB','target_speed_kmh':20.})
    write_csv(root/'windows.csv',pd.DataFrame(rows));write_csv(root/'sites.csv',pd.DataFrame({'id':range(1,10),'chainage_m':np.arange(1,10)*100.}))
    write_csv(root/'frames.csv',pd.DataFrame(columns=['frame_id','run_id','relative_path','timestamp_ns','nominal_chainage_m','chainage_m','rejection_reason','reference_distance_m']))
    artifacts={str(p.relative_to(root)).replace('\\','/'):file_hash(p) for p in root.rglob('*') if p.is_file()}
    write_json(root/'READY.json',{'schema_version':1,'preparation_config':cfg['data'],'data_fingerprint':'synthetic-fixture','artifacts':artifacts})
    return root
