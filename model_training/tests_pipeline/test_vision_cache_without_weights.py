"""Exercise feature-cache IO with a synthetic encoder stub; no weights/downloads."""
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from railvivid.common import write_csv,write_json,read_json,file_hash
from railvivid.vision import extract_vision,vision_path
from railvivid.dataset import VisionStore
from railvivid.errors import DataIntegrityError

def setup_frames(prepared,cfg):
    rows=[]
    for i in range(3):
        relative=f'AtoB_20_{i+1}/frames/{i}.jpg';p=Path(cfg['data']['root'])/relative;p.parent.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(16,16),color=(50+i*40,20,30)).save(p)
        rows.append({'frame_id':str(i),'run_id':f'AtoB_20_{i+1}','relative_path':relative,'rejection_reason':'',
            'chainage_m':100.,'nominal_chainage_m':100.})
    write_csv(prepared/'frames.csv',pd.DataFrame(rows));meta=read_json(prepared/'READY.json');meta['artifacts']['frames.csv']=file_hash(prepared/'frames.csv');write_json(prepared/'READY.json',meta)
    cfg['vision']['revision']='f'*40

def fake_encoder(monkeypatch):
    transformers=pytest.importorskip('transformers')
    class Batch(dict):
        def to(self,device):return Batch({k:v.to(device) for k,v in self.items()})
    class Processor:
        def __call__(self,images,return_tensors):return Batch(pixel_values=torch.zeros(len(images),3,16,16))
        def to_dict(self):return {'fixture_only':True}
    class Model:
        config=SimpleNamespace(num_register_tokens=0)
        def to(self,device):return self
        def eval(self):return self
        def __call__(self,**batch):return SimpleNamespace(last_hidden_state=torch.ones(len(batch['pixel_values']),5,12))
    def processor(*args,**kwargs):
        assert kwargs['local_files_only'] and kwargs['trust_remote_code'] is False
        return Processor()
    monkeypatch.setattr(transformers.AutoImageProcessor,'from_pretrained',processor)
    monkeypatch.setattr(transformers.AutoModel,'from_pretrained',lambda *args,**kwargs:Model())

def test_frozen_cache_writes_integrity_checked_files_without_weights(prepared,cfg,monkeypatch):
    setup_frames(prepared,cfg);fake_encoder(monkeypatch)
    result=extract_vision(cfg,device='cpu');assert result['frames']==3 and result['embedding_dim']==12
    folder=vision_path(cfg);store=VisionStore(folder,expected_dim=12)
    assert store.features.shape==(3,12) and np.load(folder/'patches.npy').shape==(3,4,12)
    (folder/'frames.csv').write_text('corrupted')
    with pytest.raises(DataIntegrityError):VisionStore(folder,expected_dim=12)

def test_corrupt_image_leaves_no_ready_cache(prepared,cfg,monkeypatch):
    setup_frames(prepared,cfg);fake_encoder(monkeypatch)
    path=Path(cfg['data']['root'])/'AtoB_20_1/frames/0.jpg';path.write_bytes(b'not an image')
    with pytest.raises(DataIntegrityError,match='No samples were silently dropped'):extract_vision(cfg,device='cpu')
    folder=vision_path(cfg);assert (folder/'FAILURE.json').exists() and not (folder/'READY.json').exists()
