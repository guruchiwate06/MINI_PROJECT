import copy,json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from railvivid.common import file_hash,write_json,write_csv
from railvivid.splits import build_split,load_split
from railvivid.dataset import training_normalizer,WindowDataset
from railvivid.models import make_model
from railvivid.training import experiment_identity,save_checkpoint,run_path,load_checkpoint
from railvivid.evaluation import evaluate
from railvivid.errors import ArtifactError,DataIntegrityError

def test_complete_evaluation_path_with_untrained_fixture_checkpoint(prepared,cfg):
    """Tests IO/metrics only; random fixture weights are never presented as research scores."""
    build_split(cfg);split,info,_=load_split(cfg);folder=run_path(cfg);folder.mkdir(parents=True)
    identity=experiment_identity(cfg,info);model=make_model(cfg).eval()
    save_checkpoint(folder/'best.pt',{'model':model.state_dict()})
    normalizer=training_normalizer(prepared,split.loc[split.partition.eq('train')],cfg['model']['channels'])
    write_json(folder/'normalizer.json',normalizer)
    write_json(folder/'DONE.json',{'stage':'supervised','experiment_fingerprint':identity,'checkpoint_sha256':file_hash(folder/'best.pt')})
    write_json(folder/'calibration.json',{'temperature':1.,'threshold':.5,'experiment_fingerprint':identity})
    result=evaluate(cfg,device='cpu');output=Path(result['evaluation'])
    assert (output/'REPORT.md').is_file() and (output/'curves.png').is_file()
    predictions=pd.read_csv(output/'predictions.csv');assert set(predictions.run_id)=={'AtoB_20_3'}
    with pytest.raises(ArtifactError,match='preserved'):evaluate(cfg,device='cpu')
    changed=copy.deepcopy(cfg);changed['model']['width']=16
    with pytest.raises(ArtifactError,match='differs'):evaluate(changed,device='cpu',tag='changed')

def test_corrupt_checkpoint_has_actionable_error(tmp_path):
    path=tmp_path/'bad.pt';path.write_bytes(b'not a checkpoint')
    with pytest.raises(ArtifactError,match='Cannot load checkpoint'):load_checkpoint(path)

def test_missing_channel_corruption_is_explicit_not_imputation(prepared,cfg):
    build_split(cfg);frame,_,_=load_split(cfg);train=frame.loc[frame.partition.eq('train')]
    norm=training_normalizer(prepared,train,cfg['model']['channels'])
    ds=WindowDataset(prepared,train,norm,cfg,corruption={'missing_channels':[2,4]})
    item=ds[0];assert not item['waveform'][1].any() and not item['waveform'][3].any()
    assert item['waveform'][0].abs().sum()>0

def test_cross_run_pairs_are_constructed_only_from_dataset_rows(prepared,cfg):
    cfg['split']['protocol']='spatial';cfg['training']['cross_run_weight']=.1
    build_split(cfg);frame,_,_=load_split(cfg);train=frame.loc[frame.partition.eq('train')]
    norm=training_normalizer(prepared,train,cfg['model']['channels']);ds=WindowDataset(prepared,train,norm,cfg,augment=True)
    assert ds.partners
    for i,partners in ds.partners.items():
        for j in partners:
            assert ds.frame.iloc[i].run_id!=ds.frame.iloc[j].run_id
            assert ds.frame.iloc[j].window_id in set(train.window_id)
    assert ds[0]['partner_available']

def test_cli_training_is_blocked_before_any_fit(tmp_path,monkeypatch):
    from railvivid.cli import main
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    path=tmp_path/'config.json';path.write_text('{}')
    assert main(['train','--config',str(path)])==2
