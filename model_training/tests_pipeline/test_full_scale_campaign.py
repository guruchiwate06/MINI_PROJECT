"""Full-scale orchestration and performance regressions without fitting any model."""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from railvivid.common import file_hash,read_json,write_json,write_csv
from railvivid.benchmark import plan_suite
from railvivid.campaign import create_plan,inspect_plan,validate_plan_source,main,jobs_for
from railvivid.config import load_config
from railvivid.dataset import WindowDataset,training_normalizer
from railvivid.models import make_model
from railvivid.splits import build_split,load_split
from railvivid.training import make_loader
from railvivid.errors import ArtifactError,DataIntegrityError,ConfigurationError


def paired_fixture(prepared):
    windows=pd.read_csv(prepared/'windows.csv')
    frames=windows[['run_id','timestamp_ns','chainage_m','reference_distance_m']].copy()
    frames['frame_id']=['f'+str(i) for i in range(len(frames))]
    frames['relative_path']=frames.frame_id+'.jpg';frames['rejection_reason']=''
    write_csv(prepared/'frames.csv',frames)
    meta=read_json(prepared/'READY.json');meta['artifacts']['frames.csv']=file_hash(prepared/'frames.csv')
    write_json(prepared/'READY.json',meta)


def test_full_config_does_not_enable_consistency_in_control_models(tmp_path,cfg):
    cfg['model'].update(kind='fusion',fusion='attention_gated');cfg['training']['cross_run_weight']=.1
    plan_suite(cfg,tmp_path/'plan')
    plan,jobs=jobs_for(tmp_path/'plan/plan.json')
    assert len(jobs)==117
    assert sum(j['method']=='pretrain' for j,_ in jobs)==9
    for job,c in jobs:
        assert c['training']['cross_run_weight']==(.1 if job['method'] in ['full','cross_run'] else 0.)
        if job['method']=='full':
            assert c['model']['fusion']=='attention_gated' and c['model']['initial_encoder']


def test_workload_counts_all_methods_folds_and_seeds_without_fitting(tmp_path,prepared,cfg):
    paired_fixture(prepared);suite=tmp_path/'suite.json'
    write_json(suite,{'methods':['tcn','ssl_tcn','full'],'seeds':[42,43],'folds':[1,2,3]})
    plan=tmp_path/'plan/plan.json';create_plan(cfg,suite,plan)
    result=inspect_plan(plan,require_features=False)
    assert result['planned_tasks']==24 and not result['training_started']
    # Every fixture fold has ten labelled train windows: ceil(10/4)=3.
    assert result['max_neural_updates']==3*2*(3*80*3+50*2)
    assert result['failures']==0 and len(result['splits'])==3
    configuration=plan.parent/read_json(plan)['jobs'][0]['config']
    configuration.write_text(configuration.read_text()+' ')
    with pytest.raises(DataIntegrityError,match='config changed'):validate_plan_source(plan)


def test_preflight_reports_missing_features_without_fitting(tmp_path,prepared,cfg):
    paired_fixture(prepared);suite=tmp_path/'suite.json'
    write_json(suite,{'methods':['vision','patch_memory'],'seeds':[42],'folds':[3]})
    plan=tmp_path/'plan/plan.json';create_plan(cfg,suite,plan)
    result=inspect_plan(plan)
    assert result['failures']==2 and not result['training_started']
    assert all('error' in check for check in result['checks'])


def test_source_change_and_cpu_block_training_before_dispatch(tmp_path,prepared,cfg,monkeypatch):
    paired_fixture(prepared);suite=tmp_path/'suite.json'
    write_json(suite,{'methods':['tcn'],'seeds':[42],'folds':[3]})
    plan=tmp_path/'plan/plan.json';create_plan(cfg,suite,plan)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    import railvivid.benchmark as benchmark
    def forbidden(*a,**kw):raise AssertionError('A rejected preflight must never dispatch training')
    monkeypatch.setattr(benchmark,'execute_suite',forbidden)
    config=plan.parent/read_json(plan)['jobs'][0]['config']
    assert main(['train','--plan',str(plan),'--config',str(config)])==1
    import railvivid.campaign as campaign
    monkeypatch.setattr(campaign,'source_hash',lambda:'changed')
    with pytest.raises(ArtifactError,match='code changed'):validate_plan_source(plan)


def test_shared_fusion_encoding_preserves_eval_logits_and_avoids_auxiliary_pass(cfg):
    cfg['model'].update(kind='fusion',fusion='attention_gated')
    model=make_model(cfg).eval();batch={'waveform':torch.randn(3,6,128),'vision':torch.randn(3,3,12),
        'vision_mask':torch.tensor([[True,False,True],[False,False,False],[True,True,True]]),
        'offsets':torch.zeros(3,3),'quality':torch.zeros(3,3,2)}
    calls=[];handle=model.encoder.net.register_forward_hook(lambda *args:calls.append(1))
    with torch.inference_mode():
        logits,encoding=model.forward_features(batch)
        auxiliary=model.auxiliary_loss(batch,torch.tensor([1.,0.,1.]),encoding)
        assert len(calls)==1
        old_auxiliary=model.auxiliary_loss(batch,torch.tensor([1.,0.,1.]))
        torch.testing.assert_close(auxiliary,old_auxiliary)
        torch.testing.assert_close(logits,model(batch))
    handle.remove()
    assert not hasattr(model,'last_encoding')


def test_compact_dataset_determinism_matches_worker_count(prepared,cfg):
    build_split(cfg);frame,_,_=load_split(cfg);train=frame.loc[frame.partition.eq('train')]
    normalizer=training_normalizer(prepared,train,cfg['model']['channels'])
    ds=WindowDataset(prepared,train,normalizer,cfg,augment=True)
    first=list(make_loader(ds,cfg,epoch=2));cfg['training']['workers']=1
    second=list(make_loader(ds,cfg,epoch=2))
    for a,b in zip(first,second):
        for key in a:torch.testing.assert_close(a[key],b[key])
    third=list(make_loader(ds,cfg,epoch=3))
    assert not torch.equal(first[0]['waveform'],third[0]['waveform'])
    assert ds.max_open_runs>=train.run_id.nunique()


@pytest.mark.parametrize('override',['training.workers=-1','training.workers=1.5','training.prefetch_factor=0','training.max_open_runs=0'])
def test_invalid_loader_settings_rejected(tmp_path,override):
    path=tmp_path/'config.json';write_json(path,{})
    with pytest.raises(ConfigurationError):load_config(path,[override])
