import json
import numpy as np
import pandas as pd
import pytest
from railvivid.splits import overlap_mask,build_split,load_split,assert_integrity,corrected_frames
from railvivid.dataset import WindowDataset,training_normalizer,merged_intervals
from railvivid.errors import DataIntegrityError,LeakageError,ArtifactError

def test_shared_source_samples_not_just_duplicate_rows():
    a=pd.DataFrame({'run_id':['a','b'],'start_idx':[0,0],'end_idx':[1000,1000]})
    b=pd.DataFrame({'run_id':['a'],'start_idx':[500],'end_idx':[1500]})
    assert overlap_mask(a,b).tolist()==[True,False]
    b['start_idx']=1000;assert not overlap_mask(a,b).any()

def test_nested_reference_interval():
    a=pd.DataFrame({'run_id':['a'],'start_idx':[500],'end_idx':[600]})
    b=pd.DataFrame({'run_id':['a','a'],'start_idx':[0,100],'end_idx':[1000,200]})
    assert overlap_mask(a,b)[0]

def test_real_group_semantics_and_lazy_windows(prepared,cfg):
    info=build_split(cfg);frame,_,folder=load_split(cfg)
    train=frame.loc[frame.partition.eq('train')];val=frame.loc[frame.partition.eq('validation')];test=frame.loc[frame.partition.eq('test')]
    assert set(train.run_id)=={'AtoB_20_1'} and set(test.site_id)=={7,8,9}
    assert not set(train.site_id)&set(test.site_id);assert_integrity(frame,cfg)
    normalizer=training_normalizer(prepared,train,cfg['model']['channels'])
    assert normalizer['unique_training_samples']==len(train)*128
    dataset=WindowDataset(prepared,train,normalizer,cfg);item=dataset[0]
    assert tuple(item['waveform'].shape)==(6,128) and not item['vision_mask'].any()
    with pytest.raises(ArtifactError):build_split(cfg)
    (folder/'manifest.csv').write_text('changed')
    with pytest.raises(DataIntegrityError):load_split(cfg)

def test_normalizer_excludes_test_values_and_deduplicates_overlap(prepared,cfg):
    train=pd.DataFrame({'run_id':['AtoB_20_1']*2,'start_idx':[0,64],'end_idx':[128,192],'window_id':['a','b']})
    result=training_normalizer(prepared,train,[1,3])
    raw=np.load(prepared/'runs/AtoB_20_1/acceleration.npy')[:192,[0,2]]
    assert result['unique_training_samples']==192;np.testing.assert_allclose(result['mean'],raw.mean(0),atol=1e-7)

def test_landmark_fit_and_check_must_be_independent(tmp_path,cfg):
    path=tmp_path/'landmarks.csv';cfg['alignment'].update(mode='landmark',landmarks=str(path),uncertainty_m=1.)
    frame=pd.DataFrame({'run_id':['r']*3,'rejection_reason':['']*3,'timestamp_ns':[1000000000,2000000000,3000000000],'chainage_m':[0,10,20]})
    landmarks=pd.DataFrame({'run_id':['r']*3,'frame_timestamp_ns':[1000000000,3000000000,2000000000],'chainage_m':[0,20,10.2],
        'role':['fit','fit','check'],'source':['survey']*3,'is_anomaly_marker':[False]*3});landmarks.to_csv(path,index=False)
    corrected=corrected_frames(cfg,frame);assert corrected.attrs['alignment_checks'][0]['max_error_m']==pytest.approx(.2)
    landmarks.loc[2,'frame_timestamp_ns']=1000000000;landmarks.to_csv(path,index=False)
    with pytest.raises(LeakageError):corrected_frames(cfg,frame)
    landmarks.loc[0,'is_anomaly_marker']=True;landmarks.to_csv(path,index=False)
    with pytest.raises(LeakageError):corrected_frames(cfg,frame)

def test_label_budget_only_excludes_permitted_training_positive_labels(prepared,cfg):
    cfg['split']['label_sites']=[1,2];build_split(cfg);frame,_,_=load_split(cfg)
    labelled=frame.loc[frame.partition.eq('train')&frame.eligible_supervised&frame.label.eq(1)]
    assert set(labelled.site_id)=={1,2};assert len(frame.loc[frame.partition.eq('test')&frame.label.eq(1)])==3
