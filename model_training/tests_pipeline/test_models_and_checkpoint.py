import copy
import numpy as np
import pytest
import torch
from railvivid.models import make_model,TimeFrequencySSL
from railvivid.training import save_checkpoint,load_checkpoint,rng_state,restore_rng
from railvivid.common import seed_everything,select_device
from railvivid.errors import ConfigurationError,ArtifactError

def batch(cfg):
    return {'waveform':torch.randn(3,6,128),'vision':torch.randn(3,3,12),'vision_mask':torch.tensor([[True,True,False],[False,False,False],[True,False,False]]),
        'offsets':torch.tensor([[-2.,0.,0.],[0.,0.,0.],[0.,0.,0.]]),'quality':torch.rand(3,3,2)}

@pytest.mark.parametrize('kind',['cnn','tcn','vision','fusion'])
@pytest.mark.parametrize('fusion',['concat','mean','distance','attention','gated','attention_gated'])
def test_forward_only_finite_including_missing_vision(cfg,kind,fusion):
    cfg['model'].update(kind=kind,fusion=fusion)
    # Historical CNN needs longer windows due to two large pooling stages.
    b=batch(cfg)
    if kind=='cnn':b['waveform']=torch.randn(3,6,1000)
    model=make_model(cfg).eval()
    with torch.inference_mode():out=model(b)
    assert out.shape==(3,) and torch.isfinite(out).all()

def test_time_frequency_views_are_inference_only(cfg):
    model=TimeFrequencySSL(cfg).eval()
    with torch.inference_mode():a,b=model(torch.randn(4,6,128))
    assert a.shape==b.shape==(4,16);assert torch.isfinite(a).all() and torch.isfinite(b).all()

def test_checkpoint_weights_only_and_rng_restore(tmp_path):
    seed_everything(42);state=rng_state();path=tmp_path/'checkpoint.pt'
    save_checkpoint(path,{'rng':state,'weights':torch.randn(2,3),'epoch':4})
    loaded=load_checkpoint(path);restore_rng(loaded['rng']);a=np.random.random(3);restore_rng(loaded['rng']);b=np.random.random(3)
    np.testing.assert_array_equal(a,b);assert loaded['epoch']==4

def test_no_silent_cpu_training_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(ConfigurationError,match='no CPU training fallback'):select_device('cuda',True)
    with pytest.raises(ConfigurationError):select_device('cpu',True)
    assert select_device('cpu',False).type=='cpu'
