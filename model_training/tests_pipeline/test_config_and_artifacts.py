import copy,json
from pathlib import Path
import numpy as np
import pytest
from railvivid.common import atomic_write,file_hash,read_json,save_array
from railvivid.config import load_config,validate_config
from railvivid.errors import ConfigurationError,ArtifactError

@pytest.mark.parametrize('mutate',[
    lambda c:c['split'].update(test_runs=c['split']['validation_runs']),
    lambda c:c['model'].update(channels=[1,1]),
    lambda c:c['data'].update(stride_samples=10000),
    lambda c:c['vision'].update(roi=[.5,0,.1,1]),
    lambda c:c['model'].update(embedding_dim=17),
    lambda c:c['alignment'].update(mode='landmark'),
    lambda c:c['experiment'].update(name='../overwrite'),
])
def test_invalid_config_rejected(cfg,mutate):
    mutate(cfg)
    with pytest.raises(ConfigurationError):validate_config(cfg)

def test_unknown_keys_and_relative_paths(tmp_path):
    p=tmp_path/'config.json';p.write_text(json.dumps({'training':{'epohs':5}}))
    with pytest.raises(ConfigurationError,match='Unknown'):load_config(p)
    p.write_text(json.dumps({'data':{'root':'raw','artifacts':'prepared'},'experiment':{'output':'results'}}))
    c=load_config(p,['model.width=32']);assert c['data']['root']==str(tmp_path/'raw');assert c['model']['width']==32

def test_atomic_failure_preserves_old_file(tmp_path):
    p=tmp_path/'result.txt';p.write_text('previous')
    def interrupted(f):f.write('partial');raise OSError('disk failure')
    with pytest.raises(OSError):atomic_write(p,interrupted)
    assert p.read_text()=='previous';assert list(tmp_path.iterdir())==[p]

def test_safe_array_roundtrip(tmp_path):
    a=np.arange(12).reshape(3,4);p=tmp_path/'array.npy';save_array(p,a)
    np.testing.assert_array_equal(np.load(p,allow_pickle=False),a)
    with pytest.raises(ArtifactError):read_json(tmp_path/'missing.json')
