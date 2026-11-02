import numpy as np
import pandas as pd
import pytest
from railvivid.prepare import prepare
from railvivid.errors import DataIntegrityError

def test_invalid_raw_timestamps_fail_without_ready_or_input_mutation(cfg):
    from pathlib import Path
    from railvivid.common import file_hash
    run=Path(cfg['data']['root'])/'AtoB_20_1';run.mkdir(parents=True)
    data={'Timestamp':pd.date_range('2025-04-11',periods=256,freq='500us').astype(str),
          **{f'Channel{i}':np.linspace(0,1,256) for i in range(1,7)},'Latitude':np.linspace(40.235,40.234,256),'Longitude':np.linspace(-77.89,-77.88,256)}
    frame=pd.DataFrame(data);frame.loc[20,'Timestamp']=frame.loc[19,'Timestamp'];path=run/'AtoB_20_1.csv';frame.to_csv(path,index=False);before=file_hash(path)
    with pytest.raises(DataIntegrityError,match='timestamps'):prepare(cfg)
    root=Path(cfg['data']['artifacts']);assert (root/'FAILURE.json').exists() and not (root/'READY.json').exists()
    assert file_hash(path)==before
