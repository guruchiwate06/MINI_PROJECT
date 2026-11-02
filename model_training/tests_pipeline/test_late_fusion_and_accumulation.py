import numpy as np
import pandas as pd
import pytest
from railvivid.late_fusion import matched_predictions
from railvivid.training import accumulation_examples
from railvivid.errors import DataIntegrityError

def test_late_fusion_matches_identity_not_row_order():
    a=pd.DataFrame({'window_id':['a','b'],'label':[0,1],'probability':[.1,.9]})
    b=pd.DataFrame({'window_id':['b','a'],'label':[1,0],'probability':[.8,.2]})
    _,prob=matched_predictions(a,b);np.testing.assert_allclose(prob,[.2,.8])
    with pytest.raises(DataIntegrityError):matched_predictions(a,b.iloc[:1])
    b.loc[0,'label']=0
    with pytest.raises(DataIntegrityError):matched_predictions(a,b)

def test_accumulation_handles_last_partial_group_without_an_optimizer():
    assert accumulation_examples(150,64,3,0,drop_last=True)==128
    assert accumulation_examples(150,64,3,0,drop_last=False)==150
    assert accumulation_examples(150,64,2,2,drop_last=False)==22
    assert accumulation_examples(128,64,1,1,drop_last=False)==64
