import pandas as pd
import pytest
from railvivid.comparison import paired_ap_difference
from railvivid.errors import DataIntegrityError

def test_paired_difference_uses_identical_observations():
    a=pd.DataFrame({'window_id':['a','b','c','d'],'run_id':['r1','r1','r2','r2'],'site_id':[1,1,2,2],
        'label':[0,1,0,1],'probability':[.1,.9,.2,.8]})
    b=a.copy();result=paired_ap_difference(a,b,repeats=10)
    assert result['delta_average_precision_second_minus_first']==0 and result['lower']==0 and result['upper']==0
    with pytest.raises(DataIntegrityError):paired_ap_difference(a,b.iloc[:2],repeats=10)
