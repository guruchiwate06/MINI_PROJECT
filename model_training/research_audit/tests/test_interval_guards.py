import importlib.util
from pathlib import Path
import pandas as pd
spec=importlib.util.spec_from_file_location('guards',Path(__file__).parents[1]/'scripts/interval_guards.py')
guards=importlib.util.module_from_spec(spec); spec.loader.exec_module(guards)

def windows(items):
    return pd.DataFrame(items,columns=['run_id','start_idx','end_idx'])

def test_shared_samples_different_rows_are_detected():
    a=windows([('a',0,1000),('a',1000,2000),('b',0,1000)])
    b=windows([('a',500,1500)])
    assert guards.overlap_mask(a,b).tolist()==[True,True,False]

def test_touching_half_open_endpoints_do_not_overlap():
    assert not guards.overlap_mask(windows([('a',0,1000)]),windows([('a',1000,2000)]))[0]

def test_long_reference_interval_is_not_hidden_by_shorter_intervals():
    a=windows([('a',500,600)])
    b=windows([('a',0,1000),('a',100,200)])
    assert guards.overlap_mask(a,b)[0]

def test_purge_preserves_test_and_removes_boundary_overlap():
    tr=windows([('a',0,1000),('a',3000,4000)])
    va=windows([('a',500,1500),('a',2000,3000)])
    te=windows([('a',2500,3500)])
    tr,va,te2=guards.purge_training_and_validation(tr,va,te)
    assert len(tr)==0 and len(va)==1
    pd.testing.assert_frame_equal(te,te2)
    assert not guards.overlap_mask(va,te).any()
