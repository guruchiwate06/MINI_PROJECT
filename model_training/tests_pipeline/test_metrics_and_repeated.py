import copy
import numpy as np
import pandas as pd
import pytest
from railvivid.metrics import classification_metrics,event_metrics,select_threshold,clustered_ap_interval
from railvivid.repeated import repeated_pass_study
from railvivid.errors import ConfigurationError

def predictions(run='a'):
    return pd.DataFrame({'run_id':[run]*5,'chainage_m':[0.,1.,10.,20.,21.],'support_min_m':[-.5,.5,9.5,19.5,20.5],
        'support_max_m':[.5,1.5,10.5,20.5,21.5],'probability':[.9,.8,.1,.95,.8],'label':[1,-1,0,1,-1],'site_id':[1,1,1,2,2]})

def test_ambiguous_binary_labels_not_silently_counted_negative():
    m=classification_metrics([1,0,-1],[.9,.2,.99]);assert m['observations']==2 and m['ignored_ambiguous']==1 and m['f1']==1.
    assert classification_metrics([0,0],[.1,.2])['roc_auc'] is None

def test_events_deduplicate_alert_windows_one_to_one_and_keep_transition_rows():
    sites=pd.DataFrame({'id':[1,2],'chainage_m':[0.,20.]});m,events=event_metrics(predictions(),sites,.5,tolerance_m=2,merge_gap_m=3)
    assert m['true_positive_events']==2 and m['false_positive_events']==0 and m['event_recall']==1.
    assert m['evaluated_distance_km']==pytest.approx(.005)
    assert len(events)==2

def test_threshold_comes_from_validation(cfg):
    p=predictions();sites=pd.DataFrame({'id':[1,2],'chainage_m':[0.,20.]})
    threshold=select_threshold(p,sites,cfg['evaluation']);assert 0<=threshold<=1.00000000001
    assert classification_metrics(p.label,p.probability,threshold)['f1']==1.

def test_repeated_pass_budget_is_independent_run_count(tmp_path,cfg):
    sites=pd.DataFrame({'id':[1,2],'chainage_m':[0.,20.]})
    validation=pd.concat([predictions('v1'),predictions('v2')]);test=pd.concat([predictions('t1'),predictions('t2')])
    for frame in [validation,test]:
        normal=frame.label.eq(0);frame.loc[normal,'chainage_m']=50.;frame.loc[normal,'support_min_m']=49.;frame.loc[normal,'support_max_m']=52.
    with pytest.raises(ConfigurationError,match='BOTH'):repeated_pass_study(test,validation,sites,tmp_path/'bad',cfg,counts=[8])
    result=repeated_pass_study(test,validation,sites,tmp_path/'good',cfg,counts=[1,2],methods=['mean'],repeats=2,segment_m=1.)
    assert result['evaluations']==4
    protocol=__import__('json').loads((tmp_path/'good/protocol.json').read_text());assert protocol['fixed_common_segment_population']
