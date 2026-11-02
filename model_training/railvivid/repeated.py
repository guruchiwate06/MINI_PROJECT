"""Inference-only repeated-pass study with validation operating points per count."""
import numpy as np
import pandas as pd
from .common import fingerprint,require_columns,write_csv,write_json
from .errors import ConfigurationError,DataIntegrityError,ArtifactError
from .metrics import classification_metrics,event_metrics,select_threshold,coverage_intervals

def segment_table(predictions,sites,segment_m,positive_radius,buffer_radius):
    p=predictions.copy();p['segment']=(p.chainage_m//segment_m).astype(int)
    grouped=p.groupby(['run_id','segment']).agg(probability=('probability','mean')).reset_index()
    sets=[]
    for run,g in grouped.groupby('run_id'):
        spans=coverage_intervals(p.loc[p.run_id.eq(run)])
        complete={int(segment) for segment in g.segment
                  if any(a<=segment*segment_m and (segment+1)*segment_m<=b for a,b in spans)}
        sets.append(complete)
    common=set.intersection(*sets) if sets else set()
    if not common:raise DataIntegrityError('No common observed track segments across the held-out pass pool')
    grouped=grouped.loc[grouped.segment.isin(common)].copy();grouped['chainage_m']=(grouped.segment+.5)*segment_m
    distances=abs(grouped.chainage_m.to_numpy()[:,None]-sites.chainage_m.to_numpy()[None]);j=distances.argmin(1);d=distances[np.arange(len(j)),j]
    grouped['site_id']=sites.id.to_numpy()[j];grouped['label']=np.where(d<=positive_radius,1,np.where(d<=buffer_radius,-1,0))
    grouped['support_min_m']=grouped.segment*segment_m;grouped['support_max_m']=(grouped.segment+1)*segment_m
    return grouped

def aggregate_passes(table,runs,method,identifier):
    if method not in ['mean','median','max']:raise ConfigurationError('Repeated-pass aggregation must be mean, median or max')
    selected=table.loc[table.run_id.isin(runs)]
    if selected.groupby('segment').run_id.nunique().ne(len(runs)).any():raise DataIntegrityError('Unequal per-segment pass budget')
    a=selected.groupby('segment').agg(probability=('probability',method),label=('label','first'),site_id=('site_id','first'),chainage_m=('chainage_m','first'),
        support_min_m=('support_min_m','first'),support_max_m=('support_max_m','first')).reset_index()
    a['run_id']=identifier;return a

def repeated_pass_study(test,validation,sites,output,cfg,counts=(1,2,3),methods=('mean','median','max'),repeats=30,seed=2026,segment_m=10.):
    from pathlib import Path
    output=Path(output)
    if output.exists():raise ArtifactError(f'Repeated-pass output already exists: {output}')
    for frame in [test,validation]:require_columns(frame,['run_id','chainage_m','probability'],'repeated-pass predictions')
    if set(test.run_id)&set(validation.run_id):raise DataIntegrityError('Repeated-pass test and validation run pools overlap')
    pools=[sorted(validation.run_id.unique()),sorted(test.run_id.unique())]
    if not counts or min(counts)<1 or max(counts)>min(map(len,pools)):
        raise ConfigurationError(f'Counts {list(counts)} require that many independent passes in BOTH pools; validation/test have {list(map(len,pools))}')
    if repeats<1 or segment_m<=0:raise ConfigurationError('Positive repeats and segment_m required')
    v=segment_table(validation,sites,segment_m,cfg['data']['positive_radius_m'],cfg['data']['buffer_radius_m'])
    t=segment_table(test,sites,segment_m,cfg['data']['positive_radius_m'],cfg['data']['buffer_radius_m'])
    rng=np.random.default_rng(seed);orders=[(rng.permutation(pools[0]).tolist(),rng.permutation(pools[1]).tolist()) for _ in range(repeats)]
    rows=[];draws=[]
    for method in methods:
        for count in counts:
            validation_draws=pd.concat([aggregate_passes(v,vr[:count],method,f'validation_draw_{i}') for i,(vr,tr) in enumerate(orders)],ignore_index=True)
            policy={**cfg['evaluation'],'threshold':'false_alarm_budget'}
            threshold=select_threshold(validation_draws,sites,policy)
            for i,(vr,tr) in enumerate(orders):
                result=aggregate_passes(t,tr[:count],method,f'test_draw_{i}')
                classification=classification_metrics(result.label,result.probability,threshold);events,_=event_metrics(result,sites,threshold,policy['event_tolerance_m'],policy['event_merge_gap_m'])
                rows.append({'aggregation':method,'passes':count,'draw':i,'validation_threshold':threshold,
                    'average_precision':classification['average_precision'],'f1':classification['f1'],**{k:v for k,v in events.items() if not isinstance(v,str)}})
                draws.append({'aggregation':method,'passes':count,'draw':i,'test_runs':tr[:count],'validation_runs':vr[:count]})
    output.mkdir(parents=True);write_csv(output/'results.csv',pd.DataFrame(rows));write_json(output/'draws.json',draws)
    write_json(output/'protocol.json',{'seed':seed,'counts':list(counts),'methods':list(methods),'repeats':repeats,'segment_m':segment_m,
        'fixed_common_segment_population':True,'threshold_source':'validation-only repeated pools at each count, fixed requested false-alarm budget',
        'interpretation':'draw variability reuses sites/runs; it is not independent replication or a confidence interval',
        'test_run_pool':pools[1],'validation_run_pool':pools[0]})
    return {'output':str(output),'evaluations':len(rows)}
