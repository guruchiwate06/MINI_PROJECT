"""Matched paired AP differences; resample the same site/run weights for both models."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from .common import require_columns,write_json,file_hash
from .errors import DataIntegrityError,ArtifactError
from .late_fusion import matched_predictions

def paired_ap_difference(a,b,repeats=1000,seed=73):
    a,second=matched_predictions(a,b)
    require_columns(a,['run_id','site_id'],'paired comparison')
    mask=a.label.isin([0,1]);frame=a.loc[mask];y=frame.label.to_numpy();first=frame.probability.to_numpy();second=second[mask.to_numpy()]
    if len(np.unique(y))<2:raise DataIntegrityError('Paired AP requires both classes')
    runs=sorted(frame.run_id.unique());sites=sorted(frame.site_id.unique());rng=np.random.default_rng(seed)
    ri=pd.Categorical(frame.run_id,categories=runs).codes;si=pd.Categorical(frame.site_id,categories=sites).codes
    delta=float(average_precision_score(y,second)-average_precision_score(y,first));values=[]
    for _ in range(repeats):
        rw=np.bincount(rng.integers(0,len(runs),len(runs)),minlength=len(runs));sw=np.bincount(rng.integers(0,len(sites),len(sites)),minlength=len(sites));w=rw[ri]*sw[si]
        if w[y==0].sum() and w[y==1].sum():values.append(average_precision_score(y,second,sample_weight=w)-average_precision_score(y,first,sample_weight=w))
    return {'delta_average_precision_second_minus_first':delta,'lower':float(np.percentile(values,2.5)) if values else None,
        'upper':float(np.percentile(values,97.5)) if values else None,'valid_draws':len(values),'run_clusters':len(runs),'spatial_clusters':len(sites),
        'interpretation':'paired two-way clustered exploratory resampling; very few sites, no guaranteed statistical power'}

def compare_files(first,second,output,repeats=1000,seed=73):
    from pathlib import Path
    if Path(output).exists():raise ArtifactError('Comparison output exists')
    result=paired_ap_difference(pd.read_csv(first),pd.read_csv(second),repeats,seed)
    result.update({'first_sha256':file_hash(first),'second_sha256':file_hash(second),'seed':seed,'requested_draws':repeats})
    write_json(output,result);return result
