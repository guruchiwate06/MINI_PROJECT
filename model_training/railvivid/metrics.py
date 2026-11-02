from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment,minimize_scalar
from scipy.special import expit
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_fscore_support,confusion_matrix
from .common import finite
from .errors import DataIntegrityError

def classification_metrics(labels,probabilities,threshold=.5,bins=10):
    y=np.asarray(labels);p=np.asarray(probabilities,dtype=float)
    if y.ndim!=1 or p.ndim!=1 or len(y)!=len(p):raise DataIntegrityError('Labels and probabilities must be matching one-dimensional arrays')
    if not np.isin(y,[-1,0,1]).all():raise DataIntegrityError('Labels must be -1 (ambiguous), 0 or 1')
    finite(p,'predicted probabilities')
    if ((p<0)|(p>1)).any():raise DataIntegrityError('Probabilities must be in [0,1]')
    mask=np.isin(y,[0,1]);y=y[mask].astype(int);p=p[mask]
    if not len(y):raise DataIntegrityError('No binary-labelled observations to evaluate')
    predicted=(p>=threshold).astype(int)
    precision,recall,f1,_=precision_recall_fscore_support(y,predicted,average='binary',zero_division=0)
    both=len(np.unique(y))==2;clipped=np.clip(p,1e-12,1-1e-12)
    ece=0.;reliability=[]
    for i in range(bins):
        member=(p>=i/bins)&(p<(i+1)/bins if i<bins-1 else p<=1)
        if member.any():
            confidence=float(p[member].mean());fraction=float(y[member].mean());count=int(member.sum())
            ece+=count/len(y)*abs(confidence-fraction)
            reliability.append({'bin':i,'count':count,'mean_probability':confidence,'positive_fraction':fraction})
    # Selective risk is classification error, ordered by distance from probability .5.
    order=np.argsort(-abs(p-.5),kind='stable');error=(predicted[order]!=y[order]).astype(float)
    selective=np.cumsum(error)/np.arange(1,len(y)+1)
    return {'observations':len(y),'ignored_ambiguous':int((~mask).sum()),'positive':int(y.sum()),'prevalence':float(y.mean()),
        'precision':float(precision),'recall':float(recall),'f1':float(f1),'roc_auc':float(roc_auc_score(y,p)) if both else None,
        'average_precision':float(average_precision_score(y,p)) if both else None,'threshold':float(threshold),
        'confusion_matrix':confusion_matrix(y,predicted,labels=[0,1]).tolist(),'brier':float(np.mean((p-y)**2)),
        'negative_log_likelihood':float(-np.mean(y*np.log(clipped)+(1-y)*np.log1p(-clipped))),
        'ece':float(ece),'reliability_bins':reliability,'aurc_classification_error':float(selective.mean())}

def fit_temperature(labels,logits):
    y=np.asarray(labels);z=np.asarray(logits,dtype=float);mask=np.isin(y,[0,1]);y=y[mask];z=z[mask]
    if len(np.unique(y))<2:raise DataIntegrityError('Temperature calibration needs both validation classes')
    finite(z,'validation logits')
    def objective(logt):
        scaled=z/np.exp(logt)
        return np.mean(np.logaddexp(0,scaled)-y*scaled)
    result=minimize_scalar(objective,bounds=(-3,3),method='bounded')
    if not result.success:raise DataIntegrityError('Validation temperature optimization failed')
    return float(np.exp(result.x))

def coverage_intervals(frame):
    spans=[]
    for a,b in frame[['support_min_m','support_max_m']].sort_values('support_min_m').itertuples(index=False,name=None):
        if spans and a<=spans[-1][1]+1e-6:spans[-1][1]=max(spans[-1][1],b)
        else:spans.append([float(a),float(b)])
    return spans

def event_metrics(predictions,sites,threshold,tolerance_m=10.,merge_gap_m=5.):
    """One-to-one nominal-site matching; transition rows remain in event evaluation."""
    records=[];totals={'true_positive_events':0,'false_positive_events':0,'false_negative_events':0,'covered_site_traversals':0,'evaluated_distance_km':0.}
    errors=[]
    for run,g in predictions.groupby('run_id'):
        spans=coverage_intervals(g);distance=sum(b-a for a,b in spans)/1000
        truth=sites.loc[[any(a<=s<=b for a,b in spans) for s in sites.chainage_m]].copy()
        alerts=g.loc[g.probability.ge(threshold)].sort_values('chainage_m')
        clusters=[]
        for row in alerts.itertuples():
            if not clusters or row.chainage_m-clusters[-1][-1].chainage_m>merge_gap_m:clusters.append([row])
            else:clusters[-1].append(row)
        peaks=[max(c,key=lambda r:r.probability) for c in clusters]
        matched_pred=set();matched_truth=set()
        if peaks and len(truth):
            costs=abs(np.asarray([p.chainage_m for p in peaks])[:,None]-truth.chainage_m.to_numpy()[None])
            i,j=linear_sum_assignment(np.where(costs<=tolerance_m,costs,1e9))
            for a,b in zip(i,j):
                if costs[a,b]<=tolerance_m:
                    matched_pred.add(int(a));matched_truth.add(int(b));errors.append(float(costs[a,b]))
                    records.append({'run_id':run,'status':'true_positive','site_id':int(truth.iloc[b].id),'position_m':float(peaks[a].chainage_m),'error_m':float(costs[a,b])})
        for i,p in enumerate(peaks):
            if i not in matched_pred:records.append({'run_id':run,'status':'false_positive','site_id':None,'position_m':float(p.chainage_m),'error_m':None})
        for j,s in enumerate(truth.itertuples()):
            if j not in matched_truth:records.append({'run_id':run,'status':'false_negative','site_id':int(s.id),'position_m':float(s.chainage_m),'error_m':None})
        totals['true_positive_events']+=len(matched_pred);totals['false_positive_events']+=len(peaks)-len(matched_pred)
        totals['false_negative_events']+=len(truth)-len(matched_truth);totals['covered_site_traversals']+=len(truth);totals['evaluated_distance_km']+=distance
    tp=totals['true_positive_events'];fp=totals['false_positive_events'];n=totals['covered_site_traversals'];km=totals['evaluated_distance_km']
    totals.update({'event_recall':tp/n if n else None,'event_precision':tp/(tp+fp) if tp+fp else 0.,'false_alarms_per_km':fp/km if km else None,
        'localization_mae_m':float(np.mean(errors)) if errors else None,'localization_p95_m':float(np.percentile(errors,95)) if errors else None,
        'target_semantics':'nominal documented site proximity, not independently adjudicated frame defects'})
    return totals,pd.DataFrame(records,columns=['run_id','status','site_id','position_m','error_m'])

def select_threshold(validation,sites,settings):
    mask=validation.label.isin([0,1]);y=validation.loc[mask,'label'].to_numpy();p=validation.loc[mask,'probability'].to_numpy()
    if set(y)!={0,1}:raise DataIntegrityError('Threshold selection needs both validation classes')
    candidates=np.unique(np.r_[0.,np.quantile(p,np.linspace(0,1,min(257,len(p)))),np.nextafter(1.,2.)])
    best=None
    for threshold in candidates:
        if settings['threshold']=='max_f1':
            pred=p>=threshold;tp=np.sum(pred&(y==1));fp=np.sum(pred&(y==0));fn=np.sum(~pred&(y==1))
            objective=(2*tp/max(2*tp+fp+fn,1),float(threshold))
        else:
            events,_=event_metrics(validation,sites,threshold,settings['event_tolerance_m'],settings['event_merge_gap_m'])
            fa=events['false_alarms_per_km'];recall=events['event_recall']
            if fa is None or recall is None:raise DataIntegrityError('No validation route/site coverage for false-alarm operating point')
            if fa>settings['false_alarm_budget_per_km']:continue
            objective=(recall,-fa,float(threshold))
        if best is None or objective>best[0]:best=(objective,float(threshold))
    if best is None:raise DataIntegrityError('No validation threshold satisfies the requested policy')
    return best[1]

def clustered_ap_interval(predictions,repeats=1000,seed=73):
    p=predictions.loc[predictions.label.isin([0,1])];runs=sorted(p.run_id.unique());sites=sorted(p.site_id.unique())
    if len(runs)<2 or len(sites)<2:return {'lower':None,'upper':None,'reason':'Fewer than two run/site clusters'}
    ri=pd.Categorical(p.run_id,categories=runs).codes;si=pd.Categorical(p.site_id,categories=sites).codes
    y=p.label.to_numpy();score=p.probability.to_numpy();rng=np.random.default_rng(seed);values=[]
    for _ in range(repeats):
        rw=np.bincount(rng.integers(0,len(runs),len(runs)),minlength=len(runs));sw=np.bincount(rng.integers(0,len(sites),len(sites)),minlength=len(sites));weight=rw[ri]*sw[si]
        if weight[y==0].sum() and weight[y==1].sum():values.append(average_precision_score(y,score,sample_weight=weight))
    if not values:return {'lower':None,'upper':None,'reason':'All bootstrap draws degenerate'}
    return {'lower':float(np.percentile(values,2.5)),'upper':float(np.percentile(values,97.5)),'valid_draws':len(values),
        'run_clusters':len(runs),'spatial_clusters':len(sites),'method':'two-way pigeonhole resampling; small-cluster exploratory interval, not independent-defect certainty'}
