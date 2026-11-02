from __future__ import annotations
import json,logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from .common import file_hash,fingerprint,read_json,select_device,seed_everything,utc_now,write_csv,write_json
from .errors import ArtifactError,ConfigurationError,DataIntegrityError
from .dataset import WindowDataset
from .models import make_model
from .prepare import load_prepared
from .splits import load_split
from .training import run_path,experiment_identity,load_checkpoint,predict,vision_for
from .metrics import classification_metrics,event_metrics,clustered_ap_interval

def evaluate(cfg,partition='test',corruption=None,tag=None,device=None):
    if partition not in ['validation','test']:raise ConfigurationError('Evaluation partition must be validation or test')
    root,_=load_prepared(cfg,verify=True);split,info,_=load_split(cfg);folder=run_path(cfg);done=read_json(folder/'DONE.json')
    if done['stage']!='supervised':raise ArtifactError('A self-supervised encoder is not a fitted anomaly detector')
    identity=experiment_identity(cfg,info)
    if done['experiment_fingerprint']!=identity:raise ArtifactError('Evaluation configuration/split differs from trained model')
    calibration=read_json(folder/'calibration.json')
    if calibration['experiment_fingerprint']!=identity:raise ArtifactError('Calibration belongs to another experiment')
    corruption=corruption or {};case=tag or ('clean' if not corruption else fingerprint(corruption)[:12])
    if Path(case).name!=case or case in ['.','..']:raise ConfigurationError('Evaluation tag must be a safe directory name')
    destination=folder/'evaluations'/f'{partition}_{case}'
    if destination.exists():raise ArtifactError(f'Evaluation already exists and is preserved: {destination}')
    if abs(corruption.get('alignment_prior_shift_m',0))>max(abs(x) for x in cfg['alignment']['candidate_offsets_m']):
        raise ConfigurationError('Requested shift exceeds the pre-guarded candidate range; do not silently expand test support')
    seed_everything(cfg['training']['seed'],cfg['training']['deterministic'])
    dev=select_device(device or cfg['training']['device'],require_cuda=False);frame=split.loc[split.partition.eq(partition)].copy()
    kind=cfg['model']['kind'];artifact=folder/('forest.joblib' if kind=='rf' else 'memory.npy' if kind=='patch_memory' else 'best.pt')
    if file_hash(artifact)!=done['checkpoint_sha256']:raise DataIntegrityError('Trained checkpoint changed after run completion')
    if kind in ['rf','patch_memory']:
        from .baselines import baseline_predict
        predictions=baseline_predict(cfg,frame,folder,dev,corruption)
    else:
        visual_corruption={k:v for k,v in corruption.items() if k in ['blur_radius_px','brightness_factor']}
        store=vision_for(cfg,split,visual_corruption or None)
        normalizer=read_json(folder/'normalizer.json');dataset=WindowDataset(root,frame,normalizer,cfg,store,corruption=corruption)
        model=make_model(cfg).to(dev);state=load_checkpoint(folder/'best.pt');model.load_state_dict(state['model'],strict=True)
        predictions=predict(model,dataset,cfg,dev)
    predictions['probability']=expit(predictions.logit/calibration['temperature']);threshold=calibration['threshold']
    predictions['prediction']=predictions.probability.ge(threshold).astype(int)
    return save_evaluation(cfg,predictions,folder,partition,case,identity,info,done['checkpoint_sha256'],kind,corruption)

def save_evaluation(cfg,predictions,folder,partition,case,identity,info,checkpoint_sha,kind,corruption=None):
    root,_=load_prepared(cfg);calibration=read_json(folder/'calibration.json');threshold=calibration['threshold'];corruption=corruption or {}
    destination=folder/'evaluations'/f'{partition}_{case}'
    if destination.exists():raise ArtifactError(f'Evaluation already exists and is preserved: {destination}')
    sites=pd.read_csv(root/'sites.csv');metrics=classification_metrics(predictions.label,predictions.probability,threshold,cfg['evaluation']['ece_bins'])
    events,event_rows=event_metrics(predictions,sites,threshold,cfg['evaluation']['event_tolerance_m'],cfg['evaluation']['event_merge_gap_m'])
    metrics['events']=events;metrics['ap_cluster_interval']=clustered_ap_interval(predictions,cfg['evaluation']['bootstrap_repeats'],cfg['evaluation']['bootstrap_seed'])
    metrics['corruption']=corruption;metrics['synthetic_stress_test']=bool(corruption);metrics['threshold_source']='unchanged validation calibration'
    metrics['inference']=predictions.attrs.get('inference',{'measured':False})
    group_rows=[]
    for group in ['run_id','site_id','direction','target_speed_kmh']:
        for value,g in predictions.groupby(group):
            result=classification_metrics(g.label,g.probability,threshold,cfg['evaluation']['ece_bins']) if g.label.isin([0,1]).any() else {}
            group_rows.append({'group':group,'value':value,**{k:v for k,v in result.items() if not isinstance(v,(dict,list))}})
    destination.mkdir(parents=True,exist_ok=False)
    serial=predictions.drop(columns=['candidate_frame_ids','candidate_offsets_m'],errors='ignore');write_csv(destination/'predictions.csv',serial)
    write_csv(destination/'events.csv',event_rows);write_csv(destination/'subgroups.csv',pd.DataFrame(group_rows));write_json(destination/'metrics.json',metrics)
    write_json(destination/'provenance.json',{'evaluated':utc_now(),'partition':partition,'experiment_fingerprint':identity,
        'split_fingerprint':info['split_fingerprint'],'cohort_fingerprint':fingerprint(sorted(predictions.window_id.tolist())),
        'model_kind':kind,'seed':cfg['training']['seed'],'alignment':cfg['alignment']['mode'],'corruption':corruption,
        'checkpoint_sha256':checkpoint_sha,'predictions_sha256':file_hash(destination/'predictions.csv'),
        'test_access_disclosed':partition=='test','notes':cfg['experiment']['notes']})
    render_report(destination,predictions,metrics)
    write_json(destination/'READY.json',{'completed':utc_now(),'artifacts':{p.name:file_hash(p) for p in destination.iterdir() if p.is_file()}})
    return {'evaluation':str(destination),'metrics':metrics}

def render_report(folder,predictions,metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve,roc_curve
    y=predictions.loc[predictions.label.isin([0,1]),'label'];p=predictions.loc[predictions.label.isin([0,1]),'probability']
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    if len(y.unique())==2:
        precision,recall,_=precision_recall_curve(y,p);axes[0].plot(recall,precision);axes[0].axhline(y.mean(),ls='--',color='gray')
        fpr,tpr,_=roc_curve(y,p);axes[1].plot(fpr,tpr)
    reliability=metrics['reliability_bins'];axes[2].plot([r['mean_probability'] for r in reliability],[r['positive_fraction'] for r in reliability],marker='o');axes[2].plot([0,1],[0,1],ls='--',color='gray')
    axes[0].set(xlabel='Recall',ylabel='Precision',title='Precision–recall');axes[1].set(xlabel='False positive rate',ylabel='True positive rate',title='ROC')
    axes[2].set(xlabel='Predicted probability',ylabel='Observed positive fraction',title='Calibration')
    fig.tight_layout();fig.savefig(folder/'curves.png',dpi=150);plt.close(fig)
    rows=['# Evaluation report','',f"Observations: {metrics['observations']}; positives: {metrics['positive']}; ambiguous binary labels omitted: {metrics['ignored_ambiguous']}.",'',
        '| Metric | Value |','| --- | --- |']
    for key in ['average_precision','roc_auc','precision','recall','f1','brier','negative_log_likelihood','ece','aurc_classification_error']:
        rows.append(f"| {key} | {metrics[key]} |")
    rows+=['','Event metrics include transition rows and use nominal site matching with a validation-fixed tolerance. Distance is evaluated support length, not the entire advertised corridor.',
        '','```json',json.dumps(metrics['events'],indent=2),'```','','Cluster intervals are exploratory with very few physical sites. Window counts are not independent defect counts.']
    from .common import atomic_write
    atomic_write(folder/'REPORT.md',lambda f:f.write('\n'.join(rows)+'\n'))
