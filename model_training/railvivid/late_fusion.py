"""True score-level fusion of independently fitted modality baselines."""
from pathlib import Path
import numpy as np
import pandas as pd
from .common import read_json,write_json,write_csv,file_hash,utc_now
from .errors import ArtifactError,DataIntegrityError
from .training import run_path,experiment_identity
from .prepare import load_prepared
from .splits import load_split
from .metrics import classification_metrics,select_threshold

def matched_predictions(a,b):
    if a.window_id.duplicated().any() or b.window_id.duplicated().any():raise DataIntegrityError('Late-fusion inputs contain repeated window IDs')
    if set(a.window_id)!=set(b.window_id):raise DataIntegrityError('Late fusion requires identical cohorts, not an intersection chosen after seeing results')
    b=b.set_index('window_id').loc[a.window_id]
    if not np.array_equal(a.label.to_numpy(),b.label.to_numpy()):raise DataIntegrityError('Late-fusion input labels disagree')
    return a.copy(),b.probability.to_numpy()

def fit_late(cfg,vibration_run,vision_run):
    root,_=load_prepared(cfg);_,info,_=load_split(cfg);identity=experiment_identity(cfg,info);folder=run_path(cfg)
    if folder.exists():raise ArtifactError('Late-fusion experiment already exists')
    sources=[Path(vibration_run),Path(vision_run)];source_info=[]
    for source in sources:
        run=read_json(source/'run.json');done=read_json(source/'DONE.json')
        if run['split']['split_fingerprint']!=info['split_fingerprint']:raise DataIntegrityError('Late-fusion source split differs from target split')
        source_info.append({'run_name':source.name,'experiment_fingerprint':done['experiment_fingerprint'],'checkpoint_sha256':done['checkpoint_sha256'],
            'validation_sha256':file_hash(source/'validation_predictions.csv')})
    a,pv=matched_predictions(pd.read_csv(sources[0]/'validation_predictions.csv'),pd.read_csv(sources[1]/'validation_predictions.csv'))
    best=None
    # Small prespecified validation grid; no test scores are read here.
    for weight in [0.,.25,.5,.75,1.]:
        p=weight*a.probability.to_numpy()+(1-weight)*pv
        ap=classification_metrics(a.label,p)['average_precision']
        if ap is None:raise DataIntegrityError('Validation AP is undefined for fusion')
        if best is None or ap>best[0]:best=(ap,weight,p)
    a['probability']=best[2];sites=pd.read_csv(root/'sites.csv');threshold=select_threshold(a,sites,cfg['evaluation'])
    folder.mkdir(parents=True);write_json(folder/'config.json',cfg)
    write_json(folder/'fusion.json',{'vibration_weight':best[1],'source':'validation only','sources':source_info,'weight_grid':[0.,.25,.5,.75,1.]})
    write_json(folder/'run.json',{'experiment_fingerprint':identity,'split':info,'stage':'late_fusion'})
    write_json(folder/'calibration.json',{'experiment_fingerprint':identity,'threshold':threshold,'temperature':1.,'source':'validation only','method':'independent calibrated-score mixture'})
    write_csv(folder/'validation_predictions.csv',a);write_json(folder/'validation_metrics.json',classification_metrics(a.label,a.probability,threshold))
    write_json(folder/'DONE.json',{'completed':utc_now(),'stage':'late_fusion','experiment_fingerprint':identity,'checkpoint_sha256':file_hash(folder/'fusion.json'),'test_evaluated':False})
    return {'run':str(folder),'complete':True,'test_evaluated':False}

def evaluate_late(cfg,partition='test'):
    from .evaluation import save_evaluation
    _,info,_=load_split(cfg);identity=experiment_identity(cfg,info);folder=run_path(cfg);done=read_json(folder/'DONE.json')
    if done['experiment_fingerprint']!=identity or file_hash(folder/'fusion.json')!=done['checkpoint_sha256']:raise ArtifactError('Late-fusion identity/checkpoint changed')
    fusion=read_json(folder/'fusion.json');predictions=[]
    for source in fusion['sources']:
        source_folder=Path(cfg['experiment']['output'])/source['run_name'];current=read_json(source_folder/'DONE.json')
        if current['checkpoint_sha256']!=source['checkpoint_sha256']:raise DataIntegrityError('Underlying modality model changed after fusion calibration')
        if partition=='validation':path=source_folder/'validation_predictions.csv'
        else:
            path=source_folder/'evaluations/test_clean/predictions.csv';provenance=read_json(path.parent/'provenance.json')
            if provenance['experiment_fingerprint']!=source['experiment_fingerprint']:raise DataIntegrityError('Modality predictions belong to a different fitted model')
            if provenance['checkpoint_sha256']!=source['checkpoint_sha256'] or file_hash(path)!=provenance['predictions_sha256']:raise DataIntegrityError('Underlying modality predictions/checkpoint provenance changed')
        if partition=='validation' and file_hash(path)!=source['validation_sha256']:raise DataIntegrityError('Validation predictions changed after fusion calibration')
        predictions.append(pd.read_csv(path))
    a,b=matched_predictions(*predictions);weight=fusion['vibration_weight'];a['probability']=weight*a.probability+(1-weight)*b
    a['prediction']=a.probability.ge(read_json(folder/'calibration.json')['threshold']).astype(int)
    return save_evaluation(cfg,a,folder,partition,'clean',identity,info,done['checkpoint_sha256'],'late_fusion')
