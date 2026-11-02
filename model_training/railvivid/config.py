from __future__ import annotations
import copy, json, os
from pathlib import Path
from .errors import ConfigurationError
from .common import fingerprint

DEFAULTS={
 'data': {'root':'D:/Rail-VIVID','artifacts':'../artifacts/railvivid','sampling_hz':2000,'window_samples':1000,'stride_samples':500,
          'positive_radius_m':6.,'buffer_radius_m':18.,'reference_run':'AtoB_20_1','reference_step_samples':500,
          'max_reference_distance_m':5.,'units':'g','chunk_rows':100000},
 'alignment':{'mode':'nominal','landmarks':None,'max_time_residual_s':0.1,'uncertainty_m':15.,'camera_footprint_m':15.,'pooling_bandwidth_m':10.,
              'candidate_offsets_m':[-15.,-10.,-5.,0.,5.,10.,15.],'candidate_tolerance_m':2.,'excluded_image_runs':['AtoB_22.5_4']},
 'split':{'protocol':'combined','seed':42,'test_runs':['AtoB_20_4','AtoB_22.5_4','BtoA_25_4','BtoA_27.5_4'],
          'validation_runs':['AtoB_20_3','BtoA_22.5_3','BtoA_27.5_3'],'test_sites':[7,8,9],'validation_sites':[6],
          'spatial_guard_m':0.,'sample_embargo':0,'block_samples':10000,'test_fraction':0.2,'validation_fraction':0.15,
          'cohort':'vibration','label_sites':None,'condition_direction':None,'test_speeds':[]},
 'vision':{'model_id':'facebook/dinov2-small','revision':None,'local_files_only':True,'batch_size':16,'store_patch_tokens':True,
           'roi':[0.,0.,1.,1.],'mask_boxes':[],'embedding_dir':'vision'},
 'model':{'kind':'tcn','channels':[1,2,3,4,5,6],'width':64,'depth':4,'kernel_size':7,'embedding_dim':128,
          'dropout':0.2,'attention_heads':4,'vision_dim':384,'fusion':'attention','modality_dropout':0.1,
          'initial_encoder':None,'rf_trees':500,'rf_max_depth':None,'rf_min_leaf':2,'memory_patches':20000,'patches_per_reference_frame':16},
 'training':{'seed':42,'device':'cuda','require_cuda':True,'epochs':80,'batch_size':64,'workers':0,
             'learning_rate':0.001,'weight_decay':0.0001,'patience':12,'min_delta':0.0001,'gradient_clip':1.,
             'amp':True,'deterministic':True,'accumulation_steps':1,'resume':None,
             'pretrain_epochs':50,'ssl_variance_weight':1.,'ssl_covariance_weight':0.05,'cross_run_weight':0.,'cross_run_tolerance_m':2.},
 'evaluation':{'threshold':'max_f1','temperature_scaling':True,'event_tolerance_m':10.,'event_merge_gap_m':5.,
               'false_alarm_budget_per_km':1.,'ece_bins':10,'bootstrap_repeats':1000,'bootstrap_seed':73},
 'experiment':{'name':'tcn_combined_seed42','output':'../artifacts/experiments','notes':'Exploratory within-corridor site-proximity evaluation; historical test reuse disclosed.'}
}

def merge(base,updates,path=''):
    for k,v in updates.items():
        if k not in base: raise ConfigurationError(f"Unknown configuration key: {path+k}")
        if isinstance(base[k],dict):
            if not isinstance(v,dict): raise ConfigurationError(f"{path+k} must be an object")
            merge(base[k],v,path+k+'.')
        else: base[k]=v
    return base

def load_config(path,overrides=()):
    path=Path(path).resolve()
    try: updates=json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError) as e: raise ConfigurationError(f"Cannot load configuration {path}: {e}") from e
    if not isinstance(updates,dict): raise ConfigurationError('Configuration root must be a JSON object')
    cfg=merge(copy.deepcopy(DEFAULTS),updates)
    for item in overrides:
        if '=' not in item: raise ConfigurationError(f"Override must be key=value: {item}")
        key,value=item.split('=',1)
        try: value=json.loads(value)
        except ValueError: pass
        node=cfg; pieces=key.split('.')
        for piece in pieces[:-1]:
            if piece not in node or not isinstance(node[piece],dict): raise ConfigurationError(f"Unknown override: {key}")
            node=node[piece]
        merge(node,{pieces[-1]:value},'.'.join(pieces[:-1])+'.')
    if os.environ.get('RAILVIVID_DATA_ROOT'): cfg['data']['root']=os.environ['RAILVIVID_DATA_ROOT']
    if os.environ.get('RAILVIVID_ARTIFACT_ROOT'): cfg['data']['artifacts']=os.environ['RAILVIVID_ARTIFACT_ROOT']
    for section,key in [('data','root'),('data','artifacts'),('experiment','output')]:
        p=Path(os.path.expandvars(os.path.expanduser(str(cfg[section][key]))))
        cfg[section][key]=str(p.resolve() if p.is_absolute() else (path.parent/p).resolve())
    for section,key in [('alignment','landmarks'),('model','initial_encoder'),('training','resume')]:
        if cfg[section][key]:
            p=Path(cfg[section][key]); cfg[section][key]=str(p.resolve() if p.is_absolute() else (path.parent/p).resolve())
    validate_config(cfg)
    cfg['_config_path']=str(path)
    return cfg

def validate_config(c):
    import math
    def finite_tree(value,path=''):
        if isinstance(value,dict):
            for k,v in value.items():finite_tree(v,path+'.'+k)
        elif isinstance(value,list):
            for v in value:finite_tree(v,path)
        elif isinstance(value,float) and not math.isfinite(value):raise ConfigurationError(f'Nonfinite configuration value: {path}')
    finite_tree(c)
    def positive(value,name,integer=False):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or value<=0 or (integer and not isinstance(value,int)):
            raise ConfigurationError(f'{name} must be a positive '+('integer' if integer else 'number'))
    for section,keys in [('data',['sampling_hz','window_samples','stride_samples','chunk_rows','reference_step_samples']),
                         ('training',['epochs','batch_size','patience','accumulation_steps','pretrain_epochs']),
                         ('model',['width','depth','kernel_size','embedding_dim','attention_heads','vision_dim'])]:
        for key in keys: positive(c[section][key],f'{section}.{key}',True)
    for k in ['rf_trees','rf_min_leaf','memory_patches','patches_per_reference_frame']:positive(c['model'][k],'model.'+k,True)
    if c['model']['kernel_size']%2!=1:raise ConfigurationError('model.kernel_size must be odd for residual convolutions')
    if c['training']['cross_run_weight']<0:raise ConfigurationError('cross_run_weight cannot be negative')
    if c['training']['cross_run_weight']>0 and c['model']['kind']=='cnn':raise ConfigurationError('Historical CNN does not support cross-run encoder consistency')
    positive(c['alignment']['pooling_bandwidth_m'],'alignment.pooling_bandwidth_m')
    positive(c['alignment']['candidate_tolerance_m'],'alignment.candidate_tolerance_m')
    positive(c['data']['max_reference_distance_m'],'data.max_reference_distance_m')
    positive(c['evaluation']['event_tolerance_m'],'evaluation.event_tolerance_m')
    if c['evaluation']['event_merge_gap_m']<0 or c['evaluation']['false_alarm_budget_per_km']<0:raise ConfigurationError('Event merge gap and false alarm budget must be nonnegative')
    if c['data']['stride_samples']>c['data']['window_samples']: raise ConfigurationError('Stride cannot exceed window length: gaps would invalidate coverage assumptions')
    if c['data']['window_samples']<64: raise ConfigurationError('Window length must be at least 64 samples')
    if c['data']['units']!='g': raise ConfigurationError('This release expects processed acceleration in g; resolve units before conversion')
    if not 0<c['data']['positive_radius_m']<c['data']['buffer_radius_m']: raise ConfigurationError('Require 0 < positive radius < buffer radius')
    if c['split']['protocol'] not in ['random','block','run','spatial','combined','condition']: raise ConfigurationError('Unknown split protocol')
    for key in ['test_sites','validation_sites']:
        if not c['split'][key] or not set(c['split'][key])<=set(range(1,10)): raise ConfigurationError(f'split.{key} must contain published site IDs 1..9')
    for a,b in [('test_sites','validation_sites'),('test_runs','validation_runs')]:
        if set(c['split'][a])&set(c['split'][b]): raise ConfigurationError(f'{a} overlaps {b}')
    if len(set(c['split']['test_sites'])|set(c['split']['validation_sites']))>=9: raise ConfigurationError('Spatial holdout leaves no training sites')
    if c['split']['cohort'] not in ['vibration','paired']: raise ConfigurationError('cohort must be vibration or paired')
    for k in ['test_fraction','validation_fraction']:
        if not 0<c['split'][k]<1: raise ConfigurationError(f'{k} must be between zero and one')
    if c['split']['test_fraction']+c['split']['validation_fraction']>=1: raise ConfigurationError('Holdout fractions leave no training data')
    for section,keys in [('alignment',['uncertainty_m','camera_footprint_m']),('split',['spatial_guard_m','sample_embargo'])]:
        for k in keys:
            if c[section][k]<0: raise ConfigurationError(f'{section}.{k} must be nonnegative')
    if c['alignment']['mode'] not in ['nominal','landmark']: raise ConfigurationError('Alignment mode must be nominal or landmark')
    if c['alignment']['mode']=='landmark' and not c['alignment']['landmarks']: raise ConfigurationError('Landmark mode needs an independently annotated CSV')
    offsets=c['alignment']['candidate_offsets_m']
    if not offsets or sorted(set(offsets))!=offsets or 0. not in offsets: raise ConfigurationError('Candidate offsets must be unique, sorted, and include zero')
    roi=c['vision']['roi']
    if len(roi)!=4 or not (0<=roi[0]<roi[2]<=1 and 0<=roi[1]<roi[3]<=1): raise ConfigurationError('vision.roi must be normalized [left,top,right,bottom]')
    for box in c['vision']['mask_boxes']:
        if len(box)!=4 or not (0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1): raise ConfigurationError('mask_boxes must use normalized image coordinates')
    if c['model']['kind'] not in ['cnn','tcn','vision','fusion','rf','patch_memory']: raise ConfigurationError('Unknown model kind')
    if c['model']['fusion'] not in ['concat','mean','distance','attention','gated','attention_gated']: raise ConfigurationError('Unknown fusion method')
    if not c['model']['channels'] or not set(c['model']['channels'])<=set(range(1,7)) or len(set(c['model']['channels']))!=len(c['model']['channels']): raise ConfigurationError('channels must be unique IDs 1..6')
    if c['model']['embedding_dim']%c['model']['attention_heads']: raise ConfigurationError('embedding_dim must be divisible by attention_heads')
    for k in ['dropout','modality_dropout']:
        if not 0<=c['model'][k]<1: raise ConfigurationError(f'model.{k} must be in [0,1)')
    if c['training']['workers']<0: raise ConfigurationError('workers must be nonnegative')
    for k in ['learning_rate','gradient_clip']: positive(c['training'][k],f'training.{k}')
    if c['evaluation']['threshold'] not in ['max_f1','false_alarm_budget']: raise ConfigurationError('Unknown threshold policy')
    if not c['experiment']['name'] or Path(c['experiment']['name']).name!=c['experiment']['name'] or c['experiment']['name'] in ['.','..']:
        raise ConfigurationError('Experiment name must be a single safe directory name')
    root=Path(c['data']['root']); artifacts=Path(c['data']['artifacts'])
    if artifacts==root or root in artifacts.parents: raise ConfigurationError('Place artifacts outside the raw dataset directory')

def config_hash(c): return fingerprint({k:v for k,v in c.items() if not k.startswith('_')})
