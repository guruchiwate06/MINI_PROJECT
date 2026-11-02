"""RF and frozen patch-memory baselines, fitting exclusively permitted train rows."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import expit,logit
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
import joblib
from .common import atomic_write,file_hash,read_json,save_array,utc_now,write_csv,write_json,finite
from .errors import ArtifactError,DataIntegrityError
from .prepare import load_prepared
from .dataset import VisionStore
from .vision import vision_path

def statistics_matrix(root,frame,channels):
    folder=root/'statistics';info=read_json(folder/'READY.json')
    if info['data_fingerprint']!=read_json(root/'READY.json')['data_fingerprint']:raise DataIntegrityError('Statistical features belong to another prepared dataset')
    if file_hash(folder/'features.csv')!=info['features_sha256']:raise DataIntegrityError('Statistical feature cache changed')
    features=pd.read_csv(folder/'features.csv')
    columns=[c for c in info['columns'] if any(c.startswith(f'Channel{i}_') for i in channels)]
    merged=frame[['window_id']].merge(features,on='window_id',how='left',validate='one_to_one',sort=False)
    if merged[columns].isna().any().any():raise DataIntegrityError('Statistical features missing for requested windows')
    values=merged[columns].to_numpy();finite(values,'RF input features');return values,columns

def patch_cache(store):
    path=store.folder/'patches.npy'
    if not path.is_file():raise ArtifactError('Patch-memory baseline requires vision.store_patch_tokens=true and an extracted patch cache')
    return np.load(path,mmap_mode='r')

def patch_scores(store,frame,memory,device):
    patches=patch_cache(store);result=[]
    bank=torch.tensor(memory,device=device,dtype=torch.float32)
    cache={}
    with torch.inference_mode():
        for row in frame.itertuples():
            if not row.candidate_frame_ids:raise DataIntegrityError('Patch-memory baseline cannot score a missing-image row')
            nearest=int(np.argmin(abs(np.asarray(row.candidate_offsets_m))));fid=row.candidate_frame_ids[nearest]
            if fid not in cache:
                x=torch.tensor(np.asarray(patches[store.lookup[fid]],dtype='float32'),device=device)
                x=torch.nn.functional.normalize(x,dim=-1)
                # Chunk reference bank to bound GPU memory; same exact nearest distance.
                distance=torch.full((len(x),),float('inf'),device=device)
                for start in range(0,len(bank),2048):distance=torch.minimum(distance,torch.cdist(x,bank[start:start+2048]).min(1).values)
                cache[fid]=float(distance.topk(max(1,int(np.ceil(.01*len(distance))))).values.mean())
            result.append(cache[fid])
    return np.asarray(result)

def fit_monotone_calibration(labels,scores):
    mask=np.isin(labels,[0,1]);y=np.asarray(labels)[mask];z=np.asarray(scores)[mask]
    if len(np.unique(y))<2:raise DataIntegrityError('Score calibration requires both validation classes')
    center=float(np.median(z));scale=max(float(np.std(z)),1e-6);x=(z-center)/scale
    def objective(ab):
        t=ab[0]*x+ab[1]
        return np.mean(np.logaddexp(0,t)-y*t)+1e-4*np.sum(ab*ab)
    initial=[1.,float(logit(np.clip(y.mean(),1e-6,1-1e-6)))]
    fit=minimize(objective,initial,method='L-BFGS-B',bounds=[(0.,100.),(-30.,30.)])
    if not fit.success:raise DataIntegrityError(f'Validation monotone score calibration failed: {fit.message}')
    return {'slope':float(fit.x[0]/scale),'intercept':float(fit.x[1]-fit.x[0]*center/scale),'source':'validation only','method':'monotone affine score calibration'}

def fit_baseline(cfg,split,split_info,folder,identity):
    from .training import finalize_validation,save_checkpoint,vision_for
    root,data=load_prepared(cfg);train=split.loc[split.partition.eq('train')&split.eligible_supervised];val=split.loc[split.partition.eq('validation')].copy()
    if cfg['model']['kind']=='rf':
        x,columns=statistics_matrix(root,train,cfg['model']['channels']);xv,_=statistics_matrix(root,val,cfg['model']['channels'])
        model=RandomForestClassifier(n_estimators=cfg['model']['rf_trees'],max_depth=cfg['model']['rf_max_depth'],
            min_samples_leaf=cfg['model']['rf_min_leaf'],class_weight='balanced',random_state=cfg['training']['seed'],n_jobs=-1)
        model.fit(x,train.label.to_numpy());atomic_write(folder/'forest.joblib',lambda f:joblib.dump(model,f),binary=True)
        val['logit']=logit(np.clip(model.predict_proba(xv)[:,1],1e-7,1-1e-7))
        write_json(folder/'forest.json',{'columns':columns,'sha256':file_hash(folder/'forest.joblib'),'data_fingerprint':data['data_fingerprint']})
        artifact=folder/'forest.joblib'
    else:
        store=vision_for(cfg,split);patches=patch_cache(store);rng=np.random.default_rng(cfg['training']['seed'])
        ids=sorted({f for candidate in train.loc[train.label.eq(0),'candidate_frame_ids'] for f in candidate})
        if not ids:raise DataIntegrityError('No permitted normal training frames for reference memory')
        sites=pd.read_csv(root/'sites.csv');train_sites=set(train.sector_site_id);site_positions=sites.loc[sites.id.isin(train_sites),'chainage_m'].to_numpy()
        envelope=cfg['data']['buffer_radius_m']+cfg['alignment']['uncertainty_m']+cfg['alignment']['camera_footprint_m']
        ids=[fid for fid in ids if np.min(abs(store.index.iloc[store.lookup[fid]].nominal_chainage_m-site_positions))>envelope]
        if not ids:raise DataIntegrityError('No normal reference frames remain after positional-uncertainty exclusion')
        refs=[];used=[];seen_hashes=set()
        for fid in rng.permutation(ids):
            index=store.lookup[fid];digest=store.index.iloc[index].image_sha256
            if digest in seen_hashes:continue
            seen_hashes.add(digest);a=np.asarray(patches[index],dtype='float32');finite(a,'normal patch memory')
            choose=rng.choice(len(a),size=min(len(a),cfg['model']['patches_per_reference_frame']),replace=False)
            refs.append(a[choose]);used.append(fid)
            if sum(len(r) for r in refs)>=cfg['model']['memory_patches']:break
        memory=np.concatenate(refs)[:cfg['model']['memory_patches']];memory=memory/np.linalg.norm(memory,axis=1,keepdims=True).clip(1e-8)
        save_array(folder/'memory.npy',memory);write_json(folder/'memory_frames.json',used)
        raw=patch_scores(store,val,memory,torch.device(cfg['training']['device']));cal=fit_monotone_calibration(val.label.to_numpy(),raw)
        write_json(folder/'score_transform.json',cal);val['logit']=cal['slope']*raw+cal['intercept'];artifact=folder/'memory.npy'
    finalize_validation(cfg,folder,val,pd.read_csv(root/'sites.csv'),identity)
    write_json(folder/'DONE.json',{'completed':utc_now(),'stage':'supervised','experiment_fingerprint':identity,'checkpoint_sha256':file_hash(artifact),'test_evaluated':False})
    return {'run':str(folder),'complete':True,'test_evaluated':False}

def baseline_predict(cfg,frame,folder,device,corruption=None):
    root,_=load_prepared(cfg);result=frame.copy()
    if cfg['model']['kind']=='rf':
        if corruption:raise DataIntegrityError('RF raw-signal corruption requires recomputing statistical features; cached clean features cannot represent corrupted signals')
        meta=read_json(folder/'forest.json')
        if file_hash(folder/'forest.joblib')!=meta['sha256']:raise DataIntegrityError('Forest checkpoint hash mismatch')
        model=joblib.load(folder/'forest.joblib');x,columns=statistics_matrix(root,frame,cfg['model']['channels'])
        if columns!=meta['columns']:raise DataIntegrityError('RF feature ordering differs from training')
        result['logit']=logit(np.clip(model.predict_proba(x)[:,1],1e-7,1-1e-7))
    else:
        store=VisionStore(vision_path(cfg,corruption),cfg['model']['vision_dim']);memory=np.load(folder/'memory.npy',allow_pickle=False)
        transform=read_json(folder/'score_transform.json');raw=patch_scores(store,frame,memory,device)
        result['logit']=transform['slope']*raw+transform['intercept']
    return result
