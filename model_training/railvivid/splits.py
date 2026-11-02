"""Source-aware split manifests and explicitly uncertain candidate assignment."""
from __future__ import annotations
import json,logging
from pathlib import Path
import numpy as np
import pandas as pd
from .common import fingerprint,file_hash,read_json,write_csv,write_json,utc_now
from .errors import ArtifactError,ConfigurationError,DataIntegrityError,LeakageError
from .geometry import nearest_indices
from .prepare import load_prepared

LOG=logging.getLogger('railvivid')
PARTS=['train','validation','test']

def overlap_mask(candidate,reference,embargo=0):
    result=np.zeros(len(candidate),dtype=bool)
    for run in candidate.run_id.unique():
        positions=np.flatnonzero(candidate.run_id.to_numpy()==run); other=reference.loc[reference.run_id.eq(run)].sort_values('start_idx')
        if other.empty:continue
        starts=other.start_idx.to_numpy()-embargo; ends=np.maximum.accumulate(other.end_idx.to_numpy()+embargo)
        q=candidate.iloc[positions]; j=np.searchsorted(starts,q.end_idx.to_numpy(),side='left')-1; valid=j>=0
        result[positions[valid]]=ends[j[valid]]>q.start_idx.to_numpy()[valid]
    return result

def split_identity(cfg,meta):
    a=cfg['alignment'].copy()
    if a['landmarks']: a['landmarks']=file_hash(a['landmarks'])
    return fingerprint({'data':meta['data_fingerprint'],'split':cfg['split'],'alignment':a})

def split_path(cfg):
    root,meta=load_prepared(cfg)
    return root/'splits'/split_identity(cfg,meta)[:20]

def load_split(cfg,verify=True):
    root,meta=load_prepared(cfg); folder=split_path(cfg); info=read_json(folder/'READY.json')
    if info['split_fingerprint']!=split_identity(cfg,meta):raise ArtifactError('Split provenance does not match configuration')
    if verify and file_hash(folder/'manifest.csv')!=info['manifest_sha256']:raise DataIntegrityError('Split manifest was modified')
    frame=pd.read_csv(folder/'manifest.csv',keep_default_na=False)
    frame['candidate_frame_ids']=frame.candidate_frame_ids.map(json.loads)
    frame['candidate_offsets_m']=frame.candidate_offsets_m.map(json.loads)
    return frame,info,folder

def corrected_frames(cfg,frames):
    frames=frames.copy()
    if frames.empty:return frames
    frames=frames.loc[frames.rejection_reason.eq('')&~frames.run_id.isin(cfg['alignment']['excluded_image_runs'])].copy()
    if cfg['alignment']['mode']=='nominal':return frames
    from .common import require_columns
    landmarks=pd.read_csv(cfg['alignment']['landmarks'])
    require_columns(landmarks,['run_id','frame_timestamp_ns','chainage_m','role','source','is_anomaly_marker'],'landmarks')
    if landmarks.is_anomaly_marker.astype(str).str.lower().isin(['true','1','yes']).any():
        raise LeakageError('Strict alignment calibration rejects anomaly-marker labels. Use independent non-anomaly landmarks.')
    if not set(landmarks.role)<=set(['fit','check']):raise DataIntegrityError('Landmark role must be fit or check')
    errors=[]; kept=[]
    for run,g in frames.groupby('run_id'):
        fit=landmarks.loc[landmarks.run_id.eq(run)&landmarks.role.eq('fit')].sort_values('frame_timestamp_ns')
        check=landmarks.loc[landmarks.run_id.eq(run)&landmarks.role.eq('check')]
        if len(fit)<2 or check.empty:raise DataIntegrityError(f'{run}: need >=2 fit landmarks and independent held-out check landmarks')
        x=fit.frame_timestamp_ns.to_numpy(dtype='int64');y=fit.chainage_m.to_numpy(dtype=float)
        if (np.diff(x)<=0).any() or not ((np.diff(y)>0).all() or (np.diff(y)<0).all()):raise DataIntegrityError(f'{run}: nonmonotonic/duplicate calibration landmarks')
        if check.frame_timestamp_ns.isin(fit.frame_timestamp_ns).any():raise LeakageError(f'{run}: calibration and check landmarks overlap')
        check_x=check.frame_timestamp_ns.to_numpy(dtype='int64')
        if ((check_x<x[0])|(check_x>x[-1])).any():raise DataIntegrityError(f'{run}: check landmark lies outside calibration range')
        predicted=np.interp((check_x-x[0])/1e9,(x-x[0])/1e9,y)
        residual=np.abs(predicted-check.chainage_m.to_numpy(float))
        if residual.max()>cfg['alignment']['uncertainty_m']:raise DataIntegrityError(f'{run}: held-out alignment error {residual.max():.2f}m exceeds configured envelope')
        inside=g.timestamp_ns.between(x[0],x[-1]);g=g.loc[inside].copy()
        g['chainage_m']=np.interp((g.timestamp_ns.to_numpy()-x[0])/1e9,(x-x[0])/1e9,y)
        g['alignment_status']='landmark_corrected_with_heldout_checks';kept.append(g)
        errors.append({'run_id':run,'check_count':len(check),'median_error_m':float(np.median(residual)),'max_error_m':float(residual.max())})
    result=pd.concat(kept,ignore_index=True) if kept else frames.iloc[:0]
    result.attrs['alignment_checks']=errors
    return result

def spatial_bounds(sites,groups):
    sites=sites.sort_values('chainage_m'); s=sites.chainage_m.to_numpy();ids=sites.id.to_numpy()
    boundaries=(s[:-1]+s[1:])/2; group_for={site:name for name,values in groups.items() for site in values}
    # Only boundaries separating partitions matter; internal training sites may share context.
    changes=[(boundaries[i],group_for[ids[i]],group_for[ids[i+1]]) for i in range(len(boundaries)) if group_for[ids[i]]!=group_for[ids[i+1]]]
    return sites,boundaries,group_for,changes

def assert_integrity(frame,cfg):
    if frame.window_id.duplicated().any():raise LeakageError('A window appears in multiple partitions')
    for i,a in enumerate(PARTS):
        qa=frame.loc[frame.partition.eq(a)]
        for b in PARTS[i+1:]:
            qb=frame.loc[frame.partition.eq(b)]
            if overlap_mask(qa,qb,cfg['split']['sample_embargo']).any():raise LeakageError(f'Raw source intervals overlap: {a}/{b}')
            fa={f for ids in qa.candidate_frame_ids for f in ids};fb={f for ids in qb.candidate_frame_ids for f in ids}
            if fa&fb:raise LeakageError(f'Images shared across {a}/{b}')
            if cfg['split']['protocol'] in ['run','combined','condition'] and set(qa.run_id)&set(qb.run_id):raise LeakageError(f'Runs shared across {a}/{b}')
            if cfg['split']['protocol'] in ['spatial','combined']:
                sa=set(qa.loc[qa.label.eq(1),'site_id']);sb=set(qb.loc[qb.label.eq(1),'site_id'])
                if sa&sb:raise LeakageError(f'Positive sites shared across {a}/{b}')

def build_split(cfg):
    root,meta=load_prepared(cfg,verify=True);folder=split_path(cfg)
    if folder.exists():raise ArtifactError(f'Split directory already exists: {folder}; use doctor to inspect it')
    data=pd.read_csv(root/'windows.csv');sites=pd.read_csv(root/'sites.csv');frames=pd.read_csv(root/'frames.csv',keep_default_na=False)
    frames=corrected_frames(cfg,frames);settings=cfg['split'];protocol=settings['protocol'];rng=np.random.default_rng(settings['seed'])
    available=set(data.run_id)
    if protocol in ['run','combined']:
        missing=(set(settings['test_runs'])|set(settings['validation_runs']))-available
        if missing:raise ConfigurationError(f'Configured holdout runs absent: {sorted(missing)}')
        if cfg['data']['reference_run'] in set(settings['test_runs'])|set(settings['validation_runs']):raise LeakageError('Reference geometry run must be a training run')
    spatial=protocol in ['spatial','combined']
    groups={'test':set(settings['test_sites']),'validation':set(settings['validation_sites'])}
    groups['train']=set(sites.id)-groups['test']-groups['validation']
    site_order,boundaries,group_for,changes=spatial_bounds(sites,groups)
    ordered_ids=site_order.id.to_numpy()
    data['sector_site_id']=ordered_ids[np.searchsorted(boundaries,data.chainage_m)]
    if spatial:data['spatial_partition']=data.sector_site_id.map(group_for)
    else:data['spatial_partition']=''
    data['partition']='train'
    if protocol in ['run','combined']:
        data.loc[data.run_id.isin(settings['validation_runs']),'partition']='validation'
        data.loc[data.run_id.isin(settings['test_runs']),'partition']='test'
    elif protocol=='condition':
        direction=settings['condition_direction'];speeds=settings['test_speeds']
        if direction not in ['AtoB','BtoA'] or not speeds:raise ConfigurationError('condition protocol requires condition_direction and test_speeds')
        data=data.loc[data.direction.eq(direction)].copy()
        data.loc[data.target_speed_kmh.isin(speeds),'partition']='test'
        val=data.run_id.isin(settings['validation_runs'])
        if (val&data.partition.eq('test')).any():raise ConfigurationError('Validation runs overlap held-out speed')
        data.loc[val,'partition']='validation'
    elif protocol=='random':
        order=rng.permutation(len(data));n_test=int(len(data)*settings['test_fraction']);n_val=int(len(data)*settings['validation_fraction'])
        data.loc[data.index[order[:n_test]],'partition']='test';data.loc[data.index[order[n_test:n_test+n_val]],'partition']='validation'
    elif protocol=='block':
        keys=data.run_id+':'+(data.start_idx//settings['block_samples']).astype(str);unique=keys.unique();order=rng.permutation(len(unique))
        nt=int(len(unique)*settings['test_fraction']);nv=int(len(unique)*settings['validation_fraction'])
        data.loc[keys.isin(unique[order[:nt]]),'partition']='test';data.loc[keys.isin(unique[order[nt:nt+nv]]),'partition']='validation'
    if protocol=='spatial':data['partition']=data.spatial_partition
    if protocol=='combined':data=data.loc[data.partition.eq(data.spatial_partition)].copy()
    initial=len(data);data=data.loc[data.reference_distance_m.le(cfg['data']['max_reference_distance_m'])].copy()
    # A row's entire raw support and every candidate envelope must stay within its permitted spatial component.
    def spatial_safe(low,high,part):
        if not spatial:return True
        guard=settings['spatial_guard_m']
        for boundary,left,right in changes:
            if low-guard<boundary<high+guard:return False
        center=(low+high)/2
        owner=group_for[ordered_ids[np.searchsorted(boundaries,center)]]
        return owner==part
    if spatial:
        data=data.loc[[spatial_safe(a,b,p) for a,b,p in zip(data.support_min_m,data.support_max_m,data.partition)]].copy()
    parts={p:data.loc[data.partition.eq(p)].copy() for p in PARTS}
    parts['validation']=parts['validation'].loc[~overlap_mask(parts['validation'],parts['test'],settings['sample_embargo'])]
    parts['train']=parts['train'].loc[~overlap_mask(parts['train'],pd.concat([parts['validation'],parts['test']]),settings['sample_embargo'])]
    data=pd.concat(parts.values()).sort_index()
    candidate_map={};offset_map={};reserved=set()
    frame_by_run={r:g.sort_values('chainage_m') for r,g in frames.groupby('run_id')}
    envelope=cfg['alignment']['uncertainty_m']+cfg['alignment']['camera_footprint_m']
    # Test keeps priority; validation and training cannot reuse its exact image content identity.
    for part in reversed(PARTS):
        current=set()
        for run,g in data.loc[data.partition.eq(part)].groupby('run_id'):
            ff=frame_by_run.get(run)
            for row in g.itertuples():
                ids=[];offsets=[]
                if ff is not None and not ff.empty:
                    pos=ff.chainage_m.to_numpy();query=row.chainage_m+np.asarray(cfg['alignment']['candidate_offsets_m'])
                    indices=nearest_indices(pos,query)
                    for desired,j in zip(query,indices):
                        f=ff.iloc[j];fid=f.frame_id;delta=float(f.chainage_m-row.chainage_m)
                        if fid in reserved or fid in ids or abs(f.chainage_m-desired)>cfg['alignment']['candidate_tolerance_m']:continue
                        if f.reference_distance_m>cfg['data']['max_reference_distance_m']:continue
                        if not spatial_safe(f.chainage_m-envelope,f.chainage_m+envelope,part):continue
                        # For same-run temporal splits, candidate envelope may not reach another partition's raw support.
                        other=data.loc[~data.partition.eq(part)&data.run_id.eq(run)]
                        if not other.empty and ((other.support_min_m<f.chainage_m+envelope)&(other.support_max_m>f.chainage_m-envelope)).any():continue
                        ids.append(fid);offsets.append(delta);current.add(fid)
                candidate_map[row.window_id]=ids;offset_map[row.window_id]=offsets
        reserved.update(current)
    data['candidate_frame_ids']=data.window_id.map(candidate_map);data['candidate_offsets_m']=data.window_id.map(offset_map)
    data['vision_available']=data.candidate_frame_ids.map(bool)
    if settings['cohort']=='paired':data=data.loc[data.vision_available].copy()
    data['eligible_supervised']=data.label.ge(0)
    if settings['label_sites'] is not None:
        if not set(settings['label_sites'])<=groups['train']:raise ConfigurationError('Label budget includes non-training site IDs')
        mask=data.partition.eq('train')&data.label.eq(1)&~data.site_id.isin(settings['label_sites'])
        data.loc[mask,'eligible_supervised']=False
    assert_integrity(data,cfg)
    summary=[]
    for part,g in data.groupby('partition'):
        supervised=g.loc[g.eligible_supervised]
        if set(supervised.label)!={0,1}:raise DataIntegrityError(f'{part}: both supervised classes required after guarding/cohort/budget filters')
        summary.append({'partition':part,'rows':len(g),'supervised':len(supervised),'positives':int(supervised.label.eq(1).sum()),
            'runs':g.run_id.nunique(),'positive_sites':sorted(g.loc[g.label.eq(1),'site_id'].unique().tolist()),'paired_rows':int(g.vision_available.sum())})
    if set(data.partition)!=set(PARTS):raise DataIntegrityError('Split produced an empty partition')
    serial=data.copy();serial['candidate_frame_ids']=serial.candidate_frame_ids.map(json.dumps);serial['candidate_offsets_m']=serial.candidate_offsets_m.map(json.dumps)
    folder.mkdir(parents=True,exist_ok=False);write_csv(folder/'manifest.csv',serial)
    info={'created':utc_now(),'schema_version':1,'data_fingerprint':meta['data_fingerprint'],'split_fingerprint':split_identity(cfg,meta),
        'manifest_sha256':file_hash(folder/'manifest.csv'),'protocol':protocol+('_purged' if protocol in ['random','block'] else ''),
        'cohort':settings['cohort'],'initial_assigned_rows':initial,'retained_rows':len(data),'summary':summary,
        'alignment_mode':cfg['alignment']['mode'],'alignment_checks':frames.attrs.get('alignment_checks',[]),
        'camera_envelope_assumption_m':envelope,'claim':'source guards checked; physical correspondence remains conditional on landmark evidence',
        'test_history':'Prior published local test scores were inspected; folds are exploratory unless independently reserved.'}
    write_json(folder/'READY.json',info);return info
