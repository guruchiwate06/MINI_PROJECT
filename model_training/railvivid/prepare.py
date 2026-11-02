from __future__ import annotations
import json, logging, os, re, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from src.evaluation.splitters import GROUND_TRUTH_ANOMALIES
from .common import file_hash,fingerprint,finite,read_json,require_columns,save_array,utc_now,write_csv,write_json
from .errors import ArtifactError,DataIntegrityError,ConfigurationError
from .geometry import TrackReference,nearest_indices

LOG=logging.getLogger('railvivid')
CHANNELS=[f'Channel{i}' for i in range(1,7)]
RUN_RE=re.compile(r'^(AtoB|BtoA)_(\d+(?:\.\d+)?)_(\d+)$')

def prepared_root(cfg): return Path(cfg['data']['artifacts'])

def load_prepared(cfg,verify=False):
    root=prepared_root(cfg); meta=read_json(root/'READY.json')
    if meta.get('schema_version')!=1: raise ArtifactError('Unsupported prepared-data schema')
    if 'alignment_inventory_config' in meta:
        for key in ['max_time_residual_s','excluded_image_runs']:
            if meta['alignment_inventory_config'][key]!=cfg['alignment'][key]:raise ConfigurationError(f'Inventory setting alignment.{key} changed; prepare a new artifact directory')
    if meta['preparation_config']!=cfg['data']:
        # Root/artifact paths may change on a training machine; processing settings may not.
        a={k:v for k,v in meta['preparation_config'].items() if k not in ['root','artifacts']}
        b={k:v for k,v in cfg['data'].items() if k not in ['root','artifacts']}
        if a!=b: raise ArtifactError('Prepared data settings differ from configuration. Choose a new artifact directory and prepare again.')
    if verify:
        for name,digest in meta['artifacts'].items():
            if file_hash(root/name)!=digest: raise DataIntegrityError(f'Prepared artifact hash mismatch: {name}')
    return root,meta

def labels(latitude,longitude,positive_radius,buffer_radius):
    lat=np.deg2rad(latitude)[:,None]; lon=np.deg2rad(longitude)[:,None]
    sites=pd.DataFrame(GROUND_TRUTH_ANOMALIES)
    slat=np.deg2rad(sites.lat.to_numpy()); slon=np.deg2rad(sites.lon.to_numpy())
    hav=np.sin((lat-slat)/2)**2+np.cos(lat)*np.cos(slat)*np.sin((lon-slon)/2)**2
    distances=6371000*2*np.arcsin(np.sqrt(np.clip(hav,0,1)))
    nearest=distances.argmin(1); distance=distances[np.arange(len(lat)),nearest]
    y=np.where(distance<=positive_radius,1,np.where(distance<=buffer_radius,-1,0))
    return y,sites.id.to_numpy()[nearest],distance

def count_rows(path):
    with path.open('rb') as f:
        lines=0; last=b''
        for chunk in iter(lambda:f.read(2**20),b''):
            lines+=chunk.count(b'\n'); last=chunk[-1:]
    return lines+(1 if last and last!=b'\n' else 0)-1

def prepare(cfg):
    root=prepared_root(cfg); data=Path(cfg['data']['root']); settings=cfg['data']
    if root.exists():
        raise ArtifactError(f'Artifact directory already exists: {root}. Existing data is never overwritten; use a new data.artifacts path or inspect with doctor.')
    runs=sorted(p for p in data.iterdir() if p.is_dir() and RUN_RE.fullmatch(p.name)) if data.is_dir() else []
    if not runs: raise DataIntegrityError(f'No Rail-VIVID run folders found at {data}')
    if settings['reference_run'] not in [p.name for p in runs]: raise ConfigurationError('Reference run is unavailable')
    needed=sum((p/(p.name+'.csv')).stat().st_size for p in runs)*.3
    ancestor=root.parent
    while not ancestor.exists(): ancestor=ancestor.parent
    if shutil.disk_usage(ancestor).free<needed+100_000_000: raise ArtifactError('Insufficient free storage for prepared arrays')
    root.mkdir(parents=True,exist_ok=False)
    write_json(root/'INCOMPLETE.json',{'started':utc_now(),'config':settings})
    run_records=[]; hashes={}; expected_ns=round(1e9/settings['sampling_hz'])
    try:
        for run in runs:
            source=run/(run.name+'.csv')
            if not source.is_file(): raise DataIntegrityError(f'Missing run CSV: {source}')
            n=count_rows(source)
            if n<settings['window_samples']: raise DataIntegrityError(f'{run.name}: recording too short ({n} rows)')
            target=root/'runs'/run.name; target.mkdir(parents=True)
            acc=open_memmap(target/'acceleration.npy',mode='w+',dtype='float32',shape=(n,6))
            gps=open_memmap(target/'gps.npy',mode='w+',dtype='float64',shape=(n,2))
            times=open_memmap(target/'timestamps.npy',mode='w+',dtype='int64',shape=(n,))
            offset=0; previous=None
            for chunk in pd.read_csv(source,chunksize=settings['chunk_rows']):
                require_columns(chunk,['Timestamp','Latitude','Longitude']+CHANNELS,source)
                t=pd.to_datetime(chunk.Timestamp,format='mixed',errors='raise').astype('int64').to_numpy()
                x=chunk[CHANNELS].to_numpy(dtype='float32'); g=chunk[['Latitude','Longitude']].to_numpy(dtype='float64')
                finite(x,source); finite(g,source)
                if not ((abs(g[:,0])<=90)&(abs(g[:,1])<=180)).all(): raise DataIntegrityError(f'{source}: invalid latitude/longitude')
                dt=np.diff(np.r_[previous,t]) if previous is not None else np.diff(t)
                if (abs(dt-expected_ns)>1000).any(): raise DataIntegrityError(f'{source}: irregular, duplicate or nonmonotonic timestamps')
                end=offset+len(chunk)
                if end>n: raise DataIntegrityError(f'{source}: unexpected quoted/multiline CSV records')
                acc[offset:end]=x;gps[offset:end]=g;times[offset:end]=t;offset=end;previous=t[-1]
            if offset!=n: raise DataIntegrityError(f'{source}: row count changed during read')
            acc.flush();gps.flush();times.flush()
            match=RUN_RE.fullmatch(run.name)
            run_records.append({'run_id':run.name,'rows':n,'direction':match[1],'target_speed_kmh':float(match[2]),
                'run_number':int(match[3]),'start_ns':int(times[0]),'end_ns':int(times[-1]),'source_sha256':file_hash(source)})
            del acc,gps,times
            LOG.info('Prepared %s: %s rows',run.name,n)
        refgps=np.load(root/'runs'/settings['reference_run']/'gps.npy',mmap_mode='r')[::settings['reference_step_samples']]
        if settings['reference_run'].startswith('BtoA'): refgps=refgps[::-1]
        reference=TrackReference(refgps[:,0],refgps[:,1])
        write_json(root/'reference.json',{'origin':reference.origin,'line_xy':reference.line.tolist(),'source_run':settings['reference_run'],
            'length_m':float(reference.chain[-1]),'status':'processed-GPS diagnostic, not surveyed absolute accuracy'})
        # Published anomaly coordinates are targets/split metadata, never model features.
        sites=pd.DataFrame(GROUND_TRUTH_ANOMALIES)
        sites['chainage_m'],sites['reference_distance_m']=reference.project(sites.lat,sites.lon)
        write_csv(root/'sites.csv',sites)
        windows=[];frames=[]
        for record in run_records:
            run_id=record['run_id']; folder=root/'runs'/run_id
            gps=np.load(folder/'gps.npy',mmap_mode='r'); times=np.load(folder/'timestamps.npy',mmap_mode='r')
            position=open_memmap(folder/'position.npy',mode='w+',dtype='float64',shape=(len(gps),2))
            for start in range(0,len(gps),settings['chunk_rows']):
                stop=min(start+settings['chunk_rows'],len(gps)); s,d=reference.project(gps[start:stop,0],gps[start:stop,1]);position[start:stop]=np.c_[s,d]
            position.flush()
            starts=np.arange(0,len(gps)-settings['window_samples']+1,settings['stride_samples']); ends=starts+settings['window_samples']; mid=starts+settings['window_samples']//2
            y,nearest,distance=labels(gps[mid,0],gps[mid,1],settings['positive_radius_m'],settings['buffer_radius_m'])
            support_min=np.asarray([position[a:b,0].min() for a,b in zip(starts,ends)])
            support_max=np.asarray([position[a:b,0].max() for a,b in zip(starts,ends)])
            support_error=np.asarray([position[a:b,1].max() for a,b in zip(starts,ends)])
            w=pd.DataFrame({'window_id':[f'{run_id}:{a}:{b}' for a,b in zip(starts,ends)],'run_id':run_id,'start_idx':starts,'end_idx':ends,
                'timestamp_ns':times[mid],'latitude':gps[mid,0],'longitude':gps[mid,1],'chainage_m':position[mid,0],
                'support_min_m':support_min,'support_max_m':support_max,'reference_distance_m':support_error,
                'label':y,'site_id':nearest,'site_distance_m':distance,'direction':record['direction'],'target_speed_kmh':record['target_speed_kmh']})
            windows.append(w)
            paths=sorted((data/run_id).glob('Frames*/*.jpg'))
            if paths:
                ft=pd.to_datetime([p.stem for p in paths],format='%Y-%m-%d_%H-%M-%S.%f',errors='raise').asi8
                j=nearest_indices(times,ft); residual=abs(times[j]-ft)/1e9
                for i,p in enumerate(paths):
                    rejected='known_wrong_run_images' if run_id in cfg['alignment']['excluded_image_runs'] else (
                        'outside_sensor_interval' if ft[i]<times[0] or ft[i]>times[-1] else (
                        'time_residual_exceeded' if residual[i]>cfg['alignment']['max_time_residual_s'] else ''))
                    frames.append({'frame_id':f'{run_id}/{p.parent.name}/{p.name}','run_id':run_id,'relative_path':str(p.relative_to(data)).replace('\\','/'),
                        'timestamp_ns':int(ft[i]),'nominal_chainage_m':float(position[j[i],0]),'chainage_m':float(position[j[i],0]),
                        'time_residual_s':float(residual[i]),'reference_distance_m':float(position[j[i],1]),'bytes':p.stat().st_size,
                        'rejection_reason':rejected,'alignment_status':'nominal_unvalidated'})
            del position,gps,times
        frame=pd.DataFrame(frames)
        if not frame.empty:
            accepted=frame.rejection_reason.eq('')
            names=frame.relative_path.map(lambda x:Path(x).name)
            duplicate=names[accepted].duplicated(keep=False)
            if duplicate.any():
                bad_indices=duplicate.index[duplicate];frame.loc[bad_indices,'rejection_reason']='cross_run_duplicate_filename'
        write_csv(root/'windows.csv',pd.concat(windows,ignore_index=True));write_csv(root/'frames.csv',frame);write_csv(root/'runs.csv',pd.DataFrame(run_records))
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.name!='INCOMPLETE.json': hashes[str(p.relative_to(root)).replace('\\','/')]=file_hash(p)
        prep_config={k:v for k,v in settings.items()}
        meta={'schema_version':1,'created':utc_now(),'preparation_config':prep_config,'alignment_inventory_config':cfg['alignment'],
            'artifacts':hashes,'data_fingerprint':fingerprint({'raw':run_records,'artifacts':hashes}),
            'label_semantics':'published coordinate proximity; normals not independently adjudicated','physical_alignment_validated':False}
        write_json(root/'READY.json',meta);(root/'INCOMPLETE.json').unlink()
        return meta
    except (OSError,ValueError,KeyError,DataIntegrityError) as e:
        write_json(root/'FAILURE.json',{'time':utc_now(),'error':str(e),'type':type(e).__name__})
        if isinstance(e,DataIntegrityError): raise
        raise DataIntegrityError(f'Preparation failed; partial artifacts retained at {root}: {e}') from e
