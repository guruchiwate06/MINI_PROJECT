from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.io import loadmat
import h5py
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'research_audit/evidence'; DATA=Path('D:/Rail-VIVID')
REPO = ROOT.parent if ROOT.name == 'model_training' else ROOT / 'mini_project_rajguru'
sys.path.insert(0,str(REPO))
from src.evaluation.splitters import BlockGroupedWindowSplitter,SpatialHeldOutSplitter
from interval_guards import purge_training_and_validation,overlap_mask

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def xy(lat,lon): return np.column_stack([(np.asarray(lon)+77.89094)*111320*np.cos(np.deg2rad(40.234)),(np.asarray(lat)-40.23608)*111320])

def main():
    # Compare all frame filename sets, then sampled content hashes for shared names.
    byrun={r.name:{p.name:p for p in r.glob('Frames*/*.jpg')} for r in DATA.glob('*to*_*_*') if r.is_dir()}
    dup=[]; names=sorted(byrun)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            common=sorted(byrun[a].keys()&byrun[b].keys())
            if common:
                samples=common[::max(1,len(common)//10)]
                dup.append({'run_a':a,'run_b':b,'shared_filename_count':len(common),'sampled_content_hashes':len(samples),
                    'identical_sampled_sha256':sum(digest(byrun[a][n])==digest(byrun[b][n]) for n in samples)})
    (OUT/'cross_run_image_duplicates.json').write_text(json.dumps(dup,indent=2))
    d=pd.read_csv(DATA/'data/cached_window_features.csv'); pair_records=[]
    for run,g in d.groupby('run_id'):
        frames=sorted(byrun[run].values())
        ft=pd.to_datetime([p.stem for p in frames],format='%Y-%m-%d_%H-%M-%S.%f').astype('int64').to_numpy()/1e9
        wt=pd.to_datetime(g.timestamp,format='mixed').astype('int64').to_numpy()/1e9
        j=np.searchsorted(ft,wt).clip(0,len(ft)-1); k=(j-1).clip(0,len(ft)-1)
        j=np.where(abs(ft[k]-wt)<abs(ft[j]-wt),k,j); residual=abs(ft[j]-wt)
        for ix,frame,dt in zip(g.index,j,residual):
            pair_records.append({'cache_row':ix,'run_id':run,'frame':str(frames[frame]),'nominal_residual_s':dt,
                'nominal_within_100ms':bool(dt<=.1),'physical_alignment_validated':False})
    pairs=pd.DataFrame(pair_records); pairs.to_csv(OUT/'window_pair_manifest.csv',index=False)
    split_counts=[]; purged=[]
    for protocol in ['random','block','run','spatial','combined']:
        ids=json.loads((OUT/f'split_indices_{protocol}.json').read_text())
        for part,indices in ids.items():
            p=pairs[pairs.cache_row.isin(indices)]; good=p.loc[p.nominal_within_100ms]
            split_counts.append({'protocol':protocol,'partition':part,'windows':len(p),'nominal_pairs':len(good),
                'positive_nominal_pairs':int(d.loc[good.cache_row,'label'].eq(1).sum()),'unique_images':good.frame.nunique(),
                'physically_validated_pairs':0})
    pd.DataFrame(split_counts).to_csv(OUT/'window_pair_split_counts.csv',index=False)
    for name,sp in [('block',BlockGroupedWindowSplitter()),('spatial',SpatialHeldOutSplitter())]:
        old=sp.split(d); parts=purge_training_and_validation(*old)
        for part,prev,p in zip(['train','validation','test'],old,parts):
            purged.append({'protocol':name,'partition':part,'before':len(prev),'after':len(p),'dropped':len(prev)-len(p),'positive_after':int(p.label.eq(1).sum())})
        for i,j in [(0,1),(0,2),(1,2)]: assert not overlap_mask(parts[i],parts[j]).any()
        (OUT/f'purged_{name}_indices.json').write_text(json.dumps({k:x.index.tolist() for k,x in zip(['train','validation','test'],parts)}))
    pd.DataFrame(purged).to_csv(OUT/'purged_split_counts.csv',index=False)
    # Frozen, label-independent centerline from one existing training run.
    raw=pd.read_csv(DATA/'AtoB_20_1/AtoB_20_1.csv',usecols=['Latitude','Longitude']).iloc[::500]
    line=xy(raw.Latitude,raw.Longitude); good=np.r_[True,np.linalg.norm(np.diff(line,axis=0),axis=1)>.01]; line=line[good]
    vectors=np.diff(line,axis=0); lengths=np.linalg.norm(vectors,axis=1); chain=np.r_[0,np.cumsum(lengths)]
    tree=cKDTree((line[:-1]+line[1:])/2); q=xy(d.latitude,d.longitude); _,indices=tree.query(q,k=3)
    start=line[indices]; v=vectors[indices]; u=np.clip(((q[:,None,:]-start)*v).sum(-1)/(v*v).sum(-1),0,1)
    projected=start+u[:,:,None]*v; distances=np.linalg.norm(projected-q[:,None,:],axis=-1)
    best=distances.argmin(1); ii=indices[np.arange(len(q)),best]; uu=u[np.arange(len(q)),best]
    d['canonical_chainage_m']=chain[ii]+uu*lengths[ii]; d['distance_to_reference_m']=distances[np.arange(len(q)),best]
    for size in [5,7.62,10,15,25]: d[f'segment_{size:g}m']=np.floor(d.canonical_chainage_m/size).astype(int)
    d[['run_id','start_idx','label','canonical_chainage_m','distance_to_reference_m']+[c for c in d if c.startswith('segment_')]].to_csv(OUT/'canonical_segments_diagnostic.csv',index=False)
    d.groupby('run_id').agg(windows=('run_id','size'),reference_distance_median=('distance_to_reference_m','median'),reference_distance_p95=('distance_to_reference_m',lambda x:x.quantile(.95)),reference_distance_max=('distance_to_reference_m','max')).to_csv(OUT/'cross_run_route_repeatability.csv')
    (OUT/'centerline_diagnostic.json').write_text(json.dumps({'source':'AtoB_20_1 GPS every 500 samples, remove <1cm duplicates','length_m':chain[-1],
        'vertices':len(line),'absolute_accuracy':'not independently validated','camera_alignment':'not corrected'}))
    # Read v7.3 and conventional MAT data, preserving native variable names.
    samples=[]
    for vehicle in ['LRV4306','LRV4313']:
        for kind in ['accelerometer_data','gps_data']:
            f=next((Path('D:/DR_Train')/vehicle/kind).glob('*.mat'))
            try:
                content=loadmat(f,squeeze_me=True); arrays={k:np.asarray(v) for k,v in content.items() if not k.startswith('__')}
            except NotImplementedError:
                with h5py.File(f) as h: arrays={k:np.asarray(h[k]) for k in h if isinstance(h[k],h5py.Dataset)}
            for k,a in arrays.items():
                if not np.issubdtype(a.dtype,np.number): continue
                samples.append({'vehicle':vehicle,'kind':kind,'file':str(f),'variable':k,'shape':list(a.shape),'finite':bool(np.isfinite(a).all()),'min':float(a.min()),'max':float(a.max())})
    (OUT/'dr_train_decoded_records.json').write_text(json.dumps(samples,indent=2))
    print('deep checks done',dup,flush=True)

if __name__=='__main__': main()
