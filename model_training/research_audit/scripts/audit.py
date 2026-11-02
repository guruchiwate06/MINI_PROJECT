"""Read-only dataset audit. Outputs are confined to research_audit/evidence.
Run: python research_audit/scripts/audit.py --data D:/Rail-VIVID
No training or modifications of source datasets/benchmarks.
"""
from pathlib import Path
import argparse, json, sys, hashlib, platform, shutil, subprocess, importlib.metadata
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.signal import welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent if ROOT.name == 'model_training' else ROOT / 'mini_project_rajguru'
OUT = ROOT / 'research_audit' / 'evidence'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO))
from src.evaluation.splitters import (GROUND_TRUTH_ANOMALIES, assign_spatial_labels,
    RandomWindowSplitter, BlockGroupedWindowSplitter, RunHeldOutSplitter,
    SpatialHeldOutSplitter, CombinedRunAndSpatialSplitter)

def dump(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def nearest(a,b):
    j=np.searchsorted(a,b).clip(0,len(a)-1)
    k=(j-1).clip(0,len(a)-1)
    return np.where(abs(a[k]-b)<abs(a[j]-b),k,j)

def main(data):
    cache=data/'data/cached_window_features.csv'
    df=pd.read_csv(cache)
    dump('provenance.json',{'date':'2026-11-02','data_root':str(data), 'cache_sha256':sha(cache),
        'code_commit':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
        'files':{str(p.relative_to(REPO)):sha(p) for p in REPO.rglob('*.py')}})
    relabel=assign_spatial_labels(df)
    dump('cache_validation.json',{'rows':len(df),'label_counts':df.label.value_counts().to_dict(),
        'recomputed_label_mismatches':int((df.label!=relabel.label).sum()),
        'recomputed_site_mismatches':int((df.nearest_anomaly_id!=relabel.nearest_anomaly_id).sum()),
        'duplicate_run_start':int(df.duplicated(['run_id','start_idx']).sum()),
        'window_lengths':(df.end_idx-df.start_idx).value_counts().to_dict(),
        'feature_nonfinite':int((~np.isfinite(df.filter(regex='^Channel').values)).sum())})
    df.groupby(['run_id','label']).size().rename('windows').to_csv(OUT/'windows_by_run_label.csv')
    df[df.label==1].groupby(['nearest_anomaly_id','run_id']).size().rename('positive_windows').to_csv(OUT/'site_run_counts.csv')
    pd.DataFrame(GROUND_TRUTH_ANOMALIES).to_csv(OUT/'anomaly_sites.csv',index=False)
    splits={'random':RandomWindowSplitter(), 'block':BlockGroupedWindowSplitter(),
        'run':RunHeldOutSplitter(),'spatial':SpatialHeldOutSplitter(),'combined':CombinedRunAndSpatialSplitter()}
    counts=[]; overlap=[]
    for name,sp in splits.items():
        parts=sp.split(df); ids={}
        for part,p in zip(['train','validation','test'],parts):
            ids[part]=p.index.tolist()
            counts.append(dict(protocol=name,partition=part,windows=len(p),positive=int((p.label==1).sum()),
                negative=int((p.label==0).sum()),runs=p.run_id.nunique(),sites=','.join(map(str,sorted(p.loc[p.label==1,'nearest_anomaly_id'].unique())))))
        dump('split_indices_'+name+'.json',ids)
        for ia,ib in [(0,1),(0,2),(1,2)]:
            a,b=parts[ia],parts[ib]; n=0; unique_a=set(); unique_b=set()
            for run,g in a.groupby('run_id'):
                q=b[b.run_id==run].sort_values('start_idx')
                if q.empty: continue
                starts=q.start_idx.to_numpy(); ends=q.end_idx.to_numpy(); qi=q.index.to_numpy()
                for ix,row in g.iterrows():
                    lo=np.searchsorted(ends,row.start_idx,side='right')
                    hi=np.searchsorted(starts,row.end_idx,side='left')
                    n+=max(0,hi-lo)
                    if hi>lo: unique_a.add(int(ix)); unique_b.update(map(int,qi[lo:hi]))
            overlap.append(dict(protocol=name,partition_a=['train','validation','test'][ia],partition_b=['train','validation','test'][ib],
                overlapping_window_pairs=n,affected_a=len(unique_a),affected_b=len(unique_b),
                shared_runs=len(set(a.run_id)&set(b.run_id)),shared_positive_sites=len(set(a.loc[a.label==1,'nearest_anomaly_id'])&set(b.loc[b.label==1,'nearest_anomaly_id']))))
    pd.DataFrame(counts).to_csv(OUT/'split_counts.csv',index=False)
    pd.DataFrame(overlap).to_csv(OUT/'split_overlap.csv',index=False)
    inventory=[]; channels=[]; pairs=[]; decode=[]; gps=[]; hashes=[]
    fig,axs=plt.subplots(3,1,figsize=(12,9))
    panels=[]
    for run in sorted(data.glob('*to*_*_*')):
        path=run/(run.name+'.csv')
        if not path.exists(): continue
        print('Scanning',run.name,flush=True)
        raw=pd.read_csv(path)
        ts=pd.to_datetime(raw.Timestamp,format='mixed'); t=ts.astype('int64').to_numpy()/1e9
        dt=np.diff(t)
        x=raw[[f'Channel{i}' for i in range(1,7)]].to_numpy()
        lat=raw.Latitude.to_numpy(); lon=raw.Longitude.to_numpy()
        # Canonical approximate along-track coordinate from label-independent endpoints.
        # Straight chord is diagnostic only; do not use as a surveyed route topology.
        xy=np.column_stack([(lon+77.89094)*111320*np.cos(np.deg2rad(40.234)),(lat-40.23608)*111320])
        axis=np.array([(-77.87387+77.89094)*111320*np.cos(np.deg2rad(40.234)),(40.23184-40.23608)*111320]); axis/=np.linalg.norm(axis)
        s=xy@axis; cross=xy@np.array([-axis[1],axis[0]])
        step=np.diff(s); gps.append(dict(run_id=run.name,chord_span_m=float(np.ptp(s)),
            lateral_p05_m=float(np.percentile(cross,5)),lateral_p95_m=float(np.percentile(cross,95)),
            unchanged_gps_fraction=float(np.mean((np.diff(lat)==0)&(np.diff(lon)==0))),
            backward_steps=int((step < -0.01).sum()) if run.name.startswith('AtoB') else int((step>0.01).sum()),
            mean_chord_speed_kmh=float(abs(s[-1]-s[0])/(t[-1]-t[0])*3.6)))
        frames=sorted(run.glob('Frames*/*.jpg'))
        ft=pd.to_datetime([f.stem for f in frames],format='%Y-%m-%d_%H-%M-%S.%f',errors='coerce')
        good=~ft.isna(); fv=ft[good].astype('int64').to_numpy()/1e9
        fi=np.flatnonzero(good); order=np.argsort(fv); fv=fv[order]; fi=fi[order]
        fd=np.diff(fv); idx=nearest(t,fv); residual=abs(t[idx]-fv)
        valid=(fv>=t[0])&(fv<=t[-1])&(residual<=0.001)&(idx>=500)&(idx+500<=len(raw))
        # All filename timestamps attempted. Physical correspondence remains unverified.
        frame_meta=pd.DataFrame({'latitude':lat[idx],'longitude':lon[idx]})
        labels=assign_spatial_labels(frame_meta)
        selected=set(np.linspace(0,len(frames)-1,5,dtype=int).tolist())
        for site in range(1,10):
            candidates=np.where(valid & (labels.nearest_anomaly_id.to_numpy()==site))[0]
            if len(candidates):
                k=candidates[np.argmin(labels.dist_to_nearest_anomaly_m.to_numpy()[candidates])]
                selected.add(int(fi[k]))
        for j in sorted(selected):
            f=frames[j]
            try:
                with Image.open(f) as im:
                    im.load(); w,h=im.size
                    gray=np.array(im.convert('L').resize((320,180)))
                    decode.append(dict(run_id=run.name,path=str(f),width=w,height=h,mean_gray=float(gray.mean()),std_gray=float(gray.std()),decoded=True))
                    if run.name in ['AtoB_20_1','BtoA_25_1']:
                        thumb=im.convert('RGB'); thumb.thumbnail((480,270)); panels.append((run.name+' '+f.stem,thumb.copy()))
            except Exception as e: decode.append(dict(run_id=run.name,path=str(f),decoded=False,error=str(e)))
        for site in range(0,10):
            m=valid if site==0 else valid&(labels.nearest_anomaly_id.to_numpy()==site)&(labels.label.to_numpy()==1)
            pairs.append(dict(run_id=run.name,site_id=site,attempted=len(frames) if site==0 else int(((labels.nearest_anomaly_id==site)&(labels.label==1)).sum()),
                nominal_pairs=int(m.sum()),physically_validated_pairs=0))
        # Concrete raw pair manifest: 1 normal and nearest available sample per site/run.
        chosen=[]
        normal=np.flatnonzero(valid&(labels.label.to_numpy()==0))
        if len(normal): chosen.append(int(normal[len(normal)//2]))
        for site in range(1,10):
            q=np.flatnonzero(valid&(labels.nearest_anomaly_id.to_numpy()==site)&(labels.label.to_numpy()==1))
            if len(q): chosen.append(int(q[np.argmin(labels.dist_to_nearest_anomaly_m.to_numpy()[q])]))
        for k in chosen:
            hashes.append(dict(run_id=run.name,image=str(frames[fi[k]]),sensor_csv=str(path),start_idx=int(idx[k]-500),end_idx=int(idx[k]+500),
                nominal_time_residual_s=float(residual[k]),nominal_latitude=float(lat[idx[k]]),nominal_longitude=float(lon[idx[k]]),
                nearest_site=int(labels.nearest_anomaly_id.iloc[k]),proximity_label=int(labels.label.iloc[k]),
                physical_alignment='unverified; upstream reports up to ~15 m drift'))
        inventory.append(dict(run_id=run.name,rows=len(raw),csv_bytes=path.stat().st_size,images=len(frames),image_bytes=sum(f.stat().st_size for f in frames),
            columns=';'.join(raw.columns),missing_values=int(raw.isna().sum().sum()),nonfinite_channels=int((~np.isfinite(x)).sum()),
            start=str(ts.iloc[0]),end=str(ts.iloc[-1]),duration_s=float(t[-1]-t[0]),median_dt_s=float(np.median(dt)),
            dt_min_s=float(dt.min()),dt_max_s=float(dt.max()),dt_nonpositive=int((dt<=0).sum()),dt_off_2khz=int((abs(dt-.0005)>2e-6).sum()),
            frame_timestamp_parse_fail=int((~good).sum()),mean_recorded_fps=float((len(fv)-1)/(fv[-1]-fv[0])),
            frame_dt_p05=float(np.percentile(fd,5)),frame_dt_p50=float(np.median(fd)),frame_dt_p95=float(np.percentile(fd,95)),frame_dt_max=float(fd.max()),
            frame_outside_sensor_time=int(((fv<t[0])|(fv>t[-1])).sum()),nominal_valid_pairs=int(valid.sum()),
            target_speed_kmh=float(run.name.split('_')[1]),direction=run.name.split('_')[0]))
        for ch in range(6):
            f,p=welch(x[:min(len(x),120000),ch],fs=2000,nperseg=4096)
            channels.append(dict(run_id=run.name,channel=ch+1,mean=float(x[:,ch].mean()),std=float(x[:,ch].std()),min=float(x[:,ch].min()),max=float(x[:,ch].max()),
                abs_gt_10=int((abs(x[:,ch])>=10).sum()),consecutive_equal=int((np.diff(x[:,ch])==0).sum()),peak_psd_hz=float(f[np.argmax(p)]),
                power_above_500_fraction=float(p[f>500].sum()/p.sum())))
        if run.name=='AtoB_20_1':
            k=slice(0,10000,10)
            for ch in range(6): axs[0].plot(t[k]-t[0],x[k,ch],label=f'Ch{ch+1}',lw=.5)
            stride=200
            axs[1].plot(s[::stride],x[::stride,0],lw=.5)
            axs[2].semilogy(f,p)
            # Save real waveform batch for smoke test only, no training.
            np.savez_compressed(OUT/'raw_smoke_windows.npz',waveforms=np.stack([x[10000:11000].T,x[20000:21000].T]).astype('float32'))
    axs[0].set(xlabel='Elapsed seconds',ylabel='Published acceleration units: g',title='AtoB_20_1: six channels, first 5 s'); axs[0].legend(ncol=6)
    axs[1].set(xlabel='Approximate canonical chord distance (m); not surveyed chainage',ylabel='Channel1 (g)',title='Full-run spatial diagnostic')
    axs[2].set(xlabel='Frequency (Hz)',ylabel='Welch PSD',title='Channel6, first 60 s')
    fig.tight_layout(); fig.savefig(OUT/'vibration_diagnostics.png',dpi=150); plt.close(fig)
    pd.DataFrame(inventory).to_csv(OUT/'run_inventory.csv',index=False)
    pd.DataFrame(channels).to_csv(OUT/'channel_statistics.csv',index=False)
    pd.DataFrame(pairs).to_csv(OUT/'pair_counts.csv',index=False)
    pd.DataFrame(decode).to_csv(OUT/'image_decode_sample.csv',index=False)
    pd.DataFrame(gps).to_csv(OUT/'gps_diagnostics.csv',index=False)
    pd.DataFrame(hashes).to_csv(OUT/'nominal_pair_manifest.csv',index=False)
    if panels:
        canvas=Image.new('RGB',(1440,310*((len(panels)+2)//3)), 'white'); draw=ImageDraw.Draw(canvas)
        for i,(caption,im) in enumerate(panels):
            xx=(i%3)*480; yy=(i//3)*310; canvas.paste(im,(xx,yy+32)); draw.text((xx+5,yy+5),caption,fill='black')
        canvas.save(OUT/'frame_contact_sheet.jpg')
    dump('environment.json',{'python':sys.version,'platform':platform.platform(),'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','scipy','torch','Pillow']},
        'disk_free_bytes':{d:shutil.disk_usage(d).free for d in ['C:/','D:/']}})
    print('DONE: evidence written to',OUT,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--data',type=Path,default=Path('D:/Rail-VIVID'))
    main(parser.parse_args().data)
