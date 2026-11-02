"""Reproduce existing RF configurations without overwriting prior outputs."""
from pathlib import Path
import sys,json,subprocess,hashlib
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT.parent if ROOT.name == 'model_training' else ROOT/'mini_project_rajguru'
OUT=ROOT/'research_audit/evidence'
sys.path.insert(0,str(REPO))
from run_benchmarks import get_feature_matrix,evaluate_predictions
from src.models.statistical_baselines import SupervisedRandomForest,UnsupervisedIsolationForest
from src.models.autoencoder import SignalAutoencoderDetector
from src.models.cnn1d import Vibration1DCNN
from src.evaluation.splitters import RandomWindowSplitter,BlockGroupedWindowSplitter,RunHeldOutSplitter,SpatialHeldOutSplitter,CombinedRunAndSpatialSplitter
from src.features.extractor import compute_window_features

def main():
    df=pd.read_csv('D:/Rail-VIVID/data/cached_window_features.csv')
    historical=pd.read_json(REPO/'reports/baseline_results.json')
    rows=[]
    protocols=[('Random_Window (Leaky Control)',RandomWindowSplitter()),('Block_Grouped_Window (Overlap-Free)',BlockGroupedWindowSplitter()),
        ('Run_Held_Out',RunHeldOutSplitter()),('Spatial_Held_Out',SpatialHeldOutSplitter()),('Combined_Run_Spatial',CombinedRunAndSpatialSplitter())]
    for name,splitter in protocols:
        tr,va,te=splitter.split(df)
        for gps in [False,True]:
            mod='Vibration_Plus_GPS' if gps else 'Vibration_Only'
            xt,yt=get_feature_matrix(tr,include_gps=gps); xv,yv=get_feature_matrix(va,include_gps=gps); xe,ye=get_feature_matrix(te,include_gps=gps)
            rf=SupervisedRandomForest(n_estimators=100,max_depth=10,seed=42).fit(xt,yt)
            threshold=rf.calibrate_threshold(xv,yv); scores=rf.predict_proba(xe); pred=rf.predict(xe)
            metrics=evaluate_predictions(ye,pred,scores,threshold,'Validation Set (Max F1)')
            old=historical[(historical.model=='Random Forest')&(historical.modalities==mod)&(historical.split_strategy==name)].iloc[0]
            row={'model':'Random Forest','split':name,'modality':mod,**metrics}
            row['max_absolute_metric_difference']=max(abs(float(metrics[m])-float(old[m])) for m in ['precision','recall','f1','roc_auc','pr_auc','calibrated_threshold'])
            rows.append(row)
            pd.DataFrame({'cache_row':te.index,'run_id':te.run_id,'start_idx':te.start_idx,'label':ye,'score':scores,'prediction':pred}).to_csv(OUT/('rf_predictions_'+name.split(' ')[0]+'_'+mod+'.csv'),index=False)
            print(name,mod,metrics['f1'],metrics['pr_auc'],'delta',row['max_absolute_metric_difference'],flush=True)
    # Only the combined unsupervised comparison; same saved configuration.
    tr,va,te=CombinedRunAndSpatialSplitter().split(df)
    xt,yt=get_feature_matrix(tr); xv,yv=get_feature_matrix(va); xe,ye=get_feature_matrix(te)
    for model,detector in [('Isolation Forest',UnsupervisedIsolationForest(n_estimators=100,seed=42)),('Signal Autoencoder',SignalAutoencoderDetector(latent_dim=12,epochs=12,batch_size=64,seed=42))]:
        detector.fit(xt,yt); threshold=detector.calibrate_threshold(xv,yv); scores=detector.score_samples(xe); pred=detector.predict(xe)
        metrics=evaluate_predictions(ye,pred,scores,threshold,'Validation Set (Tuned)')
        old=historical[(historical.model==model)&(historical.modalities=='Vibration_Only')&(historical.split_strategy=='Combined_Run_Spatial')].iloc[0]
        rows.append({'model':model,'split':'Combined_Run_Spatial','modality':'Vibration_Only',**metrics,
            'max_absolute_metric_difference':max(abs(float(metrics[m])-float(old[m])) for m in ['precision','recall','f1','roc_auc','pr_auc','calibrated_threshold'])})
        pd.DataFrame({'cache_row':te.index,'label':ye,'score':scores,'prediction':pred}).to_csv(OUT/(model.replace(' ','_')+'_combined_predictions.csv'),index=False)
    (OUT/'benchmark_reproduction.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    pd.DataFrame(rows).to_csv(OUT/'benchmark_reproduction.csv',index=False)
    # Validate cache values against actual raw records, not just cache labels.
    checks=[]
    for run in ['AtoB_20_1','AtoB_22.5_2','BtoA_25_1','BtoA_27.5_4']:
        raw=pd.read_csv(Path('D:/Rail-VIVID')/run/(run+'.csv'),nrows=3000)
        for start in [0,1500]:
            cached=df[(df.run_id==run)&(df.start_idx==start)].iloc[0]
            diffs=[]
            for i in range(1,7):
                feat=compute_window_features(raw[f'Channel{i}'].to_numpy()[start:start+1000])
                diffs.extend(abs(feat[k]-cached[f'Channel{i}_{k}']) for k in feat)
            mid=raw.iloc[start+500]
            checks.append({'run_id':run,'start_idx':start,'max_absolute_feature_difference':max(diffs),
                'timestamp_match':str(cached.timestamp)==str(mid.Timestamp),
                'latitude_difference':abs(cached.latitude-mid.Latitude),'longitude_difference':abs(cached.longitude-mid.Longitude)})
    pd.DataFrame(checks).to_csv(OUT/'raw_cache_spotchecks.csv',index=False)
    env={'torch':torch.__version__,'torch_cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
        'gpu_driver':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip()}
    torch.manual_seed(42)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    raw=pd.read_csv('D:/Rail-VIVID/AtoB_20_1/AtoB_20_1.csv',nrows=2000)
    x=torch.tensor(raw[[f'Channel{i}' for i in range(1,7)]].to_numpy().reshape(2,1000,6).transpose(0,2,1),dtype=torch.float32,device=device)
    model=Vibration1DCNN().to(device).eval()
    with torch.no_grad(): logits=model(x)
    env['raw_cnn_forward']={'device':device,'input_shape':list(x.shape),'output_shape':list(logits.shape),
        'finite':bool(torch.isfinite(logits).all()),'parameters':sum(p.numel() for p in model.parameters())}
    env['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated() if device=='cuda' else None
    (OUT/'torch_smoke.json').write_text(json.dumps(env,indent=2),encoding='utf-8')
    # Existing DR-Train script writes stdout only; preserve its outputs separately.
    dr=subprocess.run([sys.executable,'D:/DR_Train/audit_scripts/13_model_comparison.py'],capture_output=True,text=True)
    (OUT/'dr_train_reproduction.txt').write_text(dr.stdout+'\nSTDERR:\n'+dr.stderr,encoding='utf-8')
    (OUT/'dr_train_reproduction_exit.txt').write_text(str(dr.returncode))
    print('DONE',flush=True)

if __name__=='__main__': main()
