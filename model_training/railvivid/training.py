from __future__ import annotations
import contextlib,json,logging,math,os,random,time,pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from torch.utils.data import DataLoader
from .common import atomic_write,environment,file_hash,fingerprint,read_json,seed_everything,select_device,utc_now,write_csv,write_json,source_hash
from .dataset import WindowDataset,VisionStore,collate_to_device,training_normalizer
from .errors import ArtifactError,ConfigurationError,DataIntegrityError,TrainingError
from .metrics import classification_metrics,fit_temperature,select_threshold
from .models import make_model,TimeFrequencySSL,ssl_loss
from .prepare import load_prepared
from .splits import load_split
from .vision import vision_path

LOG=logging.getLogger('railvivid')

def run_path(cfg):return Path(cfg['experiment']['output'])/cfg['experiment']['name']

def experiment_identity(cfg,split_info):
    config=json.loads(json.dumps(cfg));config.pop('_config_path',None)
    for section,keys in [('data',['root','artifacts']),('experiment',['output']),('training',['resume'])]:
        for k in keys:config[section].pop(k,None)
    if config['model']['initial_encoder']:
        config['model']['initial_encoder']=file_hash(config['model']['initial_encoder'])
    return fingerprint({'config':config,'split':split_info['split_fingerprint'],'source':source_hash()})

def rng_state():
    state=np.random.get_state()
    return {'python':random.getstate(),'numpy':{'kind':state[0],'keys':torch.from_numpy(state[1].astype(np.int64)),'position':state[2],'has_gaussian':state[3],'cached_gaussian':state[4]},
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(state):
    random.setstate(state['python']);n=state['numpy'];np.random.set_state((n['kind'],n['keys'].numpy().astype(np.uint32),n['position'],n['has_gaussian'],n['cached_gaussian']))
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])

def save_checkpoint(path,state):atomic_write(path,lambda f:torch.save(state,f),binary=True)

def load_checkpoint(path):
    try:return torch.load(path,map_location='cpu',weights_only=True)
    except (OSError,RuntimeError,ValueError,EOFError,pickle.UnpicklingError) as e:raise ArtifactError(f'Cannot load checkpoint {path}: {e}') from e

def make_loader(dataset,cfg,shuffle=False,epoch=0,drop_last=False):
    dataset.epoch=epoch;generator=torch.Generator().manual_seed(cfg['training']['seed']+epoch)
    return DataLoader(dataset,batch_size=cfg['training']['batch_size'],shuffle=shuffle,drop_last=drop_last,
        num_workers=cfg['training']['workers'],pin_memory=cfg['training']['device'].startswith('cuda') and torch.cuda.is_available(),
        persistent_workers=False,generator=generator,
        **({'prefetch_factor':cfg['training']['prefetch_factor']} if cfg['training']['workers'] else {}))

def accumulation_examples(dataset_size,batch_size,accumulation_steps,step,drop_last=False):
    effective=(dataset_size//batch_size)*batch_size if drop_last else dataset_size
    group_start=(step//accumulation_steps)*accumulation_steps
    return min(effective-group_start*batch_size,accumulation_steps*batch_size)

def predict(model,dataset,cfg,device):
    model.eval();scores=np.full(len(dataset),np.nan,dtype=float)
    if device.type=='cuda':torch.cuda.synchronize(device)
    start_time=time.perf_counter()
    with torch.inference_mode():
        for batch in make_loader(dataset,cfg):
            indices=batch['index'].numpy();batch=collate_to_device(batch,device)
            logits=model(batch)
            if logits.ndim!=1 or len(logits)!=len(indices) or not torch.isfinite(logits).all():raise TrainingError('Inference produced invalid logits/shape')
            scores[indices]=logits.float().cpu().numpy()
    if not np.isfinite(scores).all():raise TrainingError('Prediction coverage is incomplete')
    if device.type=='cuda':torch.cuda.synchronize(device)
    elapsed=time.perf_counter()-start_time
    result=dataset.frame.copy();result['logit']=scores
    result.attrs['inference']={'seconds_including_loader_and_transfer':elapsed,'windows':len(dataset),
        'windows_per_second':len(dataset)/max(elapsed,1e-12),'batch_size':cfg['training']['batch_size'],'device':str(device),
        'process_peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None}
    return result

def vision_for(cfg,split,corruption=None):
    if cfg['model']['kind'] not in ['vision','fusion','patch_memory']:return None
    if cfg['model']['kind'] in ['vision','patch_memory'] and cfg['split']['cohort']!='paired':raise ConfigurationError('Vision-only baselines require split.cohort=paired')
    store=VisionStore(vision_path(cfg,corruption),cfg['model']['vision_dim'])
    required=split[['window_id','partition','candidate_frame_ids']]
    missing={f for ids in required.candidate_frame_ids for f in ids}-set(store.lookup)
    if missing:raise ArtifactError(f'Vision feature cache is missing {len(missing)} required frames; no partial cohort will be silently fitted')
    used={part:{store.index.iloc[store.lookup[f]].image_sha256 for ids in required.loc[required.partition.eq(part),'candidate_frame_ids'] for f in ids if f in store.lookup} for part in ['train','validation','test']}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        if used[a]&used[b]:raise DataIntegrityError(f'Identical image contents cross {a}/{b}; quarantine duplicate frames before fitting')
    return store

def finalize_validation(cfg,folder,predictions,sites,identity,score_kind='logit'):
    temperature=fit_temperature(predictions.label,predictions.logit) if cfg['evaluation']['temperature_scaling'] else 1.
    calibration={'temperature':temperature,'source':'validation only','method':'temperature' if cfg['evaluation']['temperature_scaling'] else 'identity'}
    predictions=predictions.copy();predictions['probability']=expit(predictions.logit/temperature)
    threshold=select_threshold(predictions,sites,cfg['evaluation']);calibration['threshold']=threshold
    calibration['policy']=cfg['evaluation']['threshold'];calibration['experiment_fingerprint']=identity
    write_json(folder/'calibration.json',calibration);write_csv(folder/'validation_predictions.csv',predictions.drop(columns=['candidate_frame_ids','candidate_offsets_m'],errors='ignore'))
    write_json(folder/'validation_metrics.json',classification_metrics(predictions.label,predictions.probability,threshold,cfg['evaluation']['ece_bins']))
    return calibration

@contextlib.contextmanager
def experiment_lock(folder):
    lock=folder/'ACTIVE.lock'
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        with os.fdopen(fd,'w') as f:f.write(json.dumps({'pid':os.getpid(),'started':utc_now()}))
    except FileExistsError as e:raise ArtifactError(f'Run is active or interrupted lock remains: {lock}. Inspect its PID before recovering.') from e
    try:yield
    finally:lock.unlink(missing_ok=True)

def train(cfg,pretrain=False):
    # This gate precedes all fitting, including CPU statistical fits launched on the training box.
    device=select_device(cfg['training']['device'],cfg['training']['require_cuda'])
    root,data_info=load_prepared(cfg,verify=True);split,split_info,_=load_split(cfg)
    seed_everything(cfg['training']['seed'],cfg['training']['deterministic'])
    identity=experiment_identity(cfg,split_info);folder=run_path(cfg);resume=cfg['training']['resume']
    if folder.exists() and not resume:raise ArtifactError(f'Experiment exists: {folder}. Choose a new name or resume its checkpoint.')
    if resume:
        if not folder.is_dir():raise ArtifactError('Resume requires the original experiment directory')
        old=read_json(folder/'run.json')
        if old['experiment_fingerprint']!=identity:raise ArtifactError('Resume configuration/data/split differs from original run')
        if (folder/'DONE.json').exists():raise ArtifactError('Run already completed; use a new experiment name for another fit')
    else:
        folder.mkdir(parents=True,exist_ok=False)
        write_json(folder/'config.json',cfg);write_json(folder/'environment.json',environment())
        write_json(folder/'run.json',{'started':utc_now(),'experiment_fingerprint':identity,'split':split_info,
            'stage':'pretraining' if pretrain else 'supervised','data_fingerprint':data_info['data_fingerprint']})
    with experiment_lock(folder):
        try:
            training_rows=split.loc[split.partition.eq('train')]
            normalizer=training_normalizer(root,training_rows,cfg['model']['channels']);write_json(folder/'normalizer.json',normalizer)
            if cfg['model']['kind'] in ['rf','patch_memory'] and not pretrain:
                from .baselines import fit_baseline
                return fit_baseline(cfg,split,split_info,folder,identity)
            vision=None if pretrain else vision_for(cfg,split)
            tr=training_rows if pretrain else training_rows.loc[training_rows.eligible_supervised]
            val=split.loc[split.partition.eq('validation')]
            if pretrain and len(tr)<2:raise DataIntegrityError('At least two allowed training windows are needed for self-supervision')
            dataset=WindowDataset(root,tr,normalizer,cfg,vision,augment=True)
            val_dataset=WindowDataset(root,val,normalizer,cfg,vision)
            model=(TimeFrequencySSL(cfg) if pretrain else make_model(cfg)).to(device)
            initial=cfg['model']['initial_encoder']
            if initial and not resume and not pretrain:
                if not hasattr(model,'encoder'):raise ConfigurationError('Encoder initialization is supported for TCN-based models, not historical CNN')
                pretrained=load_checkpoint(initial)
                if pretrained.get('data_fingerprint')!=data_info['data_fingerprint'] or pretrained.get('split_fingerprint')!=split_info['split_fingerprint']:
                    raise DataIntegrityError('Pretraining checkpoint used a different permitted data/split. Never initialize from test-exposed pretraining.')
                model.encoder.load_state_dict(pretrained['encoder'],strict=True)
            optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay'])
            epochs=cfg['training']['pretrain_epochs'] if pretrain else cfg['training']['epochs']
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
            use_amp=bool(cfg['training']['amp'] and device.type=='cuda');scaler=torch.amp.GradScaler('cuda',enabled=use_amp)
            first=0;best=-math.inf;bad_epochs=0
            if resume:
                checkpoint=load_checkpoint(resume)
                if checkpoint['experiment_fingerprint']!=identity or checkpoint['pretrain']!=pretrain:raise ArtifactError('Checkpoint identity/stage mismatch')
                model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler']);scaler.load_state_dict(checkpoint['scaler'])
                first=checkpoint['epoch']+1;best=checkpoint['best'];bad_epochs=checkpoint['bad_epochs'];restore_rng(checkpoint['rng'])
            positives=int(tr.label.eq(1).sum());negatives=int(tr.label.eq(0).sum())
            pos_weight=torch.tensor(negatives/max(positives,1),device=device)
            criterion=torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            overall_start=time.perf_counter()
            epoch_durations=[]
            for epoch in range(first,epochs):
                model.train();loader=make_loader(dataset,cfg,shuffle=True,epoch=epoch,drop_last=pretrain)
                if not len(loader):raise DataIntegrityError('No batches: reduce batch_size for the permitted cohort')
                if device.type=='cuda':torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
                optimizer.zero_grad(set_to_none=True);total=0.;seen=0;start_time=time.perf_counter()
                updates=0
                log_interval=max(1, len(loader)//4)
                for step,batch in enumerate(loader):
                    batch=collate_to_device(batch,device);size=len(batch['label'])
                    examples=accumulation_examples(len(dataset),cfg['training']['batch_size'],cfg['training']['accumulation_steps'],step,drop_last=pretrain)
                    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=use_amp):
                        if pretrain:
                            a,b=model(batch['waveform']);loss=ssl_loss(a.float(),b.float(),cfg['training']['ssl_variance_weight'],cfg['training']['ssl_covariance_weight'])
                        else:
                            logits,encoding=model.forward_features(batch)
                            loss=criterion(logits,batch['label'])+model.auxiliary_loss(batch,batch['label'],encoding)
                        if cfg['training']['cross_run_weight']>0 and batch['partner_available'].any():
                            if not hasattr(model,'encoder'):raise ConfigurationError('Cross-run consistency requires a TCN-based encoder')
                            member=batch['partner_available']
                            a=model.encoder(batch['waveform'][member]) if pretrain else encoding[member]
                            b=model.encoder(batch['partner_waveform'][member])
                            consistency=(1-torch.nn.functional.cosine_similarity(a,b,dim=1)).mean()
                            loss=loss+cfg['training']['cross_run_weight']*consistency
                    if not torch.isfinite(loss):raise TrainingError(f'Nonfinite loss at epoch {epoch+1}, batch {step}; no samples were silently skipped')
                    scaler.scale(loss*(size/max(examples,1))).backward()
                    if (step+1)%cfg['training']['accumulation_steps']==0 or step+1==len(loader):
                        scaler.unscale_(optimizer)
                        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['training']['gradient_clip'],error_if_nonfinite=True)
                        scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                        updates+=1
                    total+=float(loss.detach())*size;seen+=size
                    if (step+1)%log_interval==0 or (step+1)==len(loader):
                        batch_elapsed=time.perf_counter()-start_time
                        batches_per_sec=(step+1)/max(batch_elapsed,1e-6)
                        epoch_eta=(len(loader)-(step+1))/batches_per_sec
                        LOG.info('  [Epoch %d/%d] Batch %d/%d (%.0f%%) | Loss: %.4f | Speed: %.1f batch/s | Epoch ETA: %.1fs',
                                 epoch+1,epochs,step+1,len(loader),(step+1)/len(loader)*100,float(loss.detach()),batches_per_sec,epoch_eta)
                scheduler.step()
                if device.type=='cuda':torch.cuda.synchronize(device)
                training_seconds=time.perf_counter()-start_time;validation_start=time.perf_counter()
                if pretrain:
                    score=-total/seen;improved=True
                else:
                    validation=predict(model,val_dataset,cfg,device)
                    result=classification_metrics(validation.label,expit(validation.logit))
                    score=result['average_precision']
                    if score is None:raise DataIntegrityError('Validation AP is undefined')
                    improved=score>best+cfg['training']['min_delta']
                if device.type=='cuda':torch.cuda.synchronize(device)
                validation_seconds=time.perf_counter()-validation_start;checkpoint_start=time.perf_counter()
                if improved:best=score;bad_epochs=0
                else:bad_epochs+=1
                state={'epoch':epoch,'pretrain':pretrain,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                    'scaler':scaler.state_dict(),'rng':rng_state(),'best':best,'bad_epochs':bad_epochs,'experiment_fingerprint':identity}
                save_checkpoint(folder/'last.pt',state)
                if improved:save_checkpoint(folder/'best.pt',state)
                epoch_elapsed=time.perf_counter()-start_time
                epoch_durations.append(epoch_elapsed)
                avg_epoch=sum(epoch_durations)/len(epoch_durations)
                total_eta_s=(epochs-(epoch+1))*avg_epoch
                eta_str=f'{int(total_eta_s//60):02d}m {int(total_eta_s%60):02d}s' if total_eta_s>=60 else f'{total_eta_s:.1f}s'
                row={'epoch':epoch+1,'training_loss':total/seen,'validation_ap':None if pretrain else score,'learning_rate':scheduler.get_last_lr()[0],
                    'training_seconds':training_seconds,'validation_seconds':validation_seconds,
                    'checkpoint_seconds':time.perf_counter()-checkpoint_start,'optimizer_updates':updates,'training_windows':seen,
                    'training_windows_per_second':seen/max(training_seconds,1e-12),
                    'elapsed_s':epoch_elapsed,'total_elapsed_s':time.perf_counter()-overall_start,'eta_s':total_eta_s,'best':best,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None}
                write_json(folder/f'epochs/{epoch+1:04d}.json',row)
                write_json(folder/'progress.json',{'current_epoch':epoch+1,'total_epochs':epochs,'elapsed_s':time.perf_counter()-overall_start,'eta_s':total_eta_s,'eta_human':eta_str,'training_loss':total/seen,'val_score':score,'best_score':best,'patience_remaining':cfg['training']['patience']-bad_epochs if not pretrain else None})
                LOG.info('Epoch %d/%d complete in %.2fs | Loss: %.5f | Val AP: %.5f | Best AP: %.5f | Total ETA: %s',epoch+1,epochs,epoch_elapsed,total/seen,score,best,eta_str)
                if not pretrain and bad_epochs>=cfg['training']['patience']:break
            if pretrain:
                save_checkpoint(folder/'encoder.pt',{'encoder':model.encoder.state_dict(),'data_fingerprint':data_info['data_fingerprint'],
                    'split_fingerprint':split_info['split_fingerprint'],'training_windows':fingerprint(sorted(training_rows.window_id.tolist())),
                    'normalizer':normalizer,'method':'time-frequency VICReg-style pretraining, all permitted train windows, fixed epochs'})
            else:
                model.load_state_dict(load_checkpoint(folder/'best.pt')['model']);validation=predict(model,val_dataset,cfg,device)
                finalize_validation(cfg,folder,validation,pd.read_csv(root/'sites.csv'),identity)
            write_json(folder/'DONE.json',{'completed':utc_now(),'stage':'pretrain' if pretrain else 'supervised','experiment_fingerprint':identity,
                'checkpoint_sha256':file_hash(folder/('encoder.pt' if pretrain else 'best.pt')),'test_evaluated':False})
            return {'run':str(folder),'complete':True,'test_evaluated':False}
        except BaseException as e:
            detail=str(e)
            if isinstance(e,torch.cuda.OutOfMemoryError):detail+=' Reduce configured batch size or candidate/patch budget, then resume the last completed epoch; no automatic data/batch substitution was performed.'
            write_json(folder/'FAILURE.json',{'failed':utc_now(),'type':type(e).__name__,'error':detail,'resume_checkpoint':str(folder/'last.pt') if (folder/'last.pt').exists() else None})
            if isinstance(e,(KeyboardInterrupt,SystemExit)):raise
            if isinstance(e,(ArtifactError,ConfigurationError,DataIntegrityError,TrainingError)):raise
            raise TrainingError(detail) from e
