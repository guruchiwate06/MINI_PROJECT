from __future__ import annotations
import copy,json,subprocess,sys
from pathlib import Path
import pandas as pd
from .common import read_json,write_json,write_csv,utc_now,fingerprint
from .config import validate_config
from .errors import ArtifactError,ConfigurationError,DataIntegrityError

METHODS={
 'rf':{'model.kind':'rf'},'cnn':{'model.kind':'cnn'},'tcn':{'model.kind':'tcn'},
 'vision':{'model.kind':'vision'},'patch_memory':{'model.kind':'patch_memory'},
 'concat':{'model.kind':'fusion','model.fusion':'concat'},'pooling':{'model.kind':'fusion','model.fusion':'distance'},
 'attention':{'model.kind':'fusion','model.fusion':'attention'},'gated':{'model.kind':'fusion','model.fusion':'gated'},
 'cross_run':{'model.kind':'fusion','model.fusion':'attention','training.cross_run_weight':0.1},
 'ssl_tcn':{'model.kind':'tcn'},'late':{'model.kind':'fusion','model.fusion':'concat'},
 'full':{'model.kind':'fusion','model.fusion':'attention_gated','training.cross_run_weight':0.1}
}
FOLDS=[([1,2,3],[4]),([4,5,6],[7]),([7,8,9],[6])]
STRESSES={'clean':{},'missing_vision':{'missing_vision':True},'missing_axles':{'missing_channels':[2,4]},
 'noise_0p1':{'noise_std_normalized':.1},'noise_0p5':{'noise_std_normalized':.5},
 'shift_minus15':{'alignment_prior_shift_m':-15.},'shift_plus15':{'alignment_prior_shift_m':15.},
 'shift_minus5':{'alignment_prior_shift_m':-5.},'shift_plus5':{'alignment_prior_shift_m':5.},
 'blur':{'blur_radius_px':2.},'dark':{'brightness_factor':.5}}

def set_value(cfg,key,value):
    a,b=key.split('.');cfg[a][b]=value

def plan_suite(cfg,output,methods=None,seeds=(42,43,44),folds=(1,2,3)):
    output=Path(output).resolve()
    if output.exists():raise ArtifactError(f'Suite plan already exists: {output}')
    methods=methods or ['rf','cnn','tcn','vision','patch_memory','concat','pooling','attention','cross_run','ssl_tcn','full','late']
    unknown=set(methods)-set(METHODS)
    if unknown:raise ConfigurationError(f'Unknown benchmark methods: {sorted(unknown)}')
    methods=list(dict.fromkeys(methods))
    if 'late' in methods:
        methods=[m for m in methods if m!='late']
        for m in ['tcn','vision']:
            if m not in methods:methods.append(m)
        methods.append('late')
    if not set(folds)<=set([1,2,3]):raise ConfigurationError('Fold IDs must be 1, 2 or 3')
    output.mkdir(parents=True);(output/'configs').mkdir();jobs=[]
    for fold in folds:
        for seed in seeds:
            base=copy.deepcopy(cfg);base.pop('_config_path',None)
            base['split'].update({'protocol':'combined','cohort':'paired','test_sites':FOLDS[fold-1][0],'validation_sites':FOLDS[fold-1][1]})
            base['training'].update({'seed':seed,'resume':None});base['model']['initial_encoder']=None
            pretrain_name=f'fold{fold}_pretrain_s{seed}'
            if set(methods)&{'ssl_tcn','full'}:
                pre=copy.deepcopy(base);pre['model']['kind']='tcn';pre['experiment']['name']=pretrain_name
                relative=f'configs/{pretrain_name}.json';write_json(output/relative,pre)
                jobs.append({'id':pretrain_name,'method':'pretrain','fold':fold,'seed':seed,'command':'pretrain','config':relative,'depends_on':None})
            for method in methods:
                job=copy.deepcopy(base)
                for key,value in METHODS[method].items():set_value(job,key,value)
                name=f'fold{fold}_{method}_s{seed}';job['experiment']['name']=name
                if method in ['ssl_tcn','full']:job['model']['initial_encoder']=str(Path(base['experiment']['output'])/pretrain_name/'encoder.pt')
                validate_config(job);relative=f'configs/{name}.json';write_json(output/relative,job)
                jobs.append({'id':name,'method':method,'fold':fold,'seed':seed,'command':'train','config':relative,
                    'depends_on':pretrain_name if method in ['ssl_tcn','full'] else None})
                if method=='late':
                    jobs[-1].update({'command':'late-fit','depends_on':[f'fold{fold}_tcn_s{seed}',f'fold{fold}_vision_s{seed}'],
                        'vibration_run':f'fold{fold}_tcn_s{seed}','vision_run':f'fold{fold}_vision_s{seed}'})
    plan={'created':utc_now(),'jobs':jobs,'methods':methods,'folds':list(folds),'seeds':list(seeds),
          'test_evaluation_is_separate':True,'claim':'planned comparisons, no performance generated','protocol':'matched paired cohorts; fold/seed counts are not independent defect counts'}
    write_json(output/'plan.json',plan);write_csv(output/'jobs.csv',pd.DataFrame(jobs));return {'plan':str(output/'plan.json'),'jobs':len(jobs)}

def execute_suite(plan_path,stage='train',continue_on_error=False):
    plan_path=Path(plan_path).resolve();plan=read_json(plan_path);status=[];failed=set()
    for job in plan['jobs']:
        if stage=='evaluate' and job['command']=='pretrain':continue
        dependencies=job.get('depends_on') or [];dependencies=[dependencies] if isinstance(dependencies,str) else dependencies
        if any(d in failed for d in dependencies):
            status.append({'id':job['id'],'status':'dependency_failed'});failed.add(job['id']);continue
        from .config import load_config
        from .training import experiment_identity
        from .splits import split_path,build_split,load_split
        config=plan_path.parent/job['config'];cfg=load_config(config)
        command=job['command'] if stage=='train' else ('late-evaluate' if job['command']=='late-fit' else 'evaluate')
        if stage=='train' and command in ['train','pretrain']:
            from .common import select_device
            select_device(cfg['training']['device'],cfg['training']['require_cuda'])
        if not split_path(cfg).exists():build_split(cfg)
        folder=Path(cfg['experiment']['output'])/cfg['experiment']['name']
        completed=folder/'DONE.json' if stage=='train' else folder/'evaluations/test_clean/provenance.json'
        if completed.exists():
            if stage=='evaluate' and not (completed.parent/'READY.json').exists():raise ArtifactError('Incomplete evaluation output exists; inspect preserved artifacts before recovery')
            existing=read_json(completed);_,split_info,_=load_split(cfg)
            if existing['experiment_fingerprint']!=experiment_identity(cfg,split_info):raise ArtifactError(f"Existing result for {job['id']} has a different source/config identity")
            status.append({'id':job['id'],'status':'verified_complete_skipped'});continue
        args=[sys.executable,'-m','railvivid',command,'--config',str(config)]
        if command=='late-fit':
            args+=['--vibration-run',str(Path(cfg['experiment']['output'])/job['vibration_run']),
                   '--vision-run',str(Path(cfg['experiment']['output'])/job['vision_run'])]
        if stage=='train' and (folder/'last.pt').is_file():args+=['--set','training.resume='+str(folder/'last.pt')]
        result=subprocess.run(args,check=False)
        item={'id':job['id'],'command':command,'returncode':result.returncode,'status':'ok' if result.returncode==0 else 'failed'};status.append(item)
        write_json(plan_path.parent/f'{stage}_status.json',{'updated':utc_now(),'jobs':status})
        if result.returncode:
            failed.add(job['id'])
            if not continue_on_error:raise ArtifactError(f"Suite job {job['id']} failed with code {result.returncode}; recorded status, no job was silently substituted")
    return {'jobs':status,'failures':len(failed)}

def aggregate_suite(plan_path,output):
    plan_path=Path(plan_path).resolve();plan=read_json(plan_path);rows=[];cohorts={};missing=[]
    from .config import load_config
    for job in plan['jobs']:
        if job['command']=='pretrain':continue
        cfg=load_config(plan_path.parent/job['config']);evaluation=Path(cfg['experiment']['output'])/cfg['experiment']['name']/'evaluations/test_clean'
        if not (evaluation/'READY.json').is_file():missing.append(job['id']);continue
        ready=read_json(evaluation/'READY.json')
        from .common import file_hash
        for name,digest in ready['artifacts'].items():
            if file_hash(evaluation/name)!=digest:raise DataIntegrityError(f"Evaluation artifact modified: {job['id']}/{name}")
        provenance=read_json(evaluation/'provenance.json');metric=read_json(evaluation/'metrics.json')
        key=(job['fold'],job['seed']);cohorts.setdefault(key,set()).add(provenance['cohort_fingerprint'])
        rows.append({'method':job['method'],'fold':job['fold'],'seed':job['seed'],
            **{k:metric[k] for k in ['average_precision','roc_auc','f1','precision','recall','brier','ece']},
            **{k:metric['events'][k] for k in ['event_recall','false_alarms_per_km','localization_mae_m']}})
    if any(len(values)>1 for values in cohorts.values()):raise DataIntegrityError('Methods were evaluated on different cohorts in the same fold/seed; refusing a misleading ranking')
    if missing:raise ArtifactError(f'{len(missing)} planned results missing: {missing}. No incomplete leaderboard is labelled complete.')
    target=Path(output)
    if target.exists():raise ArtifactError('Aggregate output exists; choose a new report path')
    target.mkdir(parents=True);frame=pd.DataFrame(rows);write_csv(target/'all_results.csv',frame)
    summary=frame.groupby('method')[['average_precision','f1','event_recall','false_alarms_per_km']].agg(['mean','std','min','max'])
    summary.columns=['_'.join(c) for c in summary.columns];write_csv(target/'summary.csv',summary.reset_index())
    from .common import atomic_write
    markdown='| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'
    markdown+='\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))
    atomic_write(target/'REPORT.md',lambda f:f.write('# Matched-cohort benchmark\n\n'+markdown+'\n\nMeans and standard deviations describe these exploratory folds/seeds, not independent railway populations. All planned results are included; model selection must not use this test table.\n'))
    return {'report':str(target),'results':len(frame)}
