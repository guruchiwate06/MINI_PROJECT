"""Explicit full-scale stages; planning/preflight never train or load backbone weights."""
from __future__ import annotations
import argparse,logging,math,time
from pathlib import Path
import pandas as pd
from .common import file_hash,fingerprint,read_json,write_json,utc_now,source_hash
from .config import load_config
from .errors import ArtifactError,ConfigurationError,DataIntegrityError,PipelineError

NEURAL_METHODS={'cnn','tcn','vision','concat','pooling','attention','cross_run','ssl_tcn','full'}
STRESS_CASES=['missing_vision','missing_vibration','missing_axles','noise_0p1','noise_0p5',
              'shift_minus5','shift_plus5','shift_minus10','shift_plus10','shift_minus15','shift_plus15']


def jobs_for(plan_path):
    plan_path=Path(plan_path).resolve();plan=read_json(plan_path)
    if not plan.get('jobs'):raise ConfigurationError('Campaign plan contains no jobs')
    ids=[j['id'] for j in plan['jobs']]
    if len(ids)!=len(set(ids)):raise DataIntegrityError('Duplicate job identities in campaign')
    jobs=[]
    for job in plan['jobs']:
        config=(plan_path.parent/job['config']).resolve()
        if plan_path.parent not in config.parents:raise ConfigurationError('Job config escapes plan directory')
        jobs.append((job,load_config(config)))
    return plan,jobs


def validate_plan_source(plan_path):
    marker=Path(plan_path).parent/'campaign.json';info=read_json(marker)
    if info['pipeline_source_hash']!=source_hash():raise ArtifactError('Campaign code changed after planning; create a new plan/output before training')
    if file_hash(plan_path)!=info['plan_sha256']:raise DataIntegrityError('Campaign plan changed')
    for name,digest in info['configs'].items():
        if file_hash(Path(plan_path).parent/name)!=digest:raise DataIntegrityError(f'Planned config changed: {name}')
    return info


def create_plan(cfg,suite_path,plan_path):
    from .benchmark import plan_suite
    suite=read_json(suite_path)
    if set(suite)!={'methods','seeds','folds'}:raise ConfigurationError('Suite requires methods, seeds and folds')
    plan_path=Path(plan_path).resolve()
    if plan_path.name!='plan.json':raise ConfigurationError('Campaign plan filename must be plan.json')
    result=plan_suite(cfg,plan_path.parent,**suite)
    plan,jobs=jobs_for(plan_path)
    write_json(plan_path.parent/'campaign.json',{'created':utc_now(),'pipeline_source_hash':source_hash(),
        'plan_sha256':file_hash(plan_path),'configs':{j['config']:file_hash(plan_path.parent/j['config']) for j,_ in jobs},
        'full_model':'train-only time-frequency SSL + TCN + frozen DINO + candidate attention + quality gate + cross-run consistency',
        'selection':'validation-only; outer tests previously inspected and exploratory',
        'benchmark_scope':{'primary':'joint unseen site/run, matched paired cohort',
            'metrics':['average_precision','event_recall','false_alarms_per_km','localization_mae_m','ece'],
            'stress_cases':STRESS_CASES,'repeated_pass_counts':[1,2,3],
            'comparisons':['rf','cnn','tcn','vision','patch_memory','concat','pooling','attention','cross_run','ssl_tcn','late']},
        'limits':['No faithful AnomalyDINO/FoundAD/TFSSCL reproduction is claimed.',
                  'DINOv3 requires a separately pinned plan; nominal alignment needs independent checks.',
                  'Few-label and pure run/site protocols require separate plans and experiment output directories.',
                  'No score target, state-of-the-art result or fixed GPU completion time is guaranteed.']})
    return result


def inspect_plan(plan_path,require_features=True,require_cuda=False):
    """Build/verify metadata splits, count update budget, report every unmet precondition."""
    from .prepare import load_prepared
    from .splits import split_path,build_split,load_split
    from .training import vision_for
    from .baselines import statistics_matrix,patch_cache
    from .common import select_device
    validate_plan_source(plan_path);_,jobs=jobs_for(plan_path)
    splits={};assets=set();rows=[];checks=[];total_updates=0
    for job,cfg in jobs:
        key=str(split_path(cfg))
        if key not in splits:
            root,meta=load_prepared(cfg,verify=True)
            if not split_path(cfg).exists():build_split(cfg)
            frame,info,_=load_split(cfg);splits[key]=(root,frame,info)
        root,frame,info=splits[key];train=frame.loc[frame.partition.eq('train')]
        supervised=train.loc[train.eligible_supervised];pretrain=job['command']=='pretrain'
        n=len(train) if pretrain else len(supervised);batch=cfg['training']['batch_size']
        batches=n//batch if pretrain else math.ceil(n/batch)
        neural=pretrain or job['method'] in NEURAL_METHODS
        epochs=cfg['training']['pretrain_epochs'] if pretrain else cfg['training']['epochs']
        updates=math.ceil(batches/cfg['training']['accumulation_steps'])*epochs if neural else 0
        total_updates+=updates
        rows.append({'id':job['id'],'method':job['method'],'fold':job['fold'],'seed':job['seed'],
            'training_windows':n,'supervised_positive_windows':int(supervised.label.eq(1).sum()),
            'batches_per_epoch':batches if neural else None,'max_updates':updates,'split':info['split_fingerprint']})
        if require_features:
            kind=cfg['model']['kind'];asset=(key,kind)
            if asset not in assets and not pretrain:
                assets.add(asset)
                try:
                    if kind=='rf':statistics_matrix(root,frame,cfg['model']['channels'])
                    elif kind in ['vision','fusion','patch_memory']:
                        store=vision_for(cfg,frame)
                        if kind=='patch_memory':
                            patches=patch_cache(store)
                            if file_hash(store.folder/'patches.npy')!=store.meta['artifacts'].get('patches.npy'):
                                raise DataIntegrityError('Patch cache checksum differs')
                            if len(patches)!=len(store.index):raise DataIntegrityError('Patch cache/index length differs')
                    checks.append({'asset':kind,'split':info['split_fingerprint'],'ok':True})
                except (PipelineError,OSError,ValueError) as e:
                    checks.append({'asset':kind,'split':info['split_fingerprint'],'ok':False,'error':str(e)})
    if require_cuda:
        try:select_device(jobs[0][1]['training']['device'],True);checks.append({'asset':'cuda','ok':True})
        except PipelineError as e:checks.append({'asset':'cuda','ok':False,'error':str(e)})
    return {'training_started':False,'created':utc_now(),'jobs':rows,'planned_tasks':len(rows),
        'splits':[info for _,_,info in splits.values()],'max_neural_updates':total_updates,
        'checks':checks,'failures':sum(not c['ok'] for c in checks),
        'features_checked':require_features,'cuda_checked':require_cuda,
        'timing':'No throughput inferred. Use actual per-epoch training/validation/checkpoint logs; extraction and tests are additional.',
        'conditional_training_hours':{str(ms)+'_ms_per_update':total_updates*ms/3_600_000 for ms in [20,50,100]}}


def prepare_assets(cfg):
    from .prepare import load_prepared,prepare
    from .features import extract_statistics
    from .baselines import statistics_matrix
    root=Path(cfg['data']['artifacts'])
    if not root.exists():prepare(cfg)
    root,_=load_prepared(cfg,verify=True)
    if not (root/'statistics').exists():extract_statistics(cfg)
    statistics_matrix(root,pd.read_csv(root/'windows.csv'),cfg['model']['channels'])
    return {'training_started':False,'prepared':str(root),'statistics_verified':True}


def encode_assets(cfg,allow_download=False,image_stress=False):
    from .vision import vision_path,extract_vision
    from .dataset import VisionStore
    from .common import select_device
    select_device(cfg['training']['device'],True)
    results=[]
    cases=[{}]+([{'blur_radius_px':2.},{'brightness_factor':.5}] if image_stress else [])
    for corruption in cases:
        folder=vision_path(cfg,corruption)
        if folder.exists():
            store=VisionStore(folder,cfg['model']['vision_dim'])
            if cfg['vision']['store_patch_tokens'] and file_hash(folder/'patches.npy')!=store.meta['artifacts'].get('patches.npy'):
                raise DataIntegrityError('Existing patch cache absent/modified')
            results.append({'cache':str(folder),'verified_complete_skipped':True})
        else:results.append(extract_vision(cfg,cfg['training']['device'],allow_download,corruption))
    return {'training_started':False,'inference_only':True,'caches':results}


def completed_evaluation(folder,partition,case,cfg,split_info,corruption):
    from .training import experiment_identity
    target=folder/'evaluations'/f'{partition}_{case}'
    if not target.exists():return False
    ready=read_json(target/'READY.json');provenance=read_json(target/'provenance.json')
    if provenance['experiment_fingerprint']!=experiment_identity(cfg,split_info) or provenance['corruption']!=corruption:
        raise DataIntegrityError('Existing evaluation has different identity or corruption')
    done=read_json(folder/'DONE.json')
    if done['experiment_fingerprint']!=provenance['experiment_fingerprint'] or done['checkpoint_sha256']!=provenance['checkpoint_sha256']:
        raise DataIntegrityError('Evaluation does not match the completed training run')
    for name,digest in ready['artifacts'].items():
        if file_hash(target/name)!=digest:raise DataIntegrityError(f'Evaluation modified: {target/name}')
    return True


def extra_benchmarks(plan_path,stage,image_stress=False):
    from .benchmark import STRESSES
    from .splits import load_split
    from .evaluation import evaluate
    from .training import run_path,experiment_identity
    from .prepare import load_prepared
    from .repeated import repeated_pass_study
    from .comparison import compare_files
    validate_plan_source(plan_path);_,jobs=jobs_for(plan_path);rows=[]
    target=Path(plan_path).parent/'benchmarks'/stage;target.mkdir(parents=True,exist_ok=True)
    def record(row):
        rows.append(row);write_json(target/'status.json',{'updated':utc_now(),'jobs':rows})
    lookup={(j['fold'],j['seed'],j['method']):(j,c) for j,c in jobs}
    for job,cfg in jobs:
        if job['command']=='pretrain':continue
        folder=run_path(cfg);_,info,_=load_split(cfg)
        try:
            if stage=='stress':
                if job['method'] not in NEURAL_METHODS:
                    record({'id':job['id'],'status':'excluded_by_protocol','reason':'RF, patch memory and score-level late fusion lack equivalent stress adapters; clean results remain required.'});continue
                cases=list(STRESS_CASES)
                if image_stress and cfg['model']['kind'] in ['vision','fusion']:cases+=['blur','dark']
                for case in cases:
                    if completed_evaluation(folder,'test',case,cfg,info,STRESSES[case]):
                        record({'id':job['id'],'case':case,'status':'verified_complete_skipped'});continue
                    evaluate(cfg,corruption=STRESSES[case],tag=case)
                    record({'id':job['id'],'case':case,'status':'ok'})
            elif stage=='repeated':
                if not completed_evaluation(folder,'test','clean',cfg,info,{}):raise ArtifactError('Run the clean evaluate stage first')
                output=target/job['id'];identity=experiment_identity(cfg,info)
                test=folder/'evaluations/test_clean/predictions.csv';validation=folder/'validation_predictions.csv'
                expected={'identity':identity,'test':file_hash(test),'validation':file_hash(validation),'counts':[1,2,3]}
                if output.exists():
                    ready=read_json(output/'CAMPAIGN_READY.json')
                    if ready['inputs']!=expected:raise ArtifactError('Repeated study incomplete or changed; inspect preserved output')
                    for name,digest in ready['artifacts'].items():
                        if file_hash(output/name)!=digest:raise DataIntegrityError('Repeated-study output changed')
                else:
                    root,_=load_prepared(cfg)
                    repeated_pass_study(pd.read_csv(test),pd.read_csv(validation),pd.read_csv(root/'sites.csv'),output,cfg)
                    write_json(output/'CAMPAIGN_READY.json',{'inputs':expected,
                        'artifacts':{p.name:file_hash(p) for p in output.iterdir() if p.is_file()}})
                record({'id':job['id'],'status':'ok'})
            elif stage=='compare':
                if job['method']=='full':continue
                counterpart=lookup.get((job['fold'],job['seed'],'full'))
                if not counterpart:raise ConfigurationError('Paired comparisons require a full model in each fold/seed')
                _,full_cfg=counterpart;_,full_info,_=load_split(full_cfg);full_folder=run_path(full_cfg)
                if not completed_evaluation(folder,'test','clean',cfg,info,{}) or not completed_evaluation(full_folder,'test','clean',full_cfg,full_info,{}):
                    raise ArtifactError('Both clean evaluations must be complete before comparison')
                first=folder/'evaluations/test_clean/predictions.csv';second=full_folder/'evaluations/test_clean/predictions.csv'
                output=target/(job['id']+'_vs_full.json');marker=output.with_suffix('.ready.json')
                expected={'first':file_hash(first),'second':file_hash(second),'repeats':cfg['evaluation']['bootstrap_repeats'],
                          'seed':cfg['evaluation']['bootstrap_seed']}
                if output.exists():
                    ready=read_json(marker)
                    if ready['inputs']!=expected or ready['output_sha256']!=file_hash(output):raise ArtifactError('Comparison incomplete or inputs/output changed')
                else:
                    compare_files(first,second,output,cfg['evaluation']['bootstrap_repeats'],cfg['evaluation']['bootstrap_seed'])
                    write_json(marker,{'inputs':expected,'output_sha256':file_hash(output)})
                record({'id':job['id'],'status':'ok'})
        except Exception as e:
            record({'id':job['id'],'status':'failed','error':str(e)});raise
    return {'benchmark':stage,'jobs':len(rows),'failures':0,'directory':str(target)}


def main(argv=None):
    root=Path(__file__).parents[1];p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['plan','prepare','extract','workload','preflight','train','evaluate','summarize','stress','repeated','compare'])
    p.add_argument('--config',default=str(root/'configs/full_scale.json'))
    p.add_argument('--suite',default=str(root/'configs/benchmark_suite.json'))
    p.add_argument('--plan',default=str(root/'artifacts/plans/full_scale/plan.json'))
    p.add_argument('--set',action='append',default=[],metavar='KEY=VALUE')
    p.add_argument('--allow-download',action='store_true');p.add_argument('--image-stress',action='store_true')
    args=p.parse_args(argv);logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    started=time.perf_counter();plan_path=Path(args.plan).resolve()
    try:
        cfg=load_config(args.config,args.set)
        if args.set and args.stage not in ['plan','prepare','extract']:
            raise ConfigurationError('Overrides apply at plan/prepare/extract time; later stages consume immutable planned configs')
        if args.stage=='plan':result=create_plan(cfg,args.suite,plan_path)
        elif args.stage=='prepare':result=prepare_assets(cfg)
        elif args.stage=='extract':result=encode_assets(cfg,args.allow_download,args.image_stress)
        elif args.stage in ['workload','preflight','train']:
            check=inspect_plan(plan_path,require_features=args.stage!='workload',require_cuda=args.stage in ['preflight','train'])
            write_json(plan_path.parent/(args.stage+'_preflight.json'),check)
            if check['failures']:raise ArtifactError(f"{check['failures']} preflight checks failed; see {plan_path.parent/(args.stage+'_preflight.json')}")
            result=check
            if args.stage=='train':
                from .benchmark import execute_suite
                result=execute_suite(plan_path)
        elif args.stage=='evaluate':
            validate_plan_source(plan_path)
            from .benchmark import execute_suite
            result=execute_suite(plan_path,'evaluate')
        elif args.stage=='summarize':
            validate_plan_source(plan_path)
            from .benchmark import aggregate_suite
            result=aggregate_suite(plan_path,plan_path.parent/'leaderboard')
        else:result=extra_benchmarks(plan_path,args.stage,args.image_stress)
        from .common import canonical
        print(canonical(result));return 0
    except KeyboardInterrupt:
        logging.error('Interrupted; completed artifacts and last epoch checkpoint are preserved.');return 130
    except Exception as e:
        logging.exception('Campaign %s failed: %s',args.stage,e);return 1
    finally:
        if plan_path.parent.exists():
            write_json(plan_path.parent/('last_'+args.stage+'_invocation.json'),{'stage':args.stage,'wall_seconds':time.perf_counter()-started,'finished':utc_now()})


if __name__=='__main__':raise SystemExit(main())
