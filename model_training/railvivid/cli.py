"""Explicit commands: preparation/inference never dispatch training implicitly."""
from __future__ import annotations
import argparse,json,logging,sys,zipfile
from pathlib import Path
from .errors import PipelineError,ConfigurationError,DataIntegrityError,ArtifactError,TrainingError
from .common import jsonable,read_json,write_json,environment,file_hash

def parser():
    p=argparse.ArgumentParser(prog='railvivid',description='Reproducible Rail-VIVID research. Training is explicit and requires CUDA by default.')
    p.add_argument('--verbose',action='store_true');sub=p.add_subparsers(dest='command',required=True)
    for name in ['doctor','prepare','split','features','extract-vision','smoke','pretrain','train','evaluate','late-fit','late-evaluate','robustness','repeated','plan']:
        q=sub.add_parser(name);q.add_argument('--config',default=str(Path(__file__).parents[1]/'configs/pipeline.json'));q.add_argument('--set',action='append',default=[],metavar='KEY=VALUE')
        if name=='doctor':q.add_argument('--verify-cache',action='store_true')
        if name in ['extract-vision','smoke','evaluate']:q.add_argument('--device',default=None)
        if name=='extract-vision':
            q.add_argument('--allow-download',action='store_true');q.add_argument('--corruption',default='{}',help='JSON image stress specification')
        if name=='evaluate':
            q.add_argument('--partition',choices=['validation','test'],default='test');q.add_argument('--corruption',default='{}');q.add_argument('--tag')
        if name=='late-fit':q.add_argument('--vibration-run',required=True);q.add_argument('--vision-run',required=True)
        if name=='late-evaluate':q.add_argument('--partition',choices=['validation','test'],default='test')
        if name=='robustness':q.add_argument('--cases',default='clean,missing_vision,missing_axles,noise_0p1,shift_minus15,shift_plus15')
        if name=='repeated':
            q.add_argument('--predictions',required=True);q.add_argument('--validation-predictions',required=True);q.add_argument('--output',required=True)
            q.add_argument('--counts',default='1,2,3');q.add_argument('--repeats',type=int,default=30);q.add_argument('--segment-m',type=float,default=10.)
        if name=='plan':
            q.add_argument('--output',required=True);q.add_argument('--methods',default='rf,cnn,tcn,vision,patch_memory,concat,pooling,attention,cross_run,ssl_tcn,full,late')
            q.add_argument('--seeds',default='42,43,44');q.add_argument('--folds',default='1,2,3')
            q.add_argument('--suite',help='JSON suite definition containing methods, seeds and folds')
    for name in ['suite-run','suite-evaluate','summarize']:
        q=sub.add_parser(name);q.add_argument('--plan',required=True)
        if name=='summarize':q.add_argument('--output',required=True)
        else:q.add_argument('--continue-on-error',action='store_true')
    q=sub.add_parser('bundle');q.add_argument('--output',required=True)
    q=sub.add_parser('compare');q.add_argument('--first',required=True);q.add_argument('--second',required=True);q.add_argument('--output',required=True)
    q.add_argument('--repeats',type=int,default=1000);q.add_argument('--seed',type=int,default=73)
    return p

def doctor(cfg,verify=False):
    from .prepare import load_prepared
    from .splits import load_split
    from .vision import vision_path
    result={'environment':environment(),'training_started':False,'checks':[]}
    for name,operation in [('prepared_data',lambda:load_prepared(cfg,verify)),('split',lambda:load_split(cfg))]:
        try:
            value=operation();info=value[1]
            result['checks'].append({'name':name,'ok':True,'fingerprint':info.get('data_fingerprint',info.get('split_fingerprint'))})
        except PipelineError as e:result['checks'].append({'name':name,'ok':False,'error':str(e)})
    result['training_cuda_ready']=result['environment']['cuda_available'];result['vision_revision_pinned']=bool(cfg['vision']['revision'])
    result['notes']=['Local doctor, preparation and smoke do not train models.','Nominal camera positions are unvalidated; landmark mode requires independent checks.']
    return result

def smoke(cfg,device='cpu'):
    import torch
    from .common import select_device,seed_everything
    from .prepare import load_prepared
    from .splits import load_split
    from .dataset import WindowDataset,training_normalizer,collate_to_device
    from .models import make_model
    from .training import make_loader,vision_for
    root,_=load_prepared(cfg,verify=True);frame,info,_=load_split(cfg);train=frame.loc[frame.partition.eq('train')]
    if cfg['model']['kind'] in ['rf','patch_memory']:raise ConfigurationError('Smoke is forward-only for neural models; statistical fitting is never used for a smoke test')
    normalizer=training_normalizer(root,train,cfg['model']['channels']);vision=vision_for(cfg,frame)
    dataset=WindowDataset(root,train.loc[train.eligible_supervised].head(4),normalizer,cfg,vision)
    dev=select_device(device,False);seed_everything(cfg['training']['seed']);model=make_model(cfg).to(dev).eval()
    batch=next(iter(make_loader(dataset,cfg)));batch=collate_to_device(batch,dev)
    with torch.inference_mode():logits=model(batch)
    if not torch.isfinite(logits).all():raise DataIntegrityError('Nonfinite smoke output')
    return {'trained':False,'optimizer_steps':0,'inference_only':True,'device':str(dev),'real_windows':len(logits),
        'input_shape':list(batch['waveform'].shape),'output_shape':list(logits.shape),'parameters':sum(p.numel() for p in model.parameters()),'split':info['split_fingerprint']}

def dispatch(args):
    if args.command=='compare':
        from .comparison import compare_files
        return compare_files(args.first,args.second,args.output,args.repeats,args.seed)
    if args.command=='bundle':
        root=Path(__file__).parents[1];target=Path(args.output).resolve()
        if target.exists():raise ArtifactError('Bundle exists; choose a new filename')
        target.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for folder in ['railvivid','src','configs','tests_pipeline','scripts']:
                for f in (root/folder).rglob('*'):
                    if f.is_file() and '__pycache__' not in f.parts:z.write(f,str(f.relative_to(root)))
            for name in ['pyproject.toml','TRAINING_PIPELINE.md']:
                if (root/name).exists():z.write(root/name,name)
        return {'bundle':str(target),'sha256':file_hash(target),'contains_raw_data':False,'training_started':False}
    if args.command in ['suite-run','suite-evaluate','summarize']:
        from .benchmark import execute_suite,aggregate_suite
        if args.command=='summarize':return aggregate_suite(args.plan,args.output)
        return execute_suite(args.plan,'train' if args.command=='suite-run' else 'evaluate',args.continue_on_error)
    from .config import load_config
    cfg=load_config(args.config,args.set)
    if args.command=='doctor':return doctor(cfg,args.verify_cache)
    if args.command=='prepare':
        from .prepare import prepare
        return prepare(cfg)
    if args.command=='split':
        from .splits import build_split
        return build_split(cfg)
    if args.command=='features':
        from .features import extract_statistics
        return extract_statistics(cfg)
    if args.command=='extract-vision':
        from .vision import extract_vision
        return extract_vision(cfg,args.device or cfg['training']['device'],args.allow_download,json.loads(args.corruption))
    if args.command=='smoke':return smoke(cfg,args.device or 'cpu')
    if args.command in ['train','pretrain']:
        from .training import train
        return train(cfg,pretrain=args.command=='pretrain')
    if args.command=='evaluate':
        from .evaluation import evaluate
        return evaluate(cfg,args.partition,json.loads(args.corruption),args.tag,args.device)
    if args.command=='late-fit':
        from .late_fusion import fit_late
        return fit_late(cfg,args.vibration_run,args.vision_run)
    if args.command=='late-evaluate':
        from .late_fusion import evaluate_late
        return evaluate_late(cfg,args.partition)
    if args.command=='robustness':
        from .benchmark import STRESSES
        from .evaluation import evaluate
        results=[]
        for case in args.cases.split(','):
            if case not in STRESSES:raise ConfigurationError(f'Unknown stress case: {case}')
            results.append(evaluate(cfg,corruption=STRESSES[case],tag=case))
        return results
    if args.command=='plan':
        from .benchmark import plan_suite
        if args.suite:
            suite=read_json(args.suite)
            if set(suite)!=set(['methods','seeds','folds']):raise ConfigurationError('Suite definition requires exactly methods, seeds and folds')
            return plan_suite(cfg,args.output,suite['methods'],suite['seeds'],suite['folds'])
        return plan_suite(cfg,args.output,args.methods.split(','),[int(s) for s in args.seeds.split(',')],[int(s) for s in args.folds.split(',')])
    if args.command=='repeated':
        import pandas as pd
        from .repeated import repeated_pass_study
        from .prepare import load_prepared
        root,_=load_prepared(cfg)
        return repeated_pass_study(pd.read_csv(args.predictions),pd.read_csv(args.validation_predictions),pd.read_csv(root/'sites.csv'),args.output,cfg,
            counts=[int(n) for n in args.counts.split(',')],repeats=args.repeats,segment_m=args.segment_m)
    raise ConfigurationError('Unhandled command')

def main(argv=None):
    args=parser().parse_args(argv);logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    try:
        result=dispatch(args);print(json.dumps(jsonable(result),indent=2,allow_nan=False));return 1 if isinstance(result,dict) and result.get('failures',0) else 0
    except KeyboardInterrupt:
        logging.error('Interrupted. Last completed-epoch checkpoint and partial-artifact diagnostics were preserved.');return 130
    except (ConfigurationError,DataIntegrityError,ArtifactError,TrainingError) as e:
        logging.error('%s: %s',type(e).__name__,e,exc_info=args.verbose)
        return {ConfigurationError:2,DataIntegrityError:3,ArtifactError:4,TrainingError:5}.get(type(e),3)
    except (OSError,ValueError,KeyError,RuntimeError) as e:
        logging.error('Operation failed: %s',e,exc_info=args.verbose);return 6
    except Exception as e:
        logging.exception('Unexpected pipeline failure; no successful result is claimed: %s',e);return 7

if __name__=='__main__':raise SystemExit(main())
