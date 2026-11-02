"""Fit-free validation runner. It never calls any training command."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from railvivid.common import write_json,environment

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='reports/pipeline_validation.json');args=parser.parse_args()
    result=subprocess.run([sys.executable,'-m','pytest','tests_pipeline','-q'],cwd=ROOT,capture_output=True,text=True)
    report={'training_performed':False,'optimizer_steps':0,'pretrained_weights_downloaded':False,
        'scope':'Fitting-free unit tests, synthetic fixture inference, checkpoint/metrics/split/CLI validation. GPU training and real pretrained encoder execution were not run.',
        'tests_returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'environment':environment()}
    write_json(ROOT/args.output,report);print(result.stdout);print('Validation report:',ROOT/args.output)
    return result.returncode

if __name__=='__main__':raise SystemExit(main())
