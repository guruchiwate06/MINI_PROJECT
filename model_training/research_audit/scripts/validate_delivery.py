"""Check generated artifacts and run source-overlap regression tests."""
from pathlib import Path
import os,sys,json,re,subprocess
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'research_audit'; E=OUT/'evidence'
REPO = ROOT.parent if ROOT.name == 'model_training' else ROOT / 'mini_project_rajguru'

def main():
    reports=['PROJECT_CURRENT_STATE','DATASET_FEASIBILITY','LABEL_AND_SPLIT_AUDIT','MULTIMODAL_ALIGNMENT_AUDIT',
        'EXISTING_BENCHMARK_VERIFICATION','GPU_MODEL_FEASIBILITY','RESEARCH_NOVELTY_MATRIX','RECOMMENDED_ARCHITECTURE','IMPLEMENTATION_ROADMAP','FINAL_RESEARCH_VERDICT']
    for name in reports: assert (OUT/(name+'.md')).stat().st_size>500
    bad=[]
    for f in OUT.glob('*.md'):
        for link in re.findall(r'\]\(([^)]+)\)',f.read_text(encoding='utf-8')):
            if not link.startswith('http') and not (f.parent/link).exists():bad.append((f.name,link))
    assert not bad,bad
    inv=pd.read_csv(E/'run_inventory.csv'); assert len(inv)==20 and inv.rows.sum()==9401255 and inv.images.sum()==105901
    assert len(pd.read_csv(E/'nominal_pair_manifest.csv'))==190
    assert len(pd.read_csv(E/'raw_csv_sha256.csv'))==20
    metrics=pd.read_csv(E/'benchmark_reproduction.csv')
    assert metrics.loc[metrics.model.eq('Random Forest'),'max_absolute_metric_difference'].max()<1e-12
    audit_env=os.environ.copy();audit_env['PYTHONPATH']=str(REPO)
    result=subprocess.run([sys.executable,'-m','pytest',str(REPO/'tests'),str(OUT/'tests'),'-q'],cwd=ROOT,env=audit_env,capture_output=True,text=True)
    (E/'test_results.txt').write_text(result.stdout+'\nSTDERR:\n'+result.stderr,encoding='utf-8')
    assert result.returncode==0,result.stdout+result.stderr
    status={str(p):subprocess.check_output(['git','-C',str(p),'status','--porcelain'],text=True).strip() for p in [REPO,Path('D:/Rail-VIVID')]}
    assert all(not v for v in status.values()),status
    summary={'report_count':len(reports),'reports_nonempty':True,'local_markdown_links_valid':True,'rf_reproduction_saved_precision':'pass',
        'sensor_rows':int(inv.rows.sum()),'images_inventoried':int(inv.images.sum()),'original_working_trees_clean':True,
        'test_output':result.stdout.strip()}
    (E/'delivery_validation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
