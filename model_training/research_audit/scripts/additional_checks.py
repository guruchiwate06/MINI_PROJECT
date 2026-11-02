"""Cross-check prior EDA claims against actual records and executable artifacts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]; E=ROOT/'research_audit/evidence'; DATA=Path('D:/Rail-VIVID')
REPO = ROOT.parent if ROOT.name == 'model_training' else ROOT / 'mini_project_rajguru'

def main():
    inv=pd.read_csv(E/'run_inventory.csv'); rows=[]
    for run in inv.run_id:
        a=pd.read_csv(DATA/run/(run+'.csv'),usecols=['Temperature','Humidity'])
        row={'run_id':run,'date':inv.loc[inv.run_id.eq(run),'start'].iloc[0][:10]}
        row.update({c+'_'+stat:float(getattr(a[c],stat)()) for c in a for stat in ['min','max']})
        rows.append(row)
    pd.DataFrame(rows).to_csv(E/'environment_by_run.csv',index=False)
    nb=json.loads((REPO/'notebooks/01_dataset_exploration.ipynb').read_text(encoding='utf-8'))
    code=[c for c in nb['cells'] if c['cell_type']=='code']
    summary={'speed_duration_pearson_r':float(inv.target_speed_kmh.corr(inv.duration_s)),
        'notebook_code_cells':len(code),'notebook_executed_code_cells':sum(c.get('execution_count') is not None for c in code),
        'notebook_code_cells_with_outputs':sum(bool(c.get('outputs')) for c in code),
        'legacy_figure6_bin_count':159,'legacy_figure6_bin_latitude_span_deg':float((40.2360-40.2319)/159),
        'legacy_figure6_bin_northing_span_m':float((40.2360-40.2319)/159*111320),
        'legacy_figure6_true_bin_width_m':'not constant in chainage; latitude bins are not fixed 7.62 m route sections'}
    (E/'legacy_eda_checks.json').write_text(json.dumps(summary,indent=2))
    print(summary)

if __name__=='__main__':main()
