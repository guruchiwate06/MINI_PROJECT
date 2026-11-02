import json
import pytest
from railvivid.benchmark import plan_suite,aggregate_suite
from railvivid.config import load_config
from railvivid.errors import ArtifactError,ConfigurationError

def test_suite_is_only_a_plan_with_dependencies_and_identical_cohorts(tmp_path,cfg):
    result=plan_suite(cfg,tmp_path/'plan',methods=['rf','tcn','ssl_tcn','attention'],seeds=[42],folds=[3])
    plan=json.loads((tmp_path/'plan/plan.json').read_text())
    assert result['jobs']==5 and plan['test_evaluation_is_separate']
    splits=[]
    for job in plan['jobs']:
        configuration=load_config(tmp_path/'plan'/job['config']);splits.append(configuration['split'])
        assert configuration['training']['require_cuda']
        if job['method']=='ssl_tcn':assert job['depends_on']
    assert all(s==splits[0] for s in splits)
    assert not (tmp_path/'experiments').exists()
    with pytest.raises(ArtifactError):plan_suite(cfg,tmp_path/'plan')

def test_missing_results_cannot_become_a_leaderboard(tmp_path,cfg):
    plan_suite(cfg,tmp_path/'plan',methods=['tcn'],seeds=[42],folds=[3])
    with pytest.raises(ArtifactError,match='results missing'):aggregate_suite(tmp_path/'plan/plan.json',tmp_path/'report')

def test_unknown_method_is_not_silently_substituted(tmp_path,cfg):
    with pytest.raises(ConfigurationError):plan_suite(cfg,tmp_path/'plan',methods=['imaginary_model'])
