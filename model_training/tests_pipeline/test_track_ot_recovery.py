"""Recovery regressions: gradients/profiling only, never parameter updates."""
import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch

from railvivid.common import read_json, write_json
from railvivid.errors import ArtifactError, DataIntegrityError, TrainingError
from track_ot.cli import synthetic_batch
from track_ot.config import DEFAULTS
from track_ot.model import TrackOTPlus
from track_ot.losses import objective
from track_ot.numerics import strict_clip_gradients
from track_ot.transport import unbalanced_transport_batch, transport_summary


@pytest.fixture
def small(cfg):
    torch.set_num_threads(2)
    c = copy.deepcopy(DEFAULTS); c['_base'] = cfg
    c['sequence'].update(windows=5, max_frames=7)
    c['encoder'].update(dimension=24, heads=3, layers=1, sensor_width=4, sensor_depth=1, spectral_bins=8, patch_tokens=4, dropout=0.0)
    return c


def reference_plan(cost, allowed, epsilon=0.25, tau=2.0):
    row, col = allowed.any(1), allowed.any(0)
    result = cost * 0
    if not row.any(): return result
    kernel = (-cost[row][:, col] / epsilon).masked_fill(~allowed[row][:, col], -float('inf'))
    a = cost.new_full((int(row.sum()),), -np.log(cost.shape[0]))
    b = cost.new_full((int(col.sum()),), -np.log(cost.shape[1]))
    u, v = torch.zeros_like(a), torch.zeros_like(b)
    for _ in range(200):
        u = tau / (tau + epsilon) * (a - torch.logsumexp(kernel + v[None], 1))
        v = tau / (tau + epsilon) * (b - torch.logsumexp(kernel + u[:, None], 0))
    r, c = torch.where(row)[0], torch.where(col)[0]
    return result.index_put((r[:, None].expand(-1, len(c)), c[None].expand(len(r), -1)), (kernel + u[:, None] + v[None]).exp())


def test_batched_uot_matches_independent_converged_reference():
    torch.manual_seed(22)
    cost = (torch.rand(3, 5, 7) * 4).requires_grad_()
    allowed = torch.ones_like(cost, dtype=torch.bool)
    allowed[0, -1] = False; allowed[0, :, -1] = False; allowed[1] = False
    plan, residual, steps = unbalanced_transport_batch(cost, allowed)
    ref = torch.stack([reference_plan(c, mask) for c, mask in zip(cost, allowed)])
    torch.testing.assert_close(plan, ref, atol=2e-5, rtol=0.003)
    g1 = torch.autograd.grad((plan * cost).sum(), cost, retain_graph=True)[0]
    g2 = torch.autograd.grad((ref * cost).sum(), cost)[0]
    torch.testing.assert_close(g1, g2, atol=0.0002, rtol=0.03)
    assert torch.isfinite(g1).all() and plan[1].eq(0).all()
    assert steps < 160 and (residual <= 0.005).all()


def test_empty_and_tiny_transport_rows_have_finite_gradients():
    plan = torch.tensor([[0.0, 0.0], [1e-35, 2e-35], [0.2, 0.3]], requires_grad=True)
    mass, weights, entropy = transport_summary(plan)
    (weights.square().sum() + entropy.sum() + mass.sum()).backward()
    assert torch.isfinite(plan.grad).all() and weights[:2].eq(0).all()


def test_padding_does_not_change_transport_reference_masses():
    cost = torch.rand(1, 5, 7)
    rows = torch.tensor([[True, False, True, False, True]])
    cols = torch.tensor([[False, True, True, False, True, True, False]])
    allowed = rows[:, :, None] & cols[:, None, :]
    plan, _, _ = unbalanced_transport_batch(cost, allowed, rows, cols)
    compact = cost[0, rows[0]][:, cols[0]]
    expected = reference_plan(compact, torch.ones_like(compact, dtype=torch.bool))
    torch.testing.assert_close(plan[0, rows[0]][:, cols[0]], expected, atol=2e-5, rtol=0.003)
    assert plan[~allowed].eq(0).all()


def test_double_norm_clips_large_finite_gradients_and_rejects_inf():
    model = torch.nn.Linear(2, 2)
    for p in model.parameters(): p.grad = torch.full_like(p, 1e30)
    norm = strict_clip_gradients(model, 1.0)
    assert torch.isfinite(norm) and norm > 1e30
    assert sum(float(p.grad.double().square().sum()) for p in model.parameters()) <= 1.000001
    model.weight.grad[0, 0] = float('inf')
    with pytest.raises(TrainingError, match='weight'): strict_clip_gradients(model, 1.0)


def test_patch_selection_does_not_read_full_frame_patch_payload():
    from track_ot.data import selected_patches
    values = np.arange(4 * 256 * 8).reshape(4, 256, 8)
    class Cache:
        shape = values.shape
        def __getitem__(self, key):
            assert isinstance(key, tuple) and len(key) == 3
            assert key[1].size == 16
            return values[key]
    selected = selected_patches(Cache(), [1, 3], 16)
    np.testing.assert_array_equal(selected, values[[1, 3]][:, np.linspace(0, 255, 16, dtype=int)])


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_full_loss_finite_without_optimizer_and_no_redundant_details(small, dtype, monkeypatch):
    model = TrackOTPlus(small)
    batch = synthetic_batch(small)
    other = synthetic_batch(small)
    for key, value in other.items(): batch['partner_' + key] = value
    batch['partner_available'] = torch.ones(2, dtype=torch.bool)
    calls = []
    original = model.details
    def counted(b):
        calls.append(1); return original(b)
    monkeypatch.setattr(model, 'details', counted)
    with torch.autocast('cpu', dtype=dtype, enabled=dtype != torch.float32):
        output = model.details(batch)
        loss, terms = objective(model, output, batch, small, torch.tensor(1.0))
    try:
        loss.backward()
    except RuntimeError as e:
        if 'DNNL does not support bf16' in str(e) or 'avx2' in str(e):
            pytest.skip(f"CPU platform does not support bfloat16 backward pass: {e}")
        raise
    strict_clip_gradients(model, 1.0)
    assert len(calls) == 2  # Main and partner; equivariance never calls details.
    assert torch.isfinite(loss)


def test_diagnostic_is_training_free_and_has_components(small, tmp_path):
    from track_ot.diagnostics import diagnose
    result = diagnose(small, tmp_path / 'diagnostics', synthetic=True, batches=1, components=True)
    report = read_json(tmp_path / 'diagnostics/diagnostics.json')
    row = report['precision_runs'][0]
    assert result['optimizer_steps'] == 0 and report['test_accessed'] is False
    assert row['weights_unchanged'] and row['all_finite']
    assert row['results'][0]['component_gradients']['cycle']['finite']


def load_script(name):
    path = Path(__file__).parents[1] / 'scripts' / name
    spec = importlib.util.spec_from_file_location(name.replace('.', '_'), path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_auto_launcher_requires_explicit_training_host_and_has_no_test_dispatch(monkeypatch):
    module = load_script('auto_train.py')
    def forbidden(*a, **k): raise AssertionError('No subprocess should start')
    monkeypatch.setattr(module.subprocess, 'run', forbidden)
    assert module.main([]) == 2
    import inspect
    assert "'evaluate'" not in inspect.getsource(module.main)


def test_failed_dashboard_entry_does_not_invent_results(tmp_path):
    module = load_script('build_phase4_data.py')
    folder = tmp_path / 'model_training/artifacts/track_ot/track_ot_pp_fold1_full_s42'
    write_json(folder / 'FAILURE.json', {'error': 'Nonfinite gradient'})
    write_json(folder / 'epochs/0001.json', {'validation_ap': 0.7916339472, 'epoch': 1, 'components': {'brier': 0.1458}})
    entry = module.track_ot_registry_entry(tmp_path)
    assert entry['status'].startswith('FAILED')
    assert entry['validation_ap'] == 0.7916339472
    for key in ['average_precision', 'brier_score', 'ece', 'throughput_windows_sec', 'latency_ms', 'checkpoint_size_kb']:
        assert entry[key] is None


def test_weights_only_warm_start_checks_source_and_never_reuses_optimizer(small, tmp_path):
    from track_ot.runner import warm_start_weights
    from railvivid.training import save_checkpoint
    model = TrackOTPlus(small)
    write_json(tmp_path / 'run.json', {'data_fingerprint': 'data1', 'split': {'split_fingerprint': 'split1'}, 'experiment_fingerprint': 'old1'})
    write_json(tmp_path / 'bags.json', {'fingerprint': 'bags1'})
    normalizer = {'mean': [0]*6, 'std': [1]*6}
    write_json(tmp_path / 'normalizer.json', normalizer)
    save_checkpoint(tmp_path / 'best.pt', {'model': model.state_dict(), 'experiment_fingerprint': 'old1', 'epoch': 0})
    source = {'data': {'data_fingerprint': 'data1'}, 'split': {'split_fingerprint': 'split1'}, 'bag_fingerprint': 'bags1', 'normalizer': normalizer}
    result = warm_start_weights(model, tmp_path, source)
    assert result['weights_only'] and not result['optimizer_state_reused']
    source['data']['data_fingerprint'] = 'different'
    with pytest.raises(ArtifactError): warm_start_weights(model, tmp_path, source)
