"""Numerical/data-contract checks only; no optimizer steps or model fitting."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from railvivid.common import file_hash, write_csv, write_json
from railvivid.errors import ConfigurationError, DataIntegrityError, LeakageError, ArtifactError
from railvivid.splits import build_split, load_split
from track_ot.config import DEFAULTS, load_config, validate
from track_ot.cli import synthetic_batch
from track_ot.model import TrackOTPlus, MonotonePositionWarp
from track_ot.transport import unbalanced_transport, reject_transport, crossing_penalty
from track_ot.losses import objective
from track_ot.data import sequence_index, SequenceDataset
from track_ot.alignment import monotone_projection, evaluate_alignment, marker_pairs
from track_ot.runner import train, select_abstention


@pytest.fixture
def track_cfg(cfg):
    torch.set_num_threads(2)
    c = copy.deepcopy(DEFAULTS)
    c["sequence"].update(windows=5, max_frames=7)
    c["encoder"].update(dimension=24, heads=3, layers=1, sensor_width=4, sensor_depth=1, spectral_bins=8, patch_tokens=4, dropout=0.0)
    c["transport"].update(iterations=100)
    c["_base"] = cfg
    return c


def test_uot_masked_rows_gradients_and_diagonal():
    cost = torch.full((4, 4), 8.0); cost.fill_diagonal_(0.01); cost.requires_grad_()
    mask = torch.ones(4, 4, dtype=torch.bool); mask[-1] = False
    plan, residual = unbalanced_transport(cost, mask)
    assert plan[-1].eq(0).all() and torch.isfinite(plan).all() and residual < 0.005
    assert plan[0, 0] > 100 * plan[0, 1]
    plan.square().sum().backward()
    assert torch.isfinite(cost.grad).all()
    empty, _ = unbalanced_transport(cost.detach(), torch.zeros_like(mask))
    assert empty.eq(0).all()


def test_transport_rejection_and_failures():
    cost = torch.zeros(3, 4)
    plan, _ = reject_transport(cost, torch.zeros_like(cost, dtype=torch.bool))
    assert plan.eq(0).all()
    with pytest.raises(DataIntegrityError): unbalanced_transport(cost + float("nan"), torch.ones_like(cost, dtype=torch.bool))
    with pytest.raises(DataIntegrityError): reject_transport(cost, torch.ones_like(cost))
    assert crossing_penalty(torch.eye(4)) == 0
    assert crossing_penalty(torch.eye(4).flip(1)) > 0


def test_position_warp_is_bounded_monotone(track_cfg):
    warp = MonotonePositionWarp(24, track_cfg["transport"])
    positions = torch.tensor([-10.0, -3.0, -3.0, 1.0, 10.0])
    features = torch.randn(5, 24)
    with torch.no_grad(): warp.shift.bias.fill_(4); warp.rate.bias.fill_(2)
    corrected = warp(positions, features, torch.randn(24))
    assert torch.all(corrected[1:] >= corrected[:-1])
    assert torch.all(abs(corrected - positions) <= track_cfg["transport"]["max_drift_m"] + 1e-5)
    with pytest.raises(DataIntegrityError): warp(positions.flip(0), features, torch.randn(24))


@pytest.mark.parametrize("matching", ["transport", "distance", "attention", "vibration"])
def test_model_forward_backward_and_missing_modes(track_cfg, matching):
    track_cfg["transport"]["matching"] = matching
    model = TrackOTPlus(track_cfg)
    batch = synthetic_batch(track_cfg)
    output = model.details(batch)
    loss, components = objective(model, output, batch, track_cfg, torch.tensor(1.0))
    loss.backward()  # Verify derivatives; no optimizer is constructed or stepped.
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert output["anchor_logit"].shape == (2,) and "equivariance" in components
    model.eval()
    batch["vision_mask"][:] = False
    with torch.inference_mode(): fallback = model.details(batch)
    assert torch.equal(fallback["instance_logits"], fallback["vibration_logits"])
    batch["vision_mask"][:] = True; batch["vibration_present"][:] = False
    if matching == "vibration":
        with pytest.raises(DataIntegrityError, match="Vibration-only"): model.details(batch)
        return
    with torch.inference_mode(): vision = model.details(batch)
    assert torch.isfinite(vision["anchor_logit"]).all()
    batch["vision_mask"][:] = False
    with pytest.raises(DataIntegrityError): model.details(batch)


def test_cycle_graph_has_finite_gradients(track_cfg):
    model = TrackOTPlus(track_cfg)
    batch = synthetic_batch(track_cfg)
    partner = synthetic_batch(track_cfg)
    for k, v in partner.items(): batch["partner_" + k] = v
    batch["partner_available"] = torch.ones(2, dtype=torch.bool)
    output = model.details(batch)
    loss, components = objective(model, output, batch, track_cfg, torch.tensor(1.0))
    loss.backward()
    assert np.isfinite(components["cycle"]) and components["recurrence"] >= 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_sequence_union_preserves_partition_and_labels(track_cfg, cfg, prepared):
    build_split(cfg); frame, _, _ = load_split(cfg)
    track_cfg["encoder"]["patch_pooling"] = False
    source, bags, digest = sequence_index(frame, track_cfg)
    assert [r["window_id"] for r in bags] == frame.window_id.tolist()
    for bag in bags:
        rows = source.iloc[bag["context_indices"]]
        assert rows.partition.nunique() == 1 and rows.run_id.nunique() == 1
        assert bag["anchor_index"] in bag["context_indices"]
    normalizer = {"mean": [0] * 6, "std": [1] * 6}
    vision = SimpleNamespace(lookup={}, features=np.zeros((0, 12)), quality=np.zeros((0, 2)))
    view = SequenceDataset(prepared, source, [b for b in bags if b["partition"] == "train"], normalizer, track_cfg, vision, augment=True)
    item = view[0]
    assert item["label"].item() == view.frame.iloc[0].label
    assert item["waveform"].shape == (5, 6, 128)
    assert "site_id" not in item and "latitude" not in item
    assert item["wave_mask"][item["anchor_slot"]]
    for index, partners in view.partners.items():
        assert all(view.frame.iloc[p].run_id != view.frame.iloc[index].run_id for p in partners)


def test_shared_candidates_are_rejected(track_cfg, cfg, prepared):
    build_split(cfg); frame, _, _ = load_split(cfg)
    a = frame.index[frame.partition.eq("train")][0]; b = frame.index[frame.partition.eq("test")][0]
    for i in [a, b]: frame.at[i, "candidate_frame_ids"] = ["same-content"]
    with pytest.raises(LeakageError): sequence_index(frame, track_cfg)


def test_training_gate_precedes_resource_access(track_cfg, monkeypatch):
    with pytest.raises(ConfigurationError, match="on-training-device"): train(track_cfg)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ConfigurationError, match="CUDA"): train(track_cfg, on_training_device=True)


def test_config_and_planning(tmp_path):
    config_path = Path(__file__).parents[1] / "configs/track_ot_plus_plus.json"
    c = load_config(config_path)
    assert c["_base"]["split"]["test_sites"] == [1, 2, 3]
    from track_ot.campaign import plan, load_jobs
    result = plan(c, tmp_path / "plan", variants=["full", "no_cycle"], folds=[1], seeds=[42])
    assert result["jobs"] == 2 and result["training_started"] is False
    jobs = load_jobs(result["plan"])
    assert jobs[1][1]["loss"]["cycle_weight"] == 0
    p = Path(result["plan"]).parent / jobs[0][0]["config"]
    p.write_text("{}")
    with pytest.raises(DataIntegrityError): load_jobs(result["plan"])
    c = copy.deepcopy(DEFAULTS); c["training"]["device"] = "cpu"
    with pytest.raises(ConfigurationError): validate(c)


def test_alignment_frozen_checks_and_missing_coverage(tmp_path):
    predicted = pd.DataFrame({"run_id": ["r", "r"], "frame_id": ["r/a", "r/b"], "position_m": [1.0, 9.0], "uncertainty_m": [2.0, 2.0]})
    references = pd.DataFrame({"marker_id": ["m1", "m2", "m3"], "run_id": ["r"] * 3,
                               "frame_id": ["r/a", "r/b", "r/c"], "chainage_m": [2.0, 8.0, 15.0],
                               "annotation_uncertainty_m": [0.5] * 3, "role": ["check"] * 3, "source": ["manual evaluator"] * 3})
    p = tmp_path / "frame_alignment.csv"; refs = tmp_path / "markers.csv"
    write_csv(p, predicted); write_csv(refs, references)
    write_json(tmp_path / "FROZEN.json", {"predictions_sha256": file_hash(p), "fit_frame_ids": []})
    metrics = evaluate_alignment(p, refs, tmp_path / "evaluation")
    assert metrics["mae_m"] == 1.0 and metrics["prediction_coverage"] == 2 / 3
    assert metrics["missing_frame_ids"] == ["r/c"]
    write_json(tmp_path / "FROZEN.json", {"predictions_sha256": file_hash(p), "fit_frame_ids": ["r/a"]})
    with pytest.raises(LeakageError): evaluate_alignment(p, refs, tmp_path / "bad")
    references["role"] = "fit"
    with pytest.raises(LeakageError): marker_pairs(predicted, references, "check")
    assert np.all(np.diff(monotone_projection([0, 3, 2, 5])) >= 0)


def test_abstention_cannot_hide_primary_failures():
    frame = pd.DataFrame({"label": [0, 1, 0, 1], "probability": [0.1, 0.9, 0.6, 0.4]})
    policy = select_abstention(frame, 0.5, {"abstention_target_error": 0.0, "abstention_min_samples": 2})
    assert policy["validation_coverage"] == 0.5 and policy["validation_error"] == 0


def test_patch_cache_and_frozen_evaluation_integration(track_cfg, cfg, prepared, tmp_path):
    """Synthetic frozen random weights exercise evaluation, never training."""
    from railvivid.common import read_json, save_array
    from railvivid.vision import vision_path, vision_identity
    from railvivid.training import save_checkpoint
    from track_ot.runner import resources, evaluate, export_model_alignment, verify_run
    cfg["split"]["cohort"] = "paired"
    cfg["model"]["kind"] = "fusion"
    windows = pd.read_csv(prepared / "windows.csv")
    frames = pd.DataFrame({"frame_id": [f"{r.run_id}/Frames/{r.start_idx}.jpg" for r in windows.itertuples()],
                           "run_id": windows.run_id, "chainage_m": windows.chainage_m, "nominal_chainage_m": windows.chainage_m,
                           "timestamp_ns": windows.timestamp_ns, "rejection_reason": "", "reference_distance_m": 0.0,
                           "image_sha256": [f"unique-{i}" for i in range(len(windows))]})
    write_csv(prepared / "frames.csv", frames)
    meta = read_json(prepared / "READY.json"); meta["artifacts"]["frames.csv"] = file_hash(prepared / "frames.csv"); write_json(prepared / "READY.json", meta)
    build_split(cfg)
    folder = vision_path(cfg); folder.mkdir(parents=True)
    rng = np.random.default_rng(11)
    write_csv(folder / "frames.csv", frames)
    save_array(folder / "embeddings.npy", rng.normal(size=(len(frames), 12)).astype("float32"))
    save_array(folder / "patches.npy", rng.normal(size=(len(frames), 9, 12)).astype("float16"))
    save_array(folder / "quality.npy", np.ones((len(frames), 2), dtype="float32"))
    write_json(folder / "READY.json", {"cache_fingerprint": vision_identity(cfg), "artifacts": {p.name: file_hash(p) for p in folder.iterdir()}})
    normalizer = {"mean": [0] * 6, "std": [1] * 6}
    track_cfg["experiment"].update(name="frozen_fixture", output=str(tmp_path / "experiments"))
    r = resources(track_cfg, normalizer=normalizer)
    destination = Path(track_cfg["experiment"]["output"]) / "frozen_fixture"; destination.mkdir(parents=True)
    model = TrackOTPlus(track_cfg).eval()
    save_checkpoint(destination / "best.pt", {"model": model.state_dict()})
    calibration = {"experiment_fingerprint": r["identity"], "temperature": 1.0, "threshold": 0.5, "abstention": {"confidence_cutoff": 0.0}}
    write_json(destination / "normalizer.json", normalizer); write_json(destination / "calibration.json", calibration)
    for name in ["config.json", "base_config.json", "bags.json", "run.json"]: write_json(destination / name, {"synthetic_fixture": True})
    write_json(destination / "DONE.json", {"experiment_fingerprint": r["identity"], "checkpoint_sha256": file_hash(destination / "best.pt"),
               "artifacts": {p.name: file_hash(p) for p in destination.iterdir()}})
    result = evaluate(track_cfg, device="cpu")
    assert result["metrics"]["observations"] == 6
    output = pd.read_csv(Path(result["evaluation"]) / "predictions.csv")
    assert output.window_id.tolist() == r["datasets"]["test"].frame.window_id.tolist()
    assert result["metrics"]["selective"]["coverage"] == 1.0
    exported = export_model_alignment(track_cfg, "train", tmp_path / "train_positions", device="cpu")
    assert exported["predicted_frames"] > 0 and exported["training_started"] is False
    from track_ot.alignment import calibrate_intervals
    train_path = Path(exported["output"]) / "frame_alignment.csv"
    training_positions = pd.read_csv(train_path).head(3)
    reference = pd.DataFrame({"marker_id": [f"synthetic-train-{i}" for i in range(len(training_positions))],
                              "run_id": training_positions.run_id, "frame_id": training_positions.frame_id,
                              "chainage_m": training_positions.position_m + 1.0,
                              "annotation_uncertainty_m": 0.2, "role": "calibration", "source": "synthetic test fixture"})
    reference_file = tmp_path / "train_references.csv"; write_csv(reference_file, reference)
    interval = calibrate_intervals(cfg, train_path, reference_file, tmp_path / "interval", coverage=0.5)
    test_path = Path(result["evaluation"]) / "frame_alignment.csv"; test_positions = pd.read_csv(test_path).head(2)
    check = pd.DataFrame({"marker_id": [f"synthetic-test-{i}" for i in range(len(test_positions))],
                          "run_id": test_positions.run_id, "frame_id": test_positions.frame_id,
                          "chainage_m": test_positions.position_m + 1.0,
                          "annotation_uncertainty_m": 0.2, "role": "check", "source": "synthetic test fixture"})
    check_file = tmp_path / "check.csv"; write_csv(check_file, check)
    checked = evaluate_alignment(test_path, check_file, tmp_path / "checked_intervals", tmp_path / "interval/calibration.json")
    assert checked["intervals_empirically_calibrated"] and checked["reference_centre_coverage"] == 1
    reference["frame_id"] = test_positions.iloc[0].frame_id
    write_csv(tmp_path / "leaky_references.csv", reference)
    with pytest.raises(LeakageError): calibrate_intervals(cfg, train_path, tmp_path / "leaky_references.csv", tmp_path / "leaky_interval", coverage=0.5)
    write_json(destination / "calibration.json", {"tampered": True})
    with pytest.raises(DataIntegrityError): verify_run(destination)


def test_optical_flow_progress_on_synthetic_frames(tmp_path):
    pytest.importorskip("cv2")
    from PIL import Image
    from track_ot.alignment import flow_progress
    rng = np.random.default_rng(23)
    image = rng.integers(0, 255, size=(300, 600), dtype="uint8")
    rows = []
    for i in range(3):
        name = f"image_{i}.jpg"; Image.fromarray(np.roll(image, i * 3, axis=0)).save(tmp_path / name)
        rows.append({"relative_path": name, "frame_id": f"r/{name}", "timestamp_ns": i * 50_000_000})
    progress, failures = flow_progress({"data": {"root": str(tmp_path)}}, pd.DataFrame(rows))
    assert not failures and np.all(np.diff(progress) > 0)


def test_invalid_stress_is_rejected():
    from track_ot.runner import validate_corruption
    for stress in [{"missing_channels": [0]}, {"noise_std_normalized": -1}, {"missing_vision": "yes"}, {"alignment_prior_shift_m": float("nan")}]:
        with pytest.raises(ConfigurationError): validate_corruption(stress)


def test_generic_transport_control(track_cfg):
    from railvivid.config import merge
    from track_ot.campaign import VARIANTS
    c = merge(copy.deepcopy(track_cfg), copy.deepcopy(VARIANTS["generic_ot"]))
    model = TrackOTPlus(c).eval()
    with torch.inference_mode(): output = model.details(synthetic_batch(c))
    assert torch.isfinite(output["anchor_logit"]).all() and output["drift_m"].eq(0).all()
