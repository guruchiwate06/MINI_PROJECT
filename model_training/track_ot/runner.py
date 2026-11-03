"""Explicit CUDA-only fitting, frozen evaluation and provenance-aware recovery."""
from __future__ import annotations

import copy
import logging
import os
import tempfile
import time
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from torch.utils.data import DataLoader

from railvivid.common import environment, file_hash, fingerprint, read_json, seed_everything, select_device, utc_now, write_csv, write_json
from railvivid.dataset import training_normalizer
from railvivid.errors import ArtifactError, ConfigurationError, DataIntegrityError, TrainingError
from railvivid.metrics import classification_metrics, event_metrics, fit_temperature, select_threshold, clustered_ap_interval
from railvivid.prepare import load_prepared
from railvivid.splits import load_split
from railvivid.training import experiment_lock, load_checkpoint, save_checkpoint, rng_state, restore_rng, vision_for
from .config import identity, public_config, source_fingerprint
from .data import SequenceDataset, split_datasets
from .losses import objective
from .model import TrackOTPlus
from .numerics import gradient_report, strict_clip_gradients, precision

LOG = logging.getLogger("track_ot")
STRESSES = {"missing_vision": {"missing_vision": True}, "missing_vibration": {"missing_vibration": True},
            "missing_axles": {"missing_channels": [2, 4]}, "noise_0p1": {"noise_std_normalized": 0.1},
            "noise_0p5": {"noise_std_normalized": 0.5},
            **{f"shift_{'plus' if sign > 0 else 'minus'}{distance}": {"alignment_prior_shift_m": sign * distance}
               for sign in [-1, 1] for distance in [5, 10, 15]}}


def run_path(c): return Path(c["experiment"]["output"]) / c["experiment"]["name"]


def training_view(r, c):
    full = r["datasets"]["train"]
    permitted = [x for x in full.records if x["bag_label"] >= 0]
    dataset = SequenceDataset(r["root"], full.source, permitted, r["normalizer"], c, r["vision"], augment=True)
    labels = np.asarray([x["bag_label"] for x in permitted]) if c["loss"]["supervision"] == "mil" else full.frame.loc[full.frame.eligible_supervised, "label"].to_numpy()
    if set(labels.tolist()) != {0, 1}: raise DataIntegrityError("Both training classes are required")
    return dataset, float((labels == 0).sum() / (labels == 1).sum())


def validate_corruption(corruption):
    import math
    allowed = {"missing_vision", "missing_vibration", "missing_channels", "noise_std_normalized", "alignment_prior_shift_m"}
    if not isinstance(corruption, dict) or set(corruption) - allowed: raise ConfigurationError("Unknown stress specification")
    for key in ["missing_vision", "missing_vibration"]:
        if key in corruption and not isinstance(corruption[key], bool): raise ConfigurationError(f"{key} must be boolean")
    channels = corruption.get("missing_channels", [])
    if not isinstance(channels, list) or any(not isinstance(k, int) or isinstance(k, bool) or k not in range(1, 7) for k in channels) or len(channels) != len(set(channels)):
        raise ConfigurationError("missing_channels must be unique IDs 1..6")
    for key in ["noise_std_normalized", "alignment_prior_shift_m"]:
        value = corruption.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value): raise ConfigurationError(f"Invalid {key}")
    if corruption.get("noise_std_normalized", 0) < 0: raise ConfigurationError("Noise scale cannot be negative")


def resources(c, verify=False, normalizer=None, augment=False, corruption=None):
    base = c["_base"]
    root, meta = load_prepared(base, verify=verify)
    frame, info, _ = load_split(base)
    vision = vision_for(base, frame)
    if normalizer is None: normalizer = training_normalizer(root, frame.loc[frame.partition.eq("train")], base["model"]["channels"])
    datasets, records, bag_hash = split_datasets(root, frame, normalizer, c, vision, augment=augment, corruption=corruption)
    ident = identity(c, info, bag_hash, fingerprint({"cache": vision.meta["cache_fingerprint"], "artifacts": vision.meta["artifacts"]}))
    return {"root": root, "data": meta, "frame": frame, "split": info, "vision": vision, "normalizer": normalizer,
            "datasets": datasets, "bags": records, "bag_fingerprint": bag_hash, "identity": ident}


def loader(dataset, c, shuffle=False, epoch=0):
    dataset.epoch = epoch
    return DataLoader(dataset, batch_size=c["training"]["batch_size"], shuffle=shuffle,
                      generator=torch.Generator().manual_seed(c["training"]["seed"] + epoch),
                      num_workers=c["training"]["workers"], persistent_workers=False,
                      pin_memory=torch.cuda.is_available() and c["training"]["device"].startswith("cuda") and sys.platform != "win32")


def to_device(batch, device): return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def predict(model, dataset, c, device, export_alignment=False):
    model.eval(); rows = []; aligned = {}
    if device.type == "cuda": torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in loader(dataset, c):
            output = model.details(to_device(batch, device))
            if not torch.isfinite(output["anchor_logit"]).all(): raise TrainingError("Nonfinite inference logits")
            logits = output["anchor_logit"].float().cpu().numpy()
            mass = output["match_mass"].float().cpu().numpy(); drift = output["drift_m"].float().cpu().numpy()
            residual = output["solver_residual"].float().cpu().numpy()
            location = output["correspondence_position_m"].float().cpu().numpy()
            variance = output["correspondence_variance_m2"].float().cpu().numpy()
            for j, index in enumerate(batch["index"].tolist()):
                valid = batch["wave_mask"][j].numpy(); imask = batch["vision_mask"][j].numpy()
                slot = int(batch["anchor_slot"][j])
                rows.append({"index": index, "logit": float(logits[j]), "match_mass": float(mass[j, slot]),
                             "mean_abs_drift_m": float(np.abs(drift[j, imask]).mean()) if imask.any() else 0,
                             "correspondence_mean_relative_m": float(location[j, slot]) if mass[j, slot] > 1e-5 else None,
                             "correspondence_std_m": float(np.sqrt(max(variance[j, slot], 0))) if mass[j, slot] > 1e-5 else None,
                             "solver_residual": float(residual[j]), "unmatched_instance_fraction": float((mass[j, valid] < 1e-5).mean())})
                if export_alignment and imask.any():
                    rec = dataset.records[index]
                    for k, fid in enumerate(rec["frame_ids"]):
                        if not imask[k]: continue
                        position = float(batch["anchor_chainage"][j]) + float(batch["image_positions"][j, k]) + float(drift[j, k])
                        count, total, squares = aligned.get(fid, (0, 0.0, 0.0))
                        aligned[fid] = count + 1, total + position, squares + position * position
    scores = pd.DataFrame(rows).sort_values("index")
    if len(scores) != len(dataset) or scores["index"].tolist() != list(range(len(dataset))): raise TrainingError("Incomplete or duplicated prediction coverage")
    result = dataset.frame.copy()
    for key in scores.columns:
        if key != "index": result[key] = scores[key].to_numpy()
    elapsed = time.perf_counter() - start
    result.attrs["inference"] = {"seconds_including_loader": elapsed, "windows": len(result), "windows_per_second": len(result) / max(elapsed, 1e-9),
                                  "device": str(device), "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    frames = []
    for fid, (count, total, squares) in aligned.items():
        mean = total / count; std = np.sqrt(max(squares / count - mean * mean, 0))
        frames.append({"frame_id": fid, "run_id": fid.split("/")[0], "position_m": mean,
                       "context_count": count, "position_std_across_contexts_m": std,
                       "uncertainty_m": c["transport"]["position_sigma_m"], "uncertainty_calibrated": False})
    return result, pd.DataFrame(frames, columns=["frame_id", "run_id", "position_m", "context_count", "position_std_across_contexts_m", "uncertainty_m", "uncertainty_calibrated"])


def select_abstention(frame, threshold, settings):
    member = frame.label.isin([0, 1]); f = frame.loc[member]
    confidence = np.abs(f.probability.to_numpy() - 0.5)
    error = (f.probability.to_numpy() >= threshold) != f.label.to_numpy()
    best = {"confidence_cutoff": 0.5000001, "validation_coverage": 0.0, "validation_error": None, "accepted": 0}
    for cutoff in np.unique(np.r_[0.0, confidence]):
        accepted = confidence >= cutoff
        count = int(accepted.sum())
        if count < settings["abstention_min_samples"]: continue
        risk = float(error[accepted].mean())
        if risk <= settings["abstention_target_error"] and count > best["accepted"]:
            best = {"confidence_cutoff": float(cutoff), "validation_coverage": count / len(f), "validation_error": risk, "accepted": count}
    return {**best, "target_error": settings["abstention_target_error"], "source": "validation only; empirical selective classification, no population risk guarantee"}


def calibrate(c, folder, validation, sites, ident):
    base = c["_base"]
    temperature = fit_temperature(validation.label, validation.logit) if base["evaluation"]["temperature_scaling"] else 1.0
    validation = validation.copy(); validation["probability"] = expit(validation.logit / temperature)
    threshold = select_threshold(validation, sites, base["evaluation"])
    calibration = {"experiment_fingerprint": ident, "temperature": temperature, "threshold": threshold,
                   "threshold_policy": base["evaluation"]["threshold"], "source": "validation only",
                   "abstention": select_abstention(validation, threshold, c["evaluation"])}
    write_json(folder / "calibration.json", calibration)
    write_csv(folder / "validation_predictions.csv", validation.drop(columns=["candidate_frame_ids", "candidate_offsets_m"], errors="ignore"))
    write_json(folder / "validation_metrics.json", classification_metrics(validation.label, validation.probability, threshold))
    return calibration


def warm_start_weights(model, previous, current):
    previous = Path(previous).resolve()
    run = read_json(previous / "run.json")
    if run.get("data_fingerprint") != current["data"]["data_fingerprint"] or run["split"]["split_fingerprint"] != current["split"]["split_fingerprint"]:
        raise ArtifactError("Warm-start source used different data/split")
    if read_json(previous / "bags.json").get("fingerprint") != current["bag_fingerprint"]:
        raise ArtifactError("Warm-start source used different sequence bags")
    if fingerprint(read_json(previous / "normalizer.json")) != fingerprint(current["normalizer"]):
        raise ArtifactError("Warm-start normalization differs")
    checkpoint = previous / "best.pt"
    state = load_checkpoint(checkpoint)
    if state.get("experiment_fingerprint") != run["experiment_fingerprint"]: raise ArtifactError("Warm-start checkpoint identity does not match its run")
    model.load_state_dict(state["model"], strict=True)
    return {"source_experiment": run["experiment_fingerprint"], "checkpoint_sha256": file_hash(checkpoint),
            "source_epoch": state["epoch"] + 1, "weights_only": True, "optimizer_state_reused": False,
            "interpretation": "New recovery experiment; not a bitwise resume or independent replication"}


def train(c, on_training_device=False, resume=False, warm_start=None):
    if not on_training_device: raise ConfigurationError("Training requires --on-training-device on the RTX machine; planning/smoke never fit")
    if resume and warm_start: raise ConfigurationError("Choose strict resume or a NEW warm-start experiment")
    device = select_device(c["training"]["device"], require_cuda=True)
    seed_everything(c["training"]["seed"], deterministic=True)
    r = resources(c, verify=True, augment=True)
    folder = run_path(c); ident = r["identity"]
    if resume:
        if read_json(folder / "run.json")["experiment_fingerprint"] != ident: raise ArtifactError("Resume data/config/code/cache identity changed")
        if (folder / "DONE.json").exists(): raise ArtifactError("Run already complete; no refit performed")
    else:
        if folder.exists(): raise ArtifactError("Experiment exists; use a new name or explicitly resume")
        folder.mkdir(parents=True)
        write_json(folder / "config.json", public_config(c)); write_json(folder / "base_config.json", c["_base"])
        write_json(folder / "environment.json", {**environment(), "track_ot_source_hash": source_fingerprint()})
        write_json(folder / "normalizer.json", r["normalizer"])
        write_json(folder / "bags.json", {"fingerprint": r["bag_fingerprint"], "bags": r["bags"]})
        write_json(folder / "run.json", {"started": utc_now(), "experiment_fingerprint": ident, "split": r["split"],
                                         "data_fingerprint": r["data"]["data_fingerprint"], "physical_alignment_validated": False,
                                         "evaluation_cohort": fingerprint(sorted(r["datasets"]["test"].frame.window_id.tolist()))})
    with experiment_lock(folder):
        try:
            training, ratio = training_view(r, c)
            pos_weight = torch.tensor(ratio, device=device)
            model = TrackOTPlus(c).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=c["training"]["learning_rate"], weight_decay=c["training"]["weight_decay"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c["training"]["epochs"])
            amp, amp_dtype = precision(c, device)
            scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16, init_scale=c["training"]["initial_loss_scale"])
            if warm_start:
                parent = warm_start_weights(model, warm_start, r)
                write_json(folder / "warm_start.json", parent)
            write_json(folder / "precision.json", {"autocast": amp, "dtype": str(amp_dtype) if amp else "torch.float32", "loss_scaling": scaler.is_enabled()})
            first, best, bad = 0, -float("inf"), 0
            if resume:
                state = load_checkpoint(folder / "last.pt")
                if state["experiment_fingerprint"] != ident: raise ArtifactError("Checkpoint identity mismatch")
                model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
                scheduler.load_state_dict(state["scheduler"]); scaler.load_state_dict(state["scaler"]); restore_rng(state["rng"])
                first, best, bad = state["epoch"] + 1, state["best"], state["bad_epochs"]
            started = time.perf_counter(); durations = []
            for epoch in range(first, c["training"]["epochs"]):
                model.train(); batches = loader(training, c, shuffle=True, epoch=epoch)
                if not len(batches): raise DataIntegrityError("Empty training loader")
                optimizer.zero_grad(set_to_none=True); seen, total, updates = 0, 0.0, 0; components = {}
                torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device); begin = time.perf_counter()
                last_batch_indices = []; component_snapshot = {}; last_gradient_norm = None
                for step, batch in enumerate(batches):
                    last_batch_indices = batch["index"].tolist()
                    batch = to_device(batch, device); size = len(batch["index"])
                    group_start = (step // c["training"]["accumulation_steps"]) * c["training"]["accumulation_steps"]
                    denominator = min(c["training"]["batch_size"] * c["training"]["accumulation_steps"], len(training) - group_start * c["training"]["batch_size"])
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
                        with torch.profiler.record_function("track_ot/main_forward"):
                            out = model.details(batch)
                        with torch.profiler.record_function("track_ot/objective"):
                            loss, terms = objective(model, out, batch, c, pos_weight)
                    component_snapshot = terms
                    if not torch.isfinite(loss): raise TrainingError(f"Nonfinite loss at epoch {epoch + 1}, batch {step + 1}")
                    with torch.profiler.record_function("track_ot/backward"):
                        scaler.scale(loss * size / denominator).backward()
                    if (step + 1) % c["training"]["accumulation_steps"] == 0 or step + 1 == len(batches):
                        scaler.unscale_(optimizer)
                        last_gradient_norm = float(strict_clip_gradients(model, c["training"]["gradient_clip"]))
                        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); updates += 1
                    total += float(loss.detach()) * size; seen += size
                    for k, v in terms.items(): components[k] = components.get(k, 0.0) + v * size
                    if (step + 1) % max(1, len(batches) // 4) == 0:
                        LOG.info("Epoch %d batch %d/%d loss %.4f", epoch + 1, step + 1, len(batches), float(loss.detach()))
                scheduler.step(); torch.cuda.synchronize(device)
                train_seconds = time.perf_counter() - begin
                training_peak_bytes = torch.cuda.max_memory_allocated(device)
                validation_begin = time.perf_counter()
                validation, _ = predict(model, r["datasets"]["validation"], c, device)
                validation_seconds = time.perf_counter() - validation_begin
                write_csv(folder / f"epochs/{epoch + 1:04d}_validation_predictions.csv", validation.drop(columns=["candidate_frame_ids", "candidate_offsets_m"], errors="ignore"))
                score = classification_metrics(validation.label, expit(validation.logit))["average_precision"]
                if score is None: raise DataIntegrityError("Validation AP undefined")
                improved = score > best + c["training"]["min_delta"]
                if improved: best, bad = score, 0
                else: bad += 1
                state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                         "scaler": scaler.state_dict(), "rng": rng_state(), "epoch": epoch, "best": best, "bad_epochs": bad,
                         "experiment_fingerprint": ident}
                save_checkpoint(folder / "last.pt", state)
                if improved: save_checkpoint(folder / "best.pt", state)
                seconds = time.perf_counter() - begin; durations.append(seconds)
                progress = {"epoch": epoch + 1, "training_loss": total / seen, "validation_ap": score, "best_ap": best,
                            "components": {k: v / seen for k, v in components.items()}, "training_seconds": train_seconds,
                            "validation_seconds": validation_seconds, "training_bags": seen, "training_bags_per_second": seen / max(train_seconds, 1e-9),
                            "last_gradient_norm_before_clip": last_gradient_norm, "validation_inference": validation.attrs.get("inference", {}),
                            "epoch_seconds": seconds, "elapsed_seconds": time.perf_counter() - started,
                            "estimated_remaining_seconds": np.mean(durations) * (c["training"]["epochs"] - epoch - 1),
                            "optimizer_updates": updates, "peak_cuda_allocated_bytes": training_peak_bytes,
                            "validation_peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)}
                write_json(folder / f"epochs/{epoch + 1:04d}.json", progress); write_json(folder / "progress.json", progress)
                LOG.info("Epoch %d AP %.4f best %.4f ETA %.1f min", epoch + 1, score, best, progress["estimated_remaining_seconds"] / 60)
                if bad >= c["training"]["patience"]: break
            model.load_state_dict(load_checkpoint(folder / "best.pt")["model"])
            validation, _ = predict(model, r["datasets"]["validation"], c, device)
            calibrate(c, folder, validation, pd.read_csv(r["root"] / "sites.csv"), ident)
            write_json(folder / "DONE.json", {"completed": utc_now(), "experiment_fingerprint": ident,
                                            "checkpoint_sha256": file_hash(folder / "best.pt"), "test_evaluated": False,
                                            "artifacts": {name: file_hash(folder / name) for name in ["best.pt", "normalizer.json", "calibration.json", "config.json", "base_config.json", "bags.json", "run.json", "precision.json", "warm_start.json"] if (folder / name).exists()}})
            return {"run": str(folder), "complete": True, "test_evaluated": False}
        except BaseException as e:
            message = str(e)
            if isinstance(e, torch.cuda.OutOfMemoryError): message += " Reduce explicit sequence/batch budgets in a NEW experiment; no samples or modules were silently removed."
            write_json(folder / "FAILURE.json", {"time": utc_now(), "type": type(e).__name__, "error": message,
                                               "last_completed_checkpoint": str(folder / "last.pt") if (folder / "last.pt").exists() else None})
            if "model" in locals():
                try:
                    write_json(folder / "gradient_failure.json", {"epoch": epoch + 1 if "epoch" in locals() else None,
                        "batch_indices": locals().get("last_batch_indices", []), "loss_components": locals().get("component_snapshot", {}),
                        "source_window_ids": [training.records[i]["window_id"] for i in locals().get("last_batch_indices", [])] if "training" in locals() else [],
                        "gradients": gradient_report(model), "precision": str(locals().get("amp_dtype")),
                        "note": "No nonfinite gradients were clipped or used for an optimizer update"})
                except (RuntimeError, OSError) as diagnostic_error:
                    LOG.error("Gradient diagnostic failed after original %s: %s", type(e).__name__, diagnostic_error)
            if isinstance(e, (KeyboardInterrupt, SystemExit, ArtifactError, ConfigurationError, DataIntegrityError, TrainingError)): raise
            raise TrainingError(message) from e


def verify_run(folder):
    done = read_json(folder / "DONE.json")
    required = {"best.pt", "normalizer.json", "calibration.json", "config.json", "base_config.json", "bags.json", "run.json"}
    if not required <= set(done.get("artifacts", {})): raise ArtifactError("Completed run lacks required artifact provenance")
    for name, digest in done["artifacts"].items():
        if Path(name).name != name: raise DataIntegrityError("Unsafe run artifact path")
        if file_hash(folder / name) != digest: raise DataIntegrityError(f"Completed run artifact changed: {name}")
    if done["checkpoint_sha256"] != file_hash(folder / "best.pt"): raise DataIntegrityError("Checkpoint hash mismatch")
    return done


def evaluate(c, partition="test", device="cuda", corruption=None, tag="clean"):
    validate_corruption(corruption or {})
    if partition not in ["validation", "test"]: raise ConfigurationError("Evaluation must be validation or test")
    if not tag or Path(tag).name != tag or tag in [".", ".."]: raise ConfigurationError("Unsafe evaluation tag")
    folder = run_path(c); done = verify_run(folder)
    if file_hash(folder / "best.pt") != done["checkpoint_sha256"]: raise DataIntegrityError("Trained checkpoint changed")
    r = resources(c, verify=True, normalizer=read_json(folder / "normalizer.json"), corruption=corruption)
    if r["identity"] != done["experiment_fingerprint"]: raise ArtifactError("Evaluation source/config/data/cache differs from training")
    calibration = read_json(folder / "calibration.json")
    if calibration["experiment_fingerprint"] != r["identity"]: raise ArtifactError("Calibration belongs to another experiment")
    destination = folder / "evaluations" / f"{partition}_{tag}"
    if destination.exists(): raise ArtifactError("Evaluation exists; no overwrite allowed")
    dev = select_device(device, False); model = TrackOTPlus(c).to(dev)
    model.load_state_dict(load_checkpoint(folder / "best.pt")["model"])
    predictions, alignment = predict(model, r["datasets"][partition], c, dev, export_alignment=True)
    predictions["probability"] = expit(predictions.logit / calibration["temperature"])
    predictions["abstain"] = np.abs(predictions.probability - 0.5) < calibration["abstention"]["confidence_cutoff"]
    settings = c["_base"]["evaluation"]
    metrics = classification_metrics(predictions.label, predictions.probability, calibration["threshold"], settings["ece_bins"])
    metrics["events"], events = event_metrics(predictions, pd.read_csv(r["root"] / "sites.csv"), calibration["threshold"], settings["event_tolerance_m"], settings["event_merge_gap_m"])
    metrics["ap_interval"] = clustered_ap_interval(predictions, settings["bootstrap_repeats"], settings["bootstrap_seed"])
    known = predictions.label.isin([0, 1]); accepted = known & ~predictions.abstain
    metrics["selective"] = {"coverage": float(accepted.sum() / max(known.sum(), 1)), "accepted": int(accepted.sum()),
                            "error": float(((predictions.loc[accepted, "probability"] >= calibration["threshold"]) != predictions.loc[accepted, "label"]).mean()) if accepted.any() else None,
                            "rule_source": "frozen validation calibration; abstained windows remain in primary event/FA metrics"}
    metrics["inference"] = predictions.attrs.get("inference", {})
    metrics["physical_alignment_validated"] = False
    metrics["stress"] = corruption or {}
    final_destination = destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix=destination.name + ".partial_", dir=destination.parent))
    if destination.resolve().parent != final_destination.resolve().parent: raise ArtifactError("Evaluation staging escaped its output directory")
    write_csv(destination / "predictions.csv", predictions.drop(columns=["candidate_frame_ids", "candidate_offsets_m"], errors="ignore"))
    write_csv(destination / "frame_alignment.csv", alignment); write_csv(destination / "events.csv", events)
    subgroups = []
    for key in ["run_id", "site_id", "direction", "target_speed_kmh"]:
        for value, g in predictions.groupby(key):
            row = classification_metrics(g.label, g.probability, calibration["threshold"]) if g.label.isin([0, 1]).any() else {}
            subgroups.append({"group": key, "value": value, **{k: v for k, v in row.items() if not isinstance(v, (dict, list))}})
    write_csv(destination / "subgroups.csv", pd.DataFrame(subgroups)); write_json(destination / "metrics.json", metrics)
    write_json(destination / "provenance.json", {"experiment_fingerprint": r["identity"], "partition": partition,
                                               "cohort_fingerprint": fingerprint(sorted(predictions.window_id.tolist())), "checkpoint_sha256": done["checkpoint_sha256"],
                                               "split_fingerprint": r["split"]["split_fingerprint"], "test_access_disclosed": partition == "test",
                                               "alignment_predictions_are_not_ground_truth": True})
    write_json(destination / "READY.json", {"completed": utc_now(), "artifacts": {p.name: file_hash(p) for p in destination.iterdir() if p.is_file()}})
    os.replace(destination, final_destination)
    return {"evaluation": str(final_destination), "metrics": metrics}


def robustness(c, cases, partition="test", device="cuda"):
    if not cases or len(set(cases)) != len(cases) or set(cases) - set(STRESSES): raise ConfigurationError("Unknown/duplicate stress cases")
    return {"evaluations": [evaluate(c, partition, device, STRESSES[name], name) for name in cases],
            "calibration": "unchanged validation temperature and threshold for all stresses"}


def export_model_alignment(c, partition, output, device="cuda"):
    from .alignment import fresh_directory
    folder = run_path(c); done = verify_run(folder)
    if file_hash(folder / "best.pt") != done["checkpoint_sha256"]: raise DataIntegrityError("Trained checkpoint changed")
    r = resources(c, verify=True, normalizer=read_json(folder / "normalizer.json"))
    if r["identity"] != done["experiment_fingerprint"]: raise ArtifactError("Frozen model identity changed")
    dev = select_device(device, False); model = TrackOTPlus(c).to(dev)
    model.load_state_dict(load_checkpoint(folder / "best.pt")["model"])
    _, alignment = predict(model, r["datasets"][partition], c, dev, export_alignment=True)
    destination = fresh_directory(output); write_csv(destination / "frame_alignment.csv", alignment)
    write_json(destination / "FROZEN.json", {"created": utc_now(), "method": "track_ot_pp", "estimator_fingerprint": r["identity"],
                "checkpoint_sha256": done["checkpoint_sha256"], "partition": partition, "fit_frame_ids": [],
                "predictions_sha256": file_hash(destination / "frame_alignment.csv"), "withheld_references_read": False,
                "physical_accuracy_verified": False, "uncertainty_calibrated": False})
    return {"output": str(destination), "predicted_frames": len(alignment), "training_started": False}
