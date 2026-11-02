"""Separate endpoint-based alignment and withheld-marker evaluation.

Optical-flow magnitude is a progress proxy, not calibrated metric visual
odometry. Metric scale is supplied by permitted route endpoint anchors.
"""
from __future__ import annotations

import html
from pathlib import Path
import numpy as np
import pandas as pd

from railvivid.common import atomic_write, file_hash, fingerprint, read_json, require_columns, utc_now, write_csv, write_json
from railvivid.errors import ArtifactError, ConfigurationError, DataIntegrityError, LeakageError
from railvivid.prepare import load_prepared
from railvivid.splits import load_split


def fresh_directory(path):
    path = Path(path)
    if path.exists(): raise ArtifactError(f"Destination exists: {path}")
    path.mkdir(parents=True)
    return path


def false_flags(values):
    return values.astype(str).str.strip().str.lower().isin(["false", "0", "no"])


def template(base, output):
    root, _ = load_prepared(base)
    frames = pd.read_csv(root / "frames.csv", keep_default_na=False)
    accepted = frames.loc[frames.rejection_reason.eq("")].copy()
    output = fresh_directory(output)
    # No anomaly-site coordinates are exported into model-input annotations.
    write_csv(output / "frame_inventory.csv", accepted.drop(columns=["chainage_m", "nominal_chainage_m"], errors="ignore"))
    write_csv(output / "endpoint_anchors.csv", pd.DataFrame(columns=["run_id", "frame_id", "chainage_m", "role", "source", "is_anomaly_marker"]))
    write_csv(output / "withheld_markers.csv", pd.DataFrame(columns=["marker_id", "run_id", "frame_id", "chainage_m", "annotation_uncertainty_m", "role", "source"]))
    evaluator = output / "evaluator_only"; evaluator.mkdir()
    write_csv(evaluator / "published_site_reference.csv", pd.read_csv(root / "sites.csv"))
    write_json(output / "PROTOCOL.json", {"created": utc_now(), "training_started": False,
                "endpoint_roles": "fit; independently identify start/end encounter frames and metric positions",
                "evaluation_roles": "check; a separate evaluator annotates surveyed cone encounters and uncertainty",
                "no_ground_truth_invented": True,
                "evaluator_reference": "evaluator_only/published_site_reference.csv; existing published targets projected onto the reference route, never alignment inputs for held-out sites",
                "warning": "Visible cone does not mean camera is physically at the cone. Account for camera viewpoint, sensor offset and encounter uncertainty. Published rounded coordinates do not certify centimetre accuracy."})
    return {"output": str(output), "accepted_frames": len(accepted), "annotations_filled": False}


def monotone_projection(values):
    """Pooled adjacent violators, equal weights. No statistical model fitting."""
    blocks = []
    for value in values:
        blocks.append([float(value), 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            b = blocks.pop(); a = blocks.pop(); count = a[1] + b[1]
            blocks.append([(a[0] * a[1] + b[0] * b[1]) / count, count])
    return np.concatenate([np.full(count, value) for value, count in blocks])


def flow_progress(base, frames, roi=(0.2, 0.45, 0.8, 1.0), low_feature_policy="abort"):
    try: import cv2
    except ImportError as e: raise ConfigurationError("Flow alignment requires the optional opencv-python-headless package on the processing device") from e
    if low_feature_policy not in ["abort", "timestamp"]: raise ConfigurationError("Unknown low-feature policy")
    previous = None; progress = [0.0]; failures = []
    times = frames.timestamp_ns.to_numpy(dtype=np.int64)
    nominal_dt = np.median(np.diff(times)) / 1e9
    steps = []
    for k, row in enumerate(frames.itertuples()):
        path = Path(base["data"]["root"]) / row.relative_path
        frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if frame is None: raise DataIntegrityError(f"Cannot decode alignment frame {path}")
        height, width = frame.shape
        image = frame[round(roi[1] * height):round(roi[3] * height), round(roi[0] * width):round(roi[2] * width)]
        if image.size == 0: raise DataIntegrityError("Empty optical-flow ROI")
        image = cv2.resize(image, (480, 240))
        if previous is not None:
            points = cv2.goodFeaturesToTrack(previous, maxCorners=250, qualityLevel=0.01, minDistance=7)
            displacement = None
            if points is not None and len(points) >= 8:
                next_points, status, _ = cv2.calcOpticalFlowPyrLK(previous, image, points, None)
                if next_points is not None and status is not None:
                    back, back_status, _ = cv2.calcOpticalFlowPyrLK(image, previous, next_points, None)
                    if back is not None and back_status is not None:
                        good = status.ravel().astype(bool) & back_status.ravel().astype(bool)
                        good &= np.linalg.norm((back - points).reshape(-1, 2), axis=1) < 1.5
                        if good.sum() >= 8:
                            displacement = float(np.median(np.linalg.norm((next_points - points).reshape(-1, 2)[good], axis=1)))
            if displacement is None or not np.isfinite(displacement):
                if low_feature_policy == "abort": raise DataIntegrityError(f"Insufficient flow support at {row.frame_id}; no silent alignment imputation")
                displacement = (np.median(steps[-50:]) if steps else 1.0) * ((times[k] - times[k - 1]) / 1e9) / max(nominal_dt, 1e-6)
                failures.append(row.frame_id)
            steps.append(max(displacement, 0.0)); progress.append(progress[-1] + steps[-1])
        previous = image
    if progress[-1] <= 0: raise DataIntegrityError("Flow supplies no route progress")
    return np.asarray(progress), failures


def estimate(base, anchors_path, output, method="flow", low_feature_policy="abort"):
    if method not in ["flow", "uniform", "nominal"]: raise ConfigurationError("Unknown alignment method")
    root, meta = load_prepared(base)
    anchors = pd.read_csv(anchors_path, keep_default_na=False)
    require_columns(anchors, ["run_id", "frame_id", "chainage_m", "role", "source", "is_anomaly_marker"], "endpoint anchors")
    if anchors.empty: raise DataIntegrityError("Endpoint anchors are empty; manual annotation is required")
    if not anchors.role.eq("fit").all() or not false_flags(anchors.is_anomaly_marker).all():
        raise LeakageError("Alignment input accepts only permitted non-anomaly fit anchors; withheld check references are forbidden")
    if anchors.frame_id.duplicated().any() or anchors.source.eq("").any(): raise DataIntegrityError("Duplicate/unattributed anchors")
    frames = pd.read_csv(root / "frames.csv", keep_default_na=False)
    frames = frames.loc[frames.rejection_reason.eq("")].copy()
    if not set(anchors.frame_id) <= set(frames.frame_id): raise DataIntegrityError("Anchor frame is absent/rejected")
    destination = fresh_directory(output)
    outputs, diagnostics = [], []
    try:
        for run, fit in anchors.groupby("run_id"):
            if len(fit) != 2: raise DataIntegrityError(f"{run}: endpoint protocol requires exactly two fit anchors")
            run_frames = frames.loc[frames.run_id.eq(run)].sort_values("timestamp_ns")
            if set(fit.frame_id) - set(run_frames.frame_id): raise DataIntegrityError("Anchor run/frame mismatch")
            mapping = dict(zip(fit.frame_id, pd.to_numeric(fit.chainage_m, errors="raise")))
            indices = [i for i, fid in enumerate(run_frames.frame_id) if fid in mapping]
            run_frames = run_frames.iloc[indices[0]:indices[1] + 1].reset_index(drop=True)
            if len(run_frames) < 2: raise DataIntegrityError("No interior frames between endpoint anchors")
            y0, y1 = mapping[run_frames.iloc[0].frame_id], mapping[run_frames.iloc[-1].frame_id]
            if not np.isfinite([y0, y1]).all() or y0 == y1: raise DataIntegrityError("Invalid metric endpoint positions")
            failures = []
            if method == "flow": progress, failures = flow_progress(base, run_frames, low_feature_policy=low_feature_policy)
            elif method == "uniform": progress = np.arange(len(run_frames), dtype=float)
            else:
                sign = 1 if y1 > y0 else -1
                progress = monotone_projection(sign * run_frames.nominal_chainage_m.to_numpy(float))
            progress = progress - progress[0]
            if progress[-1] <= 0: raise DataIntegrityError("Degenerate monotone progress")
            predicted = y0 + (y1 - y0) * progress / progress[-1]
            outputs.append(pd.DataFrame({"frame_id": run_frames.frame_id, "run_id": run, "position_m": predicted,
                                         "uncertainty_m": base["alignment"]["uncertainty_m"], "uncertainty_calibrated": False}))
            diagnostics.append({"run_id": run, "frames": len(run_frames), "fallback_frame_ids": failures,
                                "direction": "increasing" if y1 > y0 else "decreasing", "fit_frame_ids": list(mapping)})
        result = pd.concat(outputs, ignore_index=True)
        write_csv(destination / "frame_alignment.csv", result)
        write_json(destination / "FROZEN.json", {"created": utc_now(), "method": method, "data_fingerprint": meta["data_fingerprint"],
                    "estimator_fingerprint": fingerprint({"method": method, "low_feature_policy": low_feature_policy, "source": file_hash(__file__), "data": meta["data_fingerprint"]}),
                    "anchor_sha256": file_hash(anchors_path), "fit_frame_ids": anchors.frame_id.tolist(),
                    "predictions_sha256": file_hash(destination / "frame_alignment.csv"), "withheld_references_read": False,
                    "diagnostics": diagnostics, "physical_accuracy_verified": False,
                    "interpretation": "Endpoint-scaled flow is a progress proxy, not calibrated metric visual odometry. Inter-run consistency alone does not certify absolute accuracy."})
        return {"output": str(destination), "frames": len(result), "physical_accuracy_verified": False}
    except BaseException as e:
        write_json(destination / "FAILURE.json", {"type": type(e).__name__, "error": str(e)})
        raise


def read_alignment(path):
    path = Path(path)
    frozen = path.parent / "FROZEN.json"
    ready = path.parent / "READY.json"
    if frozen.exists():
        info = read_json(frozen)
        if info["predictions_sha256"] != file_hash(path): raise DataIntegrityError("Frozen alignment predictions changed")
    elif ready.exists():
        info = read_json(ready)
        if info.get("artifacts", {}).get(path.name) != file_hash(path): raise DataIntegrityError("Evaluation alignment predictions changed")
        provenance = read_json(path.parent / "provenance.json")
        info["estimator_fingerprint"] = provenance["experiment_fingerprint"]
    else: raise ArtifactError("Alignment predictions need a FROZEN.json or evaluation READY.json")
    frame = pd.read_csv(path)
    require_columns(frame, ["run_id", "frame_id", "position_m", "uncertainty_m"], "frozen alignment")
    if frame.frame_id.duplicated().any(): raise DataIntegrityError("Duplicate alignment frame IDs")
    if not np.isfinite(frame[["position_m", "uncertainty_m"]].to_numpy()).all() or (frame.uncertainty_m <= 0).any(): raise DataIntegrityError("Invalid alignment estimate/uncertainty")
    return frame, info


def marker_pairs(predictions, references, role):
    require_columns(references, ["marker_id", "run_id", "frame_id", "chainage_m", "annotation_uncertainty_m", "role", "source"], "marker annotations")
    if references.empty or not references.role.eq(role).all(): raise LeakageError(f"Only separate '{role}' references are accepted")
    if references.frame_id.duplicated().any() or references.source.eq("").any(): raise DataIntegrityError("Duplicate/unattributed marker annotations")
    if not np.isfinite(references[["chainage_m", "annotation_uncertainty_m"]].to_numpy(float)).all() or (references.annotation_uncertainty_m < 0).any():
        raise DataIntegrityError("Invalid marker target/annotation uncertainty")
    joined = references.merge(predictions, on=["frame_id", "run_id"], how="left", validate="one_to_one", indicator=True)
    return joined.loc[joined._merge.eq("both")].copy(), joined.loc[joined._merge.ne("both"), "frame_id"].tolist()


def calibrate_intervals(base, predictions_path, references_path, output, coverage=0.9):
    if not 0 < coverage < 1: raise ConfigurationError("Coverage must be between zero and one")
    predictions, prediction_info = read_alignment(predictions_path)
    refs = pd.read_csv(references_path, keep_default_na=False)
    manifest, info, _ = load_split(base)
    allowed = {fid for ids in manifest.loc[manifest.partition.eq("train"), "candidate_frame_ids"] for fid in ids}
    if not set(refs.frame_id) <= allowed: raise LeakageError("Interval calibration references must belong to permitted TRAIN candidate frames")
    pairs, missing = marker_pairs(predictions, refs, "calibration")
    if missing or pairs.empty: raise DataIntegrityError("Calibration marker predictions are missing; no reference was silently dropped")
    scores = (abs(pairs.position_m - pairs.chainage_m) + pairs.annotation_uncertainty_m) / pairs.uncertainty_m
    rank = int(np.ceil((len(scores) + 1) * coverage))
    if rank > len(scores): raise DataIntegrityError("Too few calibration encounters for the requested finite conformal quantile")
    scale = float(np.sort(scores.to_numpy())[rank - 1])
    destination = fresh_directory(output)
    result = {"scale": max(scale, 1e-6), "coverage_target": coverage, "references": len(pairs),
              "marker_groups": pairs.marker_id.nunique(), "frame_ids": refs.frame_id.tolist(),
              "predictions_sha256": file_hash(predictions_path), "references_sha256": file_hash(references_path),
              "estimator_fingerprint": prediction_info.get("estimator_fingerprint"),
              "split_fingerprint": info["split_fingerprint"], "source": "permitted training encounters only",
              "caveat": "Finite-sample quantile construction; repeated runs/sites are dependent, so exchangeable independent-sample coverage is not guaranteed."}
    write_json(destination / "calibration.json", result)
    return result


def evaluate_alignment(predictions_path, references_path, output, calibration_path=None):
    predictions, info = read_alignment(predictions_path)
    refs = pd.read_csv(references_path, keep_default_na=False)
    pairs, missing = marker_pairs(predictions, refs, "check")
    if set(refs.frame_id) & set(info.get("fit_frame_ids", [])): raise LeakageError("Fit endpoint cannot serve as an independent check marker")
    scale = 1.0
    if calibration_path:
        calibration = read_json(calibration_path)
        if set(refs.frame_id) & set(calibration["frame_ids"]): raise LeakageError("Calibration and check encounters overlap")
        if not info.get("estimator_fingerprint") or info["estimator_fingerprint"] != calibration.get("estimator_fingerprint"):
            raise ArtifactError("Alignment interval calibration belongs to another frozen estimator")
        scale = calibration["scale"]
    if pairs.empty: raise DataIntegrityError("No check marker has a prediction; report coverage before claiming accuracy")
    pairs["absolute_error_m"] = abs(pairs.position_m - pairs.chainage_m)
    pairs["excess_over_annotation_m"] = (pairs.absolute_error_m - pairs.annotation_uncertainty_m).clip(lower=0)
    pairs["interval_radius_m"] = scale * pairs.uncertainty_m
    pairs["covers_reference_centre"] = pairs.absolute_error_m <= pairs.interval_radius_m
    pairs["covers_entire_reference_interval"] = pairs.absolute_error_m + pairs.annotation_uncertainty_m <= pairs.interval_radius_m
    def summary(g):
        return {"encounters": len(g), "median_error_m": float(g.absolute_error_m.median()), "mae_m": float(g.absolute_error_m.mean()),
                "p95_error_m": float(g.absolute_error_m.quantile(0.95)), "worst_error_m": float(g.absolute_error_m.max()),
                "reference_centre_coverage": float(g.covers_reference_centre.mean()),
                "entire_reference_interval_coverage": float(g.covers_entire_reference_interval.mean()),
                "mean_interval_radius_m": float(g.interval_radius_m.mean())}
    report = {**summary(pairs), "annotated_encounters": len(refs), "prediction_coverage": len(pairs) / len(refs), "missing_frame_ids": missing,
              "independent_marker_ids": int(pairs.marker_id.nunique()), "intervals_empirically_calibrated": bool(calibration_path),
              "no_recalibration_on_check_references": True,
              "warning": "Reference geometry/coordinate precision and repeated-marker dependence limit physical accuracy claims."}
    destination = fresh_directory(output)
    write_csv(destination / "encounters.csv", pairs.drop(columns="_merge")); write_json(destination / "metrics.json", report)
    grouped = [{"group": key, "value": value, **summary(g)} for key in ["run_id", "marker_id"] for value, g in pairs.groupby(key)]
    write_csv(destination / "groups.csv", pd.DataFrame(grouped))
    write_json(destination / "provenance.json", {"predictions_sha256": file_hash(predictions_path), "references_sha256": file_hash(references_path),
                                               "calibration_sha256": file_hash(calibration_path) if calibration_path else None})
    return report


def contact_sheet(base, run_id, output, every=100, maximum=120):
    from PIL import Image, ImageDraw
    if every < 1 or maximum < 1: raise ConfigurationError("Positive contact-sheet sampling counts required")
    root, _ = load_prepared(base)
    frames = pd.read_csv(root / "frames.csv", keep_default_na=False)
    frames = frames.loc[frames.run_id.eq(run_id) & frames.rejection_reason.eq("")].sort_values("timestamp_ns")
    if frames.empty: raise DataIntegrityError("No accepted frames for this run")
    selected = frames.iloc[::every].head(maximum)
    destination = fresh_directory(output)
    thumbs = []; lines = ["<!doctype html><meta charset='utf-8'><title>Rail-VIVID annotation frames</title>", "<p>Identify encounters manually; no annotations or model results are inferred here.</p>"]
    for i, row in enumerate(selected.itertuples()):
        with Image.open(Path(base["data"]["root"]) / row.relative_path) as original:
            original.load(); thumb = original.convert("RGB"); thumb.thumbnail((360, 220))
        name = f"frame_{i:04d}.jpg"; thumb.save(destination / name, quality=85)
        lines.append(f"<figure><img src='{name}'><figcaption>{html.escape(row.frame_id)}</figcaption></figure>")
    atomic_write(destination / "index.html", lambda f: f.write("\n".join(lines)))
    return {"output": str(destination / "index.html"), "frames": len(selected), "annotations_created": False}
