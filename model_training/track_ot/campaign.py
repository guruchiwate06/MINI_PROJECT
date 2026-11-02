"""Matched ablation plans; planning never dispatches training."""
from __future__ import annotations

import copy
from pathlib import Path
import pandas as pd

from railvivid.benchmark import FOLDS
from railvivid.common import file_hash, fingerprint, read_json, utc_now, write_csv, write_json
from railvivid.config import merge
from railvivid.errors import ArtifactError, ConfigurationError, DataIntegrityError
from .config import load_config, public_config, source_fingerprint

VARIANTS = {
    "full": {},
    "no_cycle": {"loss": {"cycle_weight": 0.0}},
    "no_recurrence": {"loss": {"recurrence_weight": 0.0}},
    "no_cross_run": {"loss": {"recurrence_weight": 0.0, "cycle_weight": 0.0}},
    "no_drift": {"transport": {"learn_drift": False}, "loss": {"equivariance_weight": 0.0, "drift_regularization": 0.0}},
    "no_position_cost": {"transport": {"geometry": False}},
    "no_order": {"loss": {"crossing_weight": 0.0}},
    "radius_labels": {"loss": {"supervision": "radius"}},
    "no_spectral": {"encoder": {"spectral": False}},
    "global_images": {"encoder": {"patch_pooling": False}},
    "reject_ot": {"transport": {"solver": "reject"}},
    "ot_control": {"transport": {"learn_drift": False}, "loss": {"cycle_weight": 0.0, "transport_weight": 0.0, "crossing_weight": 0.0, "equivariance_weight": 0.0}},
    "generic_ot": {"transport": {"geometry": False, "physical_mask": False, "learn_drift": False}, "encoder": {"position_features": False},
                   "loss": {"cycle_weight": 0.0, "transport_weight": 0.0, "crossing_weight": 0.0, "equivariance_weight": 0.0}},
    "attention": {"transport": {"matching": "attention", "learn_drift": False}, "loss": {"cycle_weight": 0.0, "transport_weight": 0.0, "crossing_weight": 0.0, "equivariance_weight": 0.0}},
    "distance": {"transport": {"matching": "distance", "learn_drift": False}, "loss": {"cycle_weight": 0.0, "transport_weight": 0.0, "crossing_weight": 0.0, "equivariance_weight": 0.0}},
    "vibration": {"transport": {"matching": "vibration", "learn_drift": False}, "encoder": {"patch_pooling": False},
                  "training": {"modality_dropout": 0.0}, "loss": {"cycle_weight": 0.0, "transport_weight": 0.0, "crossing_weight": 0.0, "equivariance_weight": 0.0}},
}


def plan(c, output, variants=("full", "no_cycle", "no_cross_run", "no_drift", "no_order", "radius_labels", "ot_control", "generic_ot", "attention", "distance", "vibration"), folds=(1, 2, 3), seeds=(42, 43, 44)):
    if not variants or set(variants) - set(VARIANTS): raise ConfigurationError("Unknown/empty TRACK-OT variants")
    if not folds or set(folds) - {1, 2, 3} or not seeds or any(s < 0 for s in seeds): raise ConfigurationError("Invalid folds/seeds")
    if len(set(variants)) != len(variants) or len(set(folds)) != len(folds) or len(set(seeds)) != len(seeds): raise ConfigurationError("Duplicate campaign dimensions")
    destination = Path(output).resolve()
    if destination.exists(): raise ArtifactError("Plan directory already exists")
    destination.mkdir(parents=True); configs = destination / "configs"; configs.mkdir()
    jobs = []
    for fold in folds:
        for seed in seeds:
            base = copy.deepcopy(c["_base"]); base.pop("_config_path", None)
            base["split"].update({"test_sites": FOLDS[fold - 1][0], "validation_sites": FOLDS[fold - 1][1]})
            base["training"]["seed"] = seed
            base_name = f"base_fold{fold}_s{seed}.json"; write_json(configs / base_name, base)
            for variant in variants:
                own = merge(public_config(c), copy.deepcopy(VARIANTS[variant]))
                name = f"track_ot_pp_fold{fold}_{variant}_s{seed}"
                own["experiment"]["name"] = name; own["training"]["seed"] = seed
                own["base_config"] = base_name; own["base_overrides"] = []
                file = configs / f"{name}.json"; write_json(file, own); load_config(file)
                jobs.append({"id": name, "fold": fold, "seed": seed, "variant": variant, "config": f"configs/{file.name}", "config_sha256": file_hash(file),
                             "base_config": f"configs/{base_name}", "base_sha256": file_hash(configs / base_name)})
    result = {"created": utc_now(), "source_hash": source_fingerprint(), "jobs": jobs, "training_started": False,
              "dataset": "Rail-VIVID only", "test_execution_separate": True, "protocol": "same guarded anchors and evaluation labels; sequence context must be matched across controls"}
    write_json(destination / "plan.json", result); write_csv(destination / "jobs.csv", pd.DataFrame(jobs))
    return {"plan": str(destination / "plan.json"), "jobs": len(jobs), "training_started": False}


def load_jobs(path):
    path = Path(path).resolve(); plan = read_json(path)
    if plan["source_hash"] != source_fingerprint(): raise ArtifactError("Code changed since planning; generate a new plan")
    jobs = []
    seen = set()
    for job in plan["jobs"]:
        if job["id"] in seen: raise DataIntegrityError("Duplicate job ID")
        seen.add(job["id"])
        file = (path.parent / job["config"]).resolve()
        base = (path.parent / job["base_config"]).resolve()
        if path.parent not in file.parents or path.parent not in base.parents: raise DataIntegrityError("Planned config escapes plan directory")
        if file_hash(file) != job["config_sha256"] or file_hash(base) != job["base_sha256"]: raise DataIntegrityError("Planned config changed")
        jobs.append((job, load_config(file)))
    return jobs


def execute(path, stage, on_training_device=False, resume_interrupted=False):
    from .runner import train, evaluate, run_path
    if stage not in ["train", "evaluate"]: raise ConfigurationError("Unknown campaign stage")
    if stage == "train" and not on_training_device: raise ConfigurationError("Campaign fitting requires --on-training-device")
    rows = []
    for job, c in load_jobs(path):
        folder = run_path(c)
        if stage == "train":
            if (folder / "DONE.json").exists():
                # Revalidate the source/config identity before declaring an existing job reusable.
                from .runner import resources, verify_run
                verify_run(folder)
                r = resources(c, normalizer=read_json(folder / "normalizer.json"))
                if read_json(folder / "DONE.json")["experiment_fingerprint"] != r["identity"]: raise ArtifactError("Completed job identity changed")
                result = {"already_complete": True}
            else: result = train(c, on_training_device=True, resume=resume_interrupted and folder.exists())
        else:
            existing = folder / "evaluations/test_clean"
            if (existing / "READY.json").exists():
                from .runner import verify_run
                done = verify_run(folder)
                if read_json(existing / "provenance.json")["experiment_fingerprint"] != done["experiment_fingerprint"]:
                    raise ArtifactError("Completed evaluation belongs to another experiment")
                for name, digest in read_json(existing / "READY.json")["artifacts"].items():
                    if Path(name).name != name or file_hash(existing / name) != digest: raise DataIntegrityError("Completed evaluation was changed")
                result = {"already_evaluated": True, "evaluation": str(existing)}
            else: result = evaluate(c, "test")
        rows.append({"id": job["id"], "result": result})
        write_json(Path(path).parent / f"{stage}_status.json", {"completed": rows, "updated": utc_now()})
    return {"jobs_completed": len(rows), "stage": stage}


def summarize(path, output):
    from .runner import run_path
    rows = []; cohorts = {}
    for job, c in load_jobs(path):
        directory = run_path(c) / "evaluations/test_clean"
        metrics = read_json(directory / "metrics.json"); provenance = read_json(directory / "provenance.json")
        ready = read_json(directory / "READY.json")
        for name, digest in ready["artifacts"].items():
            if file_hash(directory / name) != digest: raise DataIntegrityError("Published evaluation artifact changed")
        cohorts.setdefault((job["fold"], job["seed"]), set()).add(provenance["cohort_fingerprint"])
        rows.append({"method": job["variant"], "fold": job["fold"], "seed": job["seed"], "average_precision": metrics["average_precision"],
                     "roc_auc": metrics["roc_auc"], "f1": metrics["f1"], "brier": metrics["brier"],
                     "event_recall": metrics["events"]["event_recall"], "false_alarms_per_km": metrics["events"]["false_alarms_per_km"]})
    if any(len(values) > 1 for values in cohorts.values()): raise DataIntegrityError("Ablations have different evaluation anchors")
    destination = Path(output)
    if destination.exists(): raise ArtifactError("Summary exists")
    destination.mkdir(parents=True)
    frame = pd.DataFrame(rows); write_csv(destination / "all_results.csv", frame)
    summary = frame.groupby("method")[["average_precision", "roc_auc", "f1", "brier", "event_recall", "false_alarms_per_km"]].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(x) for x in summary.columns]; write_csv(destination / "summary.csv", summary.reset_index())
    write_json(destination / "protocol.json", {"all_planned_results_present": True, "cohorts_match": True,
                                              "caveat": "Folds/seeds share nine physical sites and one corridor; descriptive variation is not independent-dataset uncertainty."})
    return {"output": str(destination), "evaluations": len(rows)}
