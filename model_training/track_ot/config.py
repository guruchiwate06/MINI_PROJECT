from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from railvivid.common import fingerprint, read_json
from railvivid.config import load_config as load_base, merge
from railvivid.errors import ConfigurationError


DEFAULTS = {
    "base_config": "full_scale.json", "base_overrides": [],
    "sequence": {"windows": 17, "max_frames": 48, "position_scale_m": 20.0},
    "encoder": {"dimension": 192, "heads": 6, "layers": 3, "sensor_width": 32,
                "sensor_depth": 3, "spectral_bins": 64, "spectral": True, "patch_pooling": True,
                "patch_tokens": 16, "position_features": True, "dropout": 0.15},
    "transport": {"matching": "transport", "solver": "unbalanced", "geometry": True, "physical_mask": True, "learn_drift": True,
                  "max_drift_m": 15.0, "max_log_stretch": 0.4, "position_sigma_m": 8.0,
                  "physical_band_m": 30.0, "epsilon": 0.25, "iterations": 160,
                  "tolerance": 0.005, "reject_cost": 1.5, "feature_weight": 1.0,
                  "position_weight": 1.0, "marginal_penalty": 2.0},
    "loss": {"supervision": "mil", "top_fraction": 0.2, "auxiliary_weight": 0.25,
             "crossing_weight": 0.1, "drift_regularization": 0.001,
             "equivariance_weight": 0.05, "synthetic_shift_m": 5.0,
             "recurrence_weight": 0.1, "cycle_weight": 0.1, "transport_weight": 0.01,
             "brier_weight": 0.1, "recurrence_temperature": 0.15,
             "partner_tolerance_m": 2.0, "negative_exclusion_m": 40.0},
    "training": {"device": "cuda", "seed": 42, "epochs": 100, "batch_size": 4,
                 "accumulation_steps": 8, "workers": 0, "learning_rate": 0.0002,
                 "weight_decay": 0.01, "gradient_clip": 1.0, "patience": 15,
                 "min_delta": 0.0001, "amp": True, "amp_dtype": "bfloat16", "initial_loss_scale": 512.0,
                 "modality_dropout": 0.15},
    "evaluation": {"abstention_target_error": 0.1, "abstention_min_samples": 20},
    "experiment": {"name": "track_ot_plus_fold1_s42", "output": "../artifacts/track_ot",
                   "notes": "Experimental Rail-VIVID only. Historical outer-test inspection disclosed. Learned drift is not physically validated."}
}


def validate(c):
    import math
    def walk(v, path=""):
        if isinstance(v, dict):
            for k, x in v.items(): walk(x, path + "." + k)
        elif isinstance(v, list):
            for x in v: walk(x, path)
        elif isinstance(v, float) and not math.isfinite(v):
            raise ConfigurationError(f"Nonfinite TRACK-OT setting: {path}")
    walk(c)
    for section, keys in {"encoder": ["spectral", "patch_pooling", "position_features"], "transport": ["geometry", "physical_mask", "learn_drift"], "training": ["amp"]}.items():
        for key in keys:
            if not isinstance(c[section][key], bool): raise ConfigurationError(f"{section}.{key} must be boolean")
    def positive(section, key, integer=False):
        v = c[section][key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0 or (integer and not isinstance(v, int)):
            raise ConfigurationError(f"{section}.{key} must be a positive {'integer' if integer else 'number'}")
    for section, keys in {"sequence": ["windows", "max_frames"],
                          "encoder": ["dimension", "heads", "layers", "sensor_width", "sensor_depth", "spectral_bins", "patch_tokens"],
                          "transport": ["iterations"],
                          "training": ["epochs", "batch_size", "accumulation_steps", "patience"]}.items():
        for k in keys: positive(section, k, True)
    if c["sequence"]["windows"] % 2 != 1: raise ConfigurationError("sequence.windows must be odd")
    if c["encoder"]["dimension"] % c["encoder"]["heads"]: raise ConfigurationError("dimension must divide into attention heads")
    if c["encoder"]["dimension"] % 3: raise ConfigurationError("dimension must divide into three sensor groups")
    for section, keys in {"sequence": ["position_scale_m"], "transport": ["epsilon", "tolerance", "position_sigma_m", "physical_band_m", "reject_cost"],
                          "loss": ["recurrence_temperature", "partner_tolerance_m", "negative_exclusion_m"],
                          "training": ["learning_rate", "gradient_clip"]}.items():
        for k in keys: positive(section, k)
    for k in ["max_drift_m", "max_log_stretch", "feature_weight", "position_weight"]:
        if c["transport"][k] < 0: raise ConfigurationError(f"transport.{k} cannot be negative")
    for k in ["auxiliary_weight", "crossing_weight", "drift_regularization", "equivariance_weight", "synthetic_shift_m", "recurrence_weight", "cycle_weight", "transport_weight", "brier_weight"]:
        if c["loss"][k] < 0: raise ConfigurationError(f"loss.{k} cannot be negative")
    if not 0 < c["loss"]["top_fraction"] <= 1: raise ConfigurationError("top_fraction must be in (0,1]")
    if c["loss"]["supervision"] not in ["mil", "radius"]: raise ConfigurationError("supervision must be mil or radius")
    if c["transport"]["matching"] not in ["transport", "attention", "distance", "vibration"]:
        raise ConfigurationError("Unknown matching ablation")
    if c["transport"]["solver"] not in ["unbalanced", "reject"]: raise ConfigurationError("Unknown OT solver")
    positive("transport", "marginal_penalty")
    for section, key in [("encoder", "dropout"), ("training", "modality_dropout")]:
        if not 0 <= c[section][key] < 1: raise ConfigurationError(f"{section}.{key} must be in [0,1)")
    if c["training"]["weight_decay"] < 0 or c["training"]["min_delta"] < 0: raise ConfigurationError("Negative training regularization/tolerance")
    if not isinstance(c["training"]["workers"], int) or isinstance(c["training"]["workers"], bool) or c["training"]["workers"] < 0:
        raise ConfigurationError("workers must be a nonnegative integer")
    if not isinstance(c["training"]["seed"], int) or c["training"]["seed"] < 0: raise ConfigurationError("seed must be nonnegative integer")
    if not 0 <= c["evaluation"]["abstention_target_error"] < 1: raise ConfigurationError("Invalid abstention error target")
    positive("evaluation", "abstention_min_samples", True)
    if not str(c["training"]["device"]).startswith("cuda"): raise ConfigurationError("TRACK-OT training requires CUDA; no CPU fitting mode")
    if c["training"]["amp_dtype"] not in ["bfloat16", "float16"]: raise ConfigurationError("Unknown AMP dtype")
    positive("training", "initial_loss_scale")
    name = c["experiment"]["name"]
    if not name or Path(name).name != name or name in [".", ".."]: raise ConfigurationError("Unsafe experiment name")


def load_config(path):
    path = Path(path).resolve()
    c = merge(copy.deepcopy(DEFAULTS), read_json(path))
    validate(c)
    base_path = (path.parent / c["base_config"]).resolve()
    base = load_base(base_path, c["base_overrides"])
    if base["split"]["protocol"] != "combined" or base["split"]["cohort"] != "paired":
        raise ConfigurationError("TRACK-OT requires the existing guarded combined/paired protocol")
    if base["model"]["channels"] != [1, 2, 3, 4, 5, 6]:
        raise ConfigurationError("Hierarchical sensor encoder requires all six channels in source order")
    if base["data"]["sampling_hz"] != 2000: raise ConfigurationError("Spectral encoder expects Rail-VIVID's 2kHz sampling")
    if base["split"]["label_sites"] is not None:
        raise ConfigurationError("Few-label budgets need separate bag eligibility design; not silently supported")
    base["model"]["kind"] = "fusion"
    output = Path(c["experiment"]["output"])
    c["experiment"]["output"] = str((path.parent / output).resolve())
    raw = Path(base["data"]["root"])
    if Path(c["experiment"]["output"]) == raw or raw in Path(c["experiment"]["output"]).parents:
        raise ConfigurationError("Experiment output must remain outside raw Rail-VIVID")
    if c["transport"]["max_drift_m"] > base["alignment"]["uncertainty_m"]:
        raise ConfigurationError("Learned drift exceeds the existing guarded uncertainty envelope")
    guarded_band = base["alignment"]["uncertainty_m"] + base["alignment"]["camera_footprint_m"]
    if c["transport"]["physical_band_m"] > guarded_band:
        raise ConfigurationError("Physical matching band exceeds the source-guarded envelope")
    c["_base"] = base
    c["_path"] = str(path)
    return c


def source_fingerprint():
    root = Path(__file__).parents[1]
    files = {}
    for folder in ["track_ot", "railvivid", "src"]:
        for p in sorted((root / folder).rglob("*.py")):
            files[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest()
    files["pyproject.toml"] = hashlib.sha256((root / "pyproject.toml").read_text(encoding="utf-8").encode()).hexdigest()
    return fingerprint(files)


def public_config(c):
    return {k: copy.deepcopy(v) for k, v in c.items() if not k.startswith("_")}


def identity(c, split_info, bag_fingerprint, vision_fingerprint):
    own = public_config(c)
    own.pop("base_config"); own.pop("base_overrides")
    own["experiment"].pop("output")
    base = copy.deepcopy(c["_base"])
    base.pop("_config_path", None)
    for section, keys in [("data", ["root", "artifacts"]), ("experiment", ["output"]), ("training", ["resume"])]:
        for k in keys: base[section].pop(k, None)
    return fingerprint({"track_ot": own, "base": base, "source": source_fingerprint(),
                        "split": split_info["split_fingerprint"], "bags": bag_fingerprint, "vision": vision_fingerprint})
