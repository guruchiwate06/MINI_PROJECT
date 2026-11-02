from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
import sys

from railvivid.common import jsonable, read_json
from railvivid.errors import PipelineError, ConfigurationError
from .config import load_config


def parser():
    p = argparse.ArgumentParser(prog="track-ot", description="Experimental TRACK-OT++ on unchanged Rail-VIVID; no implicit fitting")
    p.add_argument("--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    default = str(Path(__file__).parents[1] / "configs/track_ot_plus_plus.json")
    for name in ["doctor", "smoke", "prepare", "split", "extract-vision", "train", "evaluate", "robustness", "plan", "annotation-template", "contact-sheet", "align", "align-calibrate", "alignment-predict", "repeated"]:
        q = sub.add_parser(name); q.add_argument("--config", default=default)
        if name in ["smoke", "evaluate", "robustness", "alignment-predict"]: q.add_argument("--device", default="cpu" if name == "smoke" else "cuda")
        if name == "smoke": q.add_argument("--synthetic", action="store_true")
        if name == "extract-vision": q.add_argument("--allow-download", action="store_true")
        if name == "train": q.add_argument("--on-training-device", action="store_true"); q.add_argument("--resume", action="store_true")
        if name == "evaluate":
            q.add_argument("--partition", choices=["validation", "test"], default="test"); q.add_argument("--tag", default="clean"); q.add_argument("--corruption", default="{}")
            from .runner import STRESSES
            q.add_argument("--stress", choices=sorted(STRESSES))
        if name == "robustness":
            q.add_argument("--partition", choices=["validation", "test"], default="test")
            q.add_argument("--cases", default="missing_vision,missing_vibration,missing_axles,noise_0p1,noise_0p5,shift_minus5,shift_plus5,shift_minus10,shift_plus10,shift_minus15,shift_plus15")
        if name in ["plan", "annotation-template", "contact-sheet", "align", "align-calibrate", "alignment-predict", "repeated"]: q.add_argument("--output", required=True)
        if name == "alignment-predict": q.add_argument("--partition", choices=["train", "validation", "test"], required=True)
        if name == "plan":
            q.add_argument("--variants", default="full,no_cycle,no_cross_run,no_drift,no_order,radius_labels,ot_control,generic_ot,attention,distance,vibration")
            q.add_argument("--folds", default="1,2,3"); q.add_argument("--seeds", default="42,43,44")
        if name == "contact-sheet": q.add_argument("--run", required=True); q.add_argument("--every", type=int, default=100)
        if name == "align":
            q.add_argument("--anchors", required=True); q.add_argument("--method", choices=["flow", "uniform", "nominal"], default="flow")
            q.add_argument("--low-feature-policy", choices=["abort", "timestamp"], default="abort")
        if name == "align-calibrate":
            q.add_argument("--predictions", required=True); q.add_argument("--references", required=True); q.add_argument("--coverage", type=float, default=0.9)
        if name == "repeated":
            q.add_argument("--predictions", required=True); q.add_argument("--validation-predictions", required=True)
    q = sub.add_parser("align-evaluate"); q.add_argument("--predictions", required=True); q.add_argument("--references", required=True)
    q.add_argument("--calibration"); q.add_argument("--output", required=True)
    for name in ["campaign-run", "campaign-evaluate", "summarize"]:
        q = sub.add_parser(name); q.add_argument("--plan", required=True)
        if name == "campaign-run": q.add_argument("--on-training-device", action="store_true"); q.add_argument("--resume-interrupted", action="store_true")
        if name == "summarize": q.add_argument("--output", required=True)
    return p


def synthetic_batch(c, device="cpu", batch_size=2):
    import torch
    n, m, p = c["sequence"]["windows"], c["sequence"]["max_frames"], c["encoder"]["patch_tokens"]
    d, t = c["_base"]["model"]["vision_dim"], c["_base"]["data"]["window_samples"]
    batch = {"waveform": torch.randn(batch_size, n, 6, t), "wave_mask": torch.ones(batch_size, n, dtype=torch.bool),
             "wave_positions": torch.linspace(-8, 8, n).expand(batch_size, -1), "vision": torch.randn(batch_size, m, d),
             "vision_patches": torch.randn(batch_size, m, p, d), "vision_mask": torch.ones(batch_size, m, dtype=torch.bool),
             "image_positions": torch.linspace(-20, 20, m).expand(batch_size, -1), "quality": torch.ones(batch_size, m, 2),
             "vibration_present": torch.ones(batch_size, dtype=torch.bool), "anchor_slot": torch.full((batch_size,), n // 2, dtype=torch.long),
             "instance_labels": torch.zeros(batch_size, n), "bag_label": torch.arange(batch_size).remainder(2).float(),
             "label": torch.arange(batch_size).remainder(2).float(), "index": torch.arange(batch_size), "anchor_chainage": torch.arange(batch_size).float() * 100}
    return {k: v.to(device) for k, v in batch.items()}


def dispatch(args):
    if args.command == "align-evaluate":
        from .alignment import evaluate_alignment
        return evaluate_alignment(args.predictions, args.references, args.output, args.calibration)
    if args.command in ["campaign-run", "campaign-evaluate", "summarize"]:
        from .campaign import execute, summarize
        if args.command == "summarize": return summarize(args.plan, args.output)
        return execute(args.plan, "train" if args.command == "campaign-run" else "evaluate",
                       getattr(args, "on_training_device", False), getattr(args, "resume_interrupted", False))
    c = load_config(args.config)
    if args.command == "doctor":
        from railvivid.cli import doctor
        from .config import source_fingerprint
        report = doctor(c["_base"])
        report["track_ot_source_hash"] = source_fingerprint(); report["dataset"] = "Rail-VIVID only"
        try:
            import numpy as np
            from railvivid.vision import vision_path
            from railvivid.common import read_json
            folder = vision_path(c["_base"]); meta = read_json(folder / "READY.json")
            for filename in ["embeddings.npy"] + (["patches.npy"] if c["encoder"]["patch_pooling"] else []):
                if filename not in meta["artifacts"]: raise ConfigurationError(f"Vision cache lacks {filename}")
                array = np.load(folder / filename, mmap_mode="r")
                report["checks"].append({"name": filename, "ok": True, "shape": list(array.shape), "payload_hash_checked": False})
        except (PipelineError, OSError, ValueError, KeyError) as e:
            report["checks"].append({"name": "track_ot_vision_cache", "ok": False, "error": str(e)})
        report["notes"] += ["TRACK-OT++ is untrained; no publication/physical-accuracy claim follows from this check.", "Patch mode requires verified patches.npy; annotation templates must be completed by an evaluator."]
        return report
    if args.command == "smoke":
        import torch
        from railvivid.common import select_device
        from .model import TrackOTPlus
        from .runner import resources, loader, to_device
        device = select_device(args.device, False)
        if args.synthetic: batch = synthetic_batch(c, device)
        else:
            r = resources(c); batch = to_device(next(iter(loader(r["datasets"]["validation"], c))), device)
        model = TrackOTPlus(c).to(device).eval()
        with torch.inference_mode(): output = model.details(batch)
        if not torch.isfinite(output["anchor_logit"]).all(): raise ConfigurationError("Nonfinite smoke logits")
        return {"trained": False, "optimizer_steps": 0, "synthetic": args.synthetic, "parameters": sum(p.numel() for p in model.parameters()),
                "output_shape": list(output["anchor_logit"].shape), "max_solver_residual": float(output["solver_residual"].max()), "device": str(device)}
    if args.command == "train":
        from .runner import train
        return train(c, args.on_training_device, args.resume)
    if args.command == "evaluate":
        from .runner import evaluate, STRESSES, validate_corruption
        try: corruption = json.loads(args.corruption)
        except ValueError as e: raise ConfigurationError("--corruption must be a JSON object") from e
        if args.stress:
            if corruption: raise ConfigurationError("Choose --stress or --corruption, not both")
            corruption = STRESSES[args.stress]
        validate_corruption(corruption)
        return evaluate(c, args.partition, args.device, corruption, args.stress or args.tag)
    if args.command == "robustness":
        from .runner import robustness
        return robustness(c, args.cases.split(","), args.partition, args.device)
    if args.command == "alignment-predict":
        from .runner import export_model_alignment
        return export_model_alignment(c, args.partition, args.output, args.device)
    if args.command == "plan":
        from .campaign import plan
        return plan(c, args.output, args.variants.split(","), tuple(map(int, args.folds.split(","))), tuple(map(int, args.seeds.split(","))))
    if args.command in ["prepare", "split", "extract-vision"]:
        if args.command == "prepare":
            from railvivid.prepare import prepare
            return prepare(c["_base"])
        if args.command == "split":
            from railvivid.splits import build_split
            return build_split(c["_base"])
        from railvivid.vision import extract_vision
        return extract_vision(c["_base"], device="cuda", allow_download=args.allow_download)
    from . import alignment
    if args.command == "annotation-template": return alignment.template(c["_base"], args.output)
    if args.command == "contact-sheet": return alignment.contact_sheet(c["_base"], args.run, args.output, args.every)
    if args.command == "align": return alignment.estimate(c["_base"], args.anchors, args.output, args.method, args.low_feature_policy)
    if args.command == "align-calibrate": return alignment.calibrate_intervals(c["_base"], args.predictions, args.references, args.output, args.coverage)
    if args.command == "repeated":
        import pandas as pd
        from railvivid.prepare import load_prepared
        from railvivid.repeated import repeated_pass_study
        root, _ = load_prepared(c["_base"])
        return repeated_pass_study(pd.read_csv(args.predictions), pd.read_csv(args.validation_predictions), pd.read_csv(root / "sites.csv"), args.output, c["_base"])
    raise ConfigurationError("Unknown command")


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        print(json.dumps(jsonable(dispatch(args)), indent=2, allow_nan=False)); return 0
    except (PipelineError, OSError, ValueError, KeyError) as e:
        logging.error("%s: %s", type(e).__name__, e, exc_info=args.verbose)
        return 2
