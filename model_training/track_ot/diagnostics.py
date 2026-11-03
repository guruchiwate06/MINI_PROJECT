"""Bounded forward/backward diagnostics. Never constructs an optimizer.

Real data comes exclusively from permitted training bags. No validation/test
scores or calibration are produced while debugging numerical behavior.
"""
from __future__ import annotations

import contextlib
import hashlib
import time
from pathlib import Path
import torch

from railvivid.common import seed_everything, select_device, utc_now, write_json
from railvivid.errors import ArtifactError, ConfigurationError
from .config import source_fingerprint
from .losses import objective
from .model import TrackOTPlus
from .numerics import gradient_report


def weights_hash(model):
    result = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        result.update(key.encode()); result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def diagnose(c, output, device="cpu", batches=2, precisions=("float32",), synthetic=False,
             from_run=None, components=False, anomaly=False, profile=False):
    from .cli import synthetic_batch
    from .runner import resources, loader, to_device, warm_start_weights, training_view
    if batches < 1 or not precisions or set(precisions) - {"float32", "bfloat16", "float16"}:
        raise ConfigurationError("Positive batches and explicit float32/bfloat16/float16 diagnostic modes required")
    destination = Path(output)
    if destination.exists(): raise ArtifactError("Diagnostic output exists; use a new directory")
    dev = select_device(device, False)
    if from_run and synthetic: raise ConfigurationError("A real checkpoint must be diagnosed against its verified real training source")
    r = None if synthetic else resources(c, verify=True, augment=True)
    dataset, ratio = (None, 1.0) if synthetic else training_view(r, c)
    destination.mkdir(parents=True)
    report = {"created": utc_now(), "source_hash": source_fingerprint(), "synthetic": synthetic, "partition": "synthetic" if synthetic else "train",
              "optimizer_steps": 0, "test_accessed": False, "profiles": [], "precision_runs": []}
    def synchronize():
        if dev.type == "cuda": torch.cuda.synchronize(dev)
    for name in precisions:
        if name == "bfloat16" and dev.type == "cuda" and not torch.cuda.is_bf16_supported():
            report["precision_runs"].append({"precision": name, "unsupported": True}); continue
        seed_everything(c["training"]["seed"])
        model = TrackOTPlus(c).to(dev).train()
        if from_run: report["parent_checkpoint"] = warm_start_weights(model, from_run, r)
        initial = weights_hash(model); rows = []
        source = None if synthetic else iter(loader(dataset, c))
        activities = [torch.profiler.ProfilerActivity.CPU]
        if dev.type == "cuda": activities.append(torch.profiler.ProfilerActivity.CUDA)
        manager = torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) if profile else contextlib.nullcontext()
        with manager as trace:
            for step in range(batches):
                row = {"batch": step + 1, "precision": name}
                try:
                    start = time.perf_counter()
                    if synthetic:
                        batch = synthetic_batch(c, "cpu")
                        partner = synthetic_batch(c, "cpu")
                        for k, value in partner.items(): batch["partner_" + k] = value
                        batch["partner_available"] = torch.ones(len(batch["index"]), dtype=torch.bool)
                    else:
                        try: batch = next(source)
                        except StopIteration: break
                    row["loader_seconds"] = time.perf_counter() - start
                    row["source_window_ids"] = [] if synthetic else [dataset.records[i]["window_id"] for i in batch["index"].tolist()]
                    synchronize(); start = time.perf_counter(); batch = to_device(batch, dev); synchronize()
                    row["transfer_seconds"] = time.perf_counter() - start
                    model.zero_grad(set_to_none=True)
                    if dev.type == "cuda": torch.cuda.reset_peak_memory_stats(dev)
                    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]
                    scale = c["training"]["initial_loss_scale"] if name == "float16" else 1.0
                    row["loss_scale"] = scale
                    with torch.autograd.detect_anomaly(check_nan=True) if anomaly else contextlib.nullcontext():
                        with torch.autocast(device_type=dev.type, dtype=dtype, enabled=name != "float32"):
                            synchronize(); start = time.perf_counter()
                            with torch.profiler.record_function("diagnostic/main_forward"): out = model.details(batch)
                            synchronize(); row["forward_seconds"] = time.perf_counter() - start
                            start = time.perf_counter()
                            with torch.profiler.record_function("diagnostic/objective"):
                                loss, terms = objective(model, out, batch, c, torch.tensor(ratio, device=dev), tensor_terms=True)
                            synchronize(); row["auxiliary_loss_seconds"] = time.perf_counter() - start
                        row["weighted_loss_components"] = {k: float(v.detach()) for k, v in terms.items()}
                        row["loss"] = float(loss.detach()); row["solver_iterations"] = out["solver_iterations"]
                        parameters = [p for p in model.parameters() if p.requires_grad]
                        if components:
                            row["component_gradients"] = {}
                            for key, value in terms.items():
                                grads = torch.autograd.grad(value * scale, parameters, retain_graph=True, allow_unused=True)
                                grads = [None if g is None else g / scale for g in grads]
                                row["component_gradients"][key] = gradient_report(model, grads)
                        synchronize(); start = time.perf_counter()
                        with torch.profiler.record_function("diagnostic/backward"): (loss * scale).backward()
                        synchronize(); row["backward_seconds"] = time.perf_counter() - start
                    for parameter in model.parameters():
                        if parameter.grad is not None: parameter.grad.div_(scale)
                    row["gradients"] = gradient_report(model)
                    row["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None
                    row["passed"] = row["gradients"]["finite"]
                except (RuntimeError, ValueError, KeyError) as e:
                    row["passed"] = False; row["error"] = {"type": type(e).__name__, "message": str(e)}
                    row["gradients"] = gradient_report(model)
                rows.append(row)
                write_json(destination / f"{name}_batch{step + 1:04d}.json", row)
                if not row["passed"]: break
        if profile:
            trace.export_chrome_trace(str(destination / f"{name}_trace.json"))
            (destination / f"{name}_operators.txt").write_text(trace.key_averages().table(sort_by="self_cpu_time_total", row_limit=40), encoding="utf-8")
            report["profiles"].append(f"{name}_trace.json")
        unchanged = initial == weights_hash(model)
        report["precision_runs"].append({"precision": name, "batches_checked": len(rows), "all_finite": bool(rows) and all(r["passed"] for r in rows),
                                         "weights_unchanged": unchanged, "results": rows})
        if not unchanged: raise RuntimeError("A diagnostic changed model weights")
        del model
    report["caveat"] = "A bounded diagnostic cannot establish full-training convergence, GPU utilization over an epoch, or generalization. No optimizer exists in this command."
    write_json(destination / "diagnostics.json", report)
    return {"output": str(destination), "optimizer_steps": 0, "precision_summary": [{k: v for k, v in r.items() if k != "results"} for r in report["precision_runs"]]}
