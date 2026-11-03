"""Precision policy and strict, named gradient diagnostics. No optimizer logic."""
from __future__ import annotations

import torch
from railvivid.errors import ConfigurationError, TrainingError


def precision(c, device):
    enabled = bool(c["training"]["amp"])
    name = c["training"].get("amp_dtype", "bfloat16")
    if name not in ["float16", "bfloat16"]: raise ConfigurationError("amp_dtype must be float16 or bfloat16")
    if enabled and device.type == "cuda" and name == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ConfigurationError("BF16 is unavailable; use an explicit FP32 recovery config rather than silently changing precision")
    return enabled, torch.bfloat16 if name == "bfloat16" else torch.float16


def gradient_report(model, gradients=None):
    pairs = [(name, p.grad) for name, p in model.named_parameters()] if gradients is None else list(zip([name for name, p in model.named_parameters() if p.requires_grad], gradients))
    invalid, largest = [], []
    total_square = 0.0
    for name, g in pairs:
        if g is None: continue
        value = g.detach().float()
        finite = torch.isfinite(value)
        if not finite.all(): invalid.append({"parameter": name, "nonfinite_elements": int((~finite).sum())})
        if finite.any():
            values = value[finite]
            largest.append({"parameter": name, "max_abs": float(values.abs().max())})
            total_square += float(values.double().square().sum())
    return {"finite": not invalid, "nonfinite_parameters": invalid,
            "finite_elements_l2_norm": total_square ** 0.5, "largest": sorted(largest, key=lambda x: -x["max_abs"])[:10]}


def strict_clip_gradients(model, maximum):
    """FP64 norm avoids overflow of an FP32 sum of squares; NaN/Inf still abort."""
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients: raise TrainingError("No parameter gradients were produced")
    norm = torch.stack([g.detach().double().square().sum() for g in gradients]).sum().sqrt()
    if not torch.isfinite(norm):
        report = gradient_report(model)
        names = [r["parameter"] for r in report["nonfinite_parameters"][:8]]
        raise TrainingError(f"Nonfinite parameter gradients: {names}. Guard retained; no update performed.")
    coefficient = (maximum / (norm + 1e-12)).clamp(max=1.0)
    for g in gradients: g.mul_(coefficient.to(dtype=g.dtype))
    return norm
