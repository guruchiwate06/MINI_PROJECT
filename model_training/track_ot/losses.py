from __future__ import annotations

import torch
from torch.nn import functional as F
from railvivid.errors import DataIntegrityError
from .model import mil_pool
from .transport import unbalanced_transport, transport_summary


def supervised(logits, batch, cfg, pos_weight):
    if cfg["loss"]["supervision"] == "mil":
        mask = batch["bag_label"] >= 0
        pooled = mil_pool(logits, batch["wave_mask"], cfg["loss"]["top_fraction"])
        return F.binary_cross_entropy_with_logits(pooled[mask].float(), batch["bag_label"][mask].float(), pos_weight=pos_weight) if mask.any() else logits.sum() * 0
    mask = batch["wave_mask"] & (batch["instance_labels"] >= 0)
    return F.binary_cross_entropy_with_logits(logits[mask].float(), batch["instance_labels"][mask].float(), pos_weight=pos_weight) if mask.any() else logits.sum() * 0


def recurrence_loss(output, partner, batch, cfg):
    valid = batch["partner_available"] & batch["vibration_present"] & batch["partner_vibration_present"]
    if valid.sum() < 2: return output["embedding"].sum() * 0
    a, b = output["embedding"][valid], partner["embedding"][valid]
    positions = batch["anchor_chainage"][valid]
    partner_positions = batch["partner_anchor_chainage"][valid]
    distance = (positions[:, None] - partner_positions[None]).abs()
    negatives = distance > cfg["loss"]["negative_exclusion_m"]
    if "bag_support_low" in batch:
        a_low, a_high = batch["bag_support_low"][valid], batch["bag_support_high"][valid]
        b_low, b_high = batch["partner_bag_support_low"][valid], batch["partner_bag_support_high"][valid]
        negatives &= (a_high[:, None] < b_low[None]) | (b_high[None] < a_low[:, None])
    allowed = negatives | torch.eye(len(a), dtype=torch.bool, device=a.device)
    usable = negatives.any(1)
    if not usable.any(): return a.sum() * 0
    with torch.autocast(device_type=a.device.type, enabled=False):
        similarity = (a.float() @ b.float().T) / cfg["loss"]["recurrence_temperature"]
        return F.cross_entropy(similarity.masked_fill(~allowed, float("-inf"))[usable], torch.arange(len(a), device=a.device)[usable])


def cross_run_cycle(output, partner, batch, cfg):
    with torch.autocast(device_type=output["embedding"].device.type, enabled=False):
        return _cross_run_cycle(output, partner, batch, cfg)


def _cross_run_cycle(output, partner, batch, cfg):
    """A-wave -> A-image -> B-image -> B-wave, compared with spatial support.

    Geometry here uses only training-run metadata. No held-out targets or IDs
    enter model features. Repeated locations are not treated as new defects.
    """
    terms = []
    for i in range(len(batch["waveform"])):
        if not batch["partner_available"][i]: continue
        if not (batch["vibration_present"][i] and batch["partner_vibration_present"][i]): continue
        va, vb = batch["vision_mask"][i], batch["partner_vision_mask"][i]
        wa, wb = batch["wave_mask"][i], batch["partner_wave_mask"][i]
        if not va.any() or not vb.any(): continue
        pa = output["plans"][i][wa][:, va].float()
        pb = partner["plans"][i][wb][:, vb].float()
        if pa.sum() < 1e-6 or pb.sum() < 1e-6: continue
        za = F.normalize(output["image_features"][i, va].float(), dim=-1, eps=1e-6)
        zb = F.normalize(partner["image_features"][i, vb].float(), dim=-1, eps=1e-6)
        offset = (batch["partner_anchor_chainage"][i] - batch["anchor_chainage"][i]).float()
        xa = batch["image_positions"][i, va].float() + output["drift_m"][i, va]
        xb = batch["partner_image_positions"][i, vb].float() + partner["drift_m"][i, vb] + offset
        cost = 1 - za @ zb.T + 0.5 * ((xa[:, None] - xb[None]) / cfg["transport"]["position_sigma_m"]).square()
        link, _ = unbalanced_transport(cost, torch.ones_like(cost, dtype=torch.bool), cfg["transport"]["epsilon"],
                                       cfg["transport"]["iterations"], cfg["transport"]["tolerance"], cfg["transport"]["marginal_penalty"])
        a = transport_summary(pa)[1]
        bridge = transport_summary(link)[1]
        reverse_b = transport_summary(pb.T)[1]
        cycle = a @ bridge @ reverse_b
        positions_a = batch["wave_positions"][i, wa].float()
        positions_b = batch["partner_wave_positions"][i, wb].float() + offset
        supported = (positions_a >= positions_b.min()) & (positions_a <= positions_b.max()) & (pa.sum(1) > 1e-6)
        supported &= cycle.sum(1) > 1e-6
        if supported.any():
            reconstructed = (cycle @ positions_b) / cycle.sum(1).clamp_min(1e-8)
            terms.append(F.smooth_l1_loss(reconstructed[supported] / cfg["sequence"]["position_scale_m"],
                                          positions_a[supported] / cfg["sequence"]["position_scale_m"]))
    return torch.stack(terms).mean() if terms else output["embedding"].sum() * 0


def objective(model, output, batch, cfg, pos_weight, tensor_terms=False):
    settings = cfg["loss"]
    terms = {"supervised": supervised(output["instance_logits"], batch, cfg, pos_weight)}
    vmask = batch["vibration_present"]
    imask = batch["vision_mask"].any(1)
    zero = output["anchor_logit"].sum() * 0
    def subset(mask): return {k: v[mask] for k, v in batch.items()}
    terms["auxiliary"] = (supervised(output["vibration_logits"][vmask], subset(vmask), cfg, pos_weight) if vmask.any() else zero)
    if imask.any(): terms["auxiliary"] = terms["auxiliary"] + supervised(output["vision_logits"][imask], subset(imask), cfg, pos_weight)
    terms["crossing"] = output["crossing"].mean()
    terms["transport"] = output["transport_objective"].mean()
    image_mask = batch["vision_mask"]
    terms["drift"] = output["drift_m"][image_mask].square().mean() / max(cfg["transport"]["max_drift_m"] ** 2, 1) if image_mask.any() else zero
    terms["equivariance"] = zero
    if settings["equivariance_weight"] and cfg["transport"]["learn_drift"] and image_mask.any():
        shifts = torch.empty(len(image_mask), 1, device=image_mask.device).uniform_(-settings["synthetic_shift_m"], settings["synthetic_shift_m"])
        altered_positions = batch["image_positions"] + shifts * image_mask
        # Same dropout masks isolate the effect of the synthetic metadata shift.
        rng = torch.get_rng_state(); cuda_rng = torch.cuda.get_rng_state_all() if image_mask.is_cuda else None
        with torch.no_grad(), torch.profiler.record_function("track_ot/equivariance_reference"):
            reference = model.drift_from_cached(batch, output, batch["image_positions"])
        torch.set_rng_state(rng)
        if cuda_rng is not None: torch.cuda.set_rng_state_all(cuda_rng)
        with torch.profiler.record_function("track_ot/equivariance_shifted"):
            changed = model.drift_from_cached(batch, output, altered_positions)
        valid = image_mask & batch["vibration_present"][:, None]
        if valid.any():
            residual = changed - reference + shifts
            terms["equivariance"] = F.smooth_l1_loss(residual[valid] / cfg["sequence"]["position_scale_m"], torch.zeros_like(residual[valid]))
    terms["recurrence"] = zero; terms["cycle"] = zero
    if (settings["recurrence_weight"] or settings["cycle_weight"]) and batch.get("partner_available", torch.tensor(False)).any():
        partner_batch = {k[len("partner_"):]: v for k, v in batch.items() if k.startswith("partner_") and k != "partner_available"}
        with torch.profiler.record_function("track_ot/partner_forward"):
            partner = model.details(partner_batch)
        terms["recurrence"] = recurrence_loss(output, partner, batch, cfg)
        with torch.profiler.record_function("track_ot/cross_run_cycle"):
            terms["cycle"] = cross_run_cycle(output, partner, batch, cfg) if settings["cycle_weight"] else zero
    terms["brier"] = zero
    known = batch["bag_label"] >= 0
    if known.any(): terms["brier"] = (output["bag_logit"][known].float().sigmoid() - batch["bag_label"][known]).square().mean()
    weights = {"supervised": 1, "auxiliary": settings["auxiliary_weight"], "crossing": settings["crossing_weight"],
               "transport": settings["transport_weight"], "drift": settings["drift_regularization"],
               "equivariance": settings["equivariance_weight"], "recurrence": settings["recurrence_weight"],
               "cycle": settings["cycle_weight"], "brier": settings["brier_weight"]}
    total = sum(weights[k] * value for k, value in terms.items())
    if not torch.isfinite(total): raise DataIntegrityError("Nonfinite TRACK-OT objective")
    return total, ({k: weights[k] * v for k, v in terms.items()} if tensor_terms else {k: float(v.detach()) for k, v in terms.items()})
