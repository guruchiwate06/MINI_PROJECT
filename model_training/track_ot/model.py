from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from railvivid.errors import DataIntegrityError
from railvivid.models import TemporalEncoder
from .transport import crossing_penalty, reject_transport, transport_summary, unbalanced_transport_batch, unbalanced_objective


class SensorSpectrumEncoder(nn.Module):
    """Separate axle/bogie/body temporal branches plus a learnable spectral view."""
    def __init__(self, settings):
        super().__init__()
        d = settings["dimension"]
        self.spectral = settings["spectral"]
        self.bins = settings["spectral_bins"]
        self.branches = nn.ModuleList([TemporalEncoder(2, settings["sensor_width"], settings["sensor_depth"], 7, d // 3, settings["dropout"]) for _ in range(3)])
        self.sensor_fusion = nn.Sequential(nn.Linear((d // 3) * 3, d), nn.LayerNorm(d), nn.GELU())
        if self.spectral:
            self.frequency = nn.Sequential(nn.LayerNorm(6 * self.bins), nn.Linear(6 * self.bins, d), nn.GELU(), nn.Linear(d, d))
            self.view_gate = nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid())

    def forward(self, waveform):
        b, n, c, t = waveform.shape
        x = waveform.reshape(b * n, c, t)
        groups = [[0, 2], [1, 3], [4, 5]]  # Bogie, axle-box, car-body: source channel numbering.
        temporal = self.sensor_fusion(torch.cat([branch(x[:, groups[i]]) for i, branch in enumerate(self.branches)], -1))
        if self.spectral:
            with torch.autocast(device_type=x.device.type, enabled=False):
                spectrum = torch.log1p(torch.fft.rfft(x.float(), dim=-1).abs())
                # The source sampling rate is 2kHz. Retain frequencies <=500Hz.
                spectrum = spectrum[..., :t // 4 + 1]
                spectrum = F.adaptive_avg_pool1d(spectrum, self.bins).flatten(1)
            frequency = self.frequency(spectrum.to(temporal.dtype))
            gate = self.view_gate(torch.cat([temporal, frequency], -1))
            temporal = temporal + gate * frequency
        return temporal.reshape(b, n, -1)


class MonotonePositionWarp(nn.Module):
    """Bounded monotone image-position warp, not a monotone transport plan."""
    def __init__(self, dimension, settings):
        super().__init__()
        self.maximum = settings["max_drift_m"]
        self.stretch = settings["max_log_stretch"]
        self.shift = nn.Linear(2 * dimension, 1)
        self.rate = nn.Linear(dimension, 1)
        # Begin at the nominal mapping; no assumption it is physically correct.
        nn.init.zeros_(self.shift.weight); nn.init.zeros_(self.shift.bias)
        nn.init.zeros_(self.rate.weight); nn.init.zeros_(self.rate.bias)

    def forward(self, position, image, vibration_global):
        if (position[1:] < position[:-1]).any(): raise DataIntegrityError("Image positions must be canonicalized in increasing track order")
        shift = self.maximum * torch.tanh(self.shift(torch.cat([image.mean(0), vibration_global], -1))).squeeze()
        rate = torch.exp(self.stretch * torch.tanh(self.rate(image).squeeze(-1)))
        delta = position[1:] - position[:-1]
        increments = delta * (rate[1:] + rate[:-1]) / 2
        curve = torch.cat([position.new_zeros(1), increments.cumsum(0)])
        warp = curve - (curve[0] + curve[-1]) / 2 + (position[0] + position[-1]) / 2 + shift
        # Clipping between two increasing curves preserves nondecreasing order.
        return torch.minimum(torch.maximum(warp, position - self.maximum), position + self.maximum)


def mil_pool(logits, mask, fraction):
    results = []
    for z, valid in zip(logits, mask):
        count = int(valid.sum())
        if not count: raise DataIntegrityError("Empty MIL bag")
        k = max(1, math.ceil(count * fraction))
        results.append(z[valid].topk(k).values.mean())
    return torch.stack(results)


class TrackOTPlus(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        e = cfg["encoder"]; d = e["dimension"]
        self.wave_encoder = SensorSpectrumEncoder(e)
        self.image_projection = nn.Sequential(nn.LayerNorm(cfg["_base"]["model"]["vision_dim"]), nn.Linear(cfg["_base"]["model"]["vision_dim"], d), nn.GELU())
        self.patch_attention = nn.Linear(d, 1)
        self.position = nn.Sequential(nn.Linear(1, d), nn.Tanh())
        def context():
            layer = nn.TransformerEncoderLayer(d, e["heads"], 4 * d, e["dropout"], batch_first=True, norm_first=True, activation="gelu")
            return nn.TransformerEncoder(layer, e["layers"], norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.wave_context = context(); self.image_context = context()
        self.warp = MonotonePositionWarp(d, cfg["transport"])
        self.quality_gate = nn.Sequential(nn.Linear(2, d // 2), nn.GELU(), nn.Linear(d // 2, 1), nn.Sigmoid())
        self.fusion_gate = nn.Sequential(nn.Linear(2 * d + 2, d), nn.GELU(), nn.Linear(d, 1), nn.Sigmoid())
        self.instance_head = nn.Sequential(nn.LayerNorm(3 * d + 2), nn.Linear(3 * d + 2, d), nn.GELU(), nn.Dropout(e["dropout"]), nn.Linear(d, 1))
        self.vibration_head = nn.Linear(d, 1); self.vision_head = nn.Linear(d, 1)
        self.recurrence_projection = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, batch):
        return self.details(batch)["anchor_logit"]

    def contextual_images(self, base, positions, image_mask, quality):
        safe = image_mask.clone(); safe[~image_mask.any(1), 0] = True
        z = base
        if self.cfg["encoder"]["position_features"]:
            z = z + self.position(positions[:, :, None] / self.cfg["sequence"]["position_scale_m"])
        return self.image_context(z, src_key_padding_mask=~safe) * self.quality_gate(quality)

    def drift_from_cached(self, batch, output, positions):
        """Only image context + warp; never recompute waveform/FFT/OT/heads."""
        z = self.contextual_images(output["image_base"], positions, batch["vision_mask"], batch["quality"])
        curves = []
        with torch.autocast(device_type=z.device.type, enabled=False):
            for i in range(len(z)):
                mask = batch["vision_mask"][i]
                curve = positions[i].float() * 0
                if mask.any() and batch["vibration_present"][i]:
                    ip = positions[i, mask].float()
                    vi = output["wave_features"][i, batch["wave_mask"][i]].float().mean(0)
                    curve = curve.masked_scatter(mask, self.warp(ip, z[i, mask].float(), vi) - ip)
                curves.append(curve)
        return torch.stack(curves)

    def details(self, batch):
        waveform, mask = batch["waveform"], batch["wave_mask"]
        images, image_mask = batch["vision"], batch["vision_mask"]
        if not mask.any(1).all(): raise DataIntegrityError("A sequence bag contains no waveform support")
        if self.cfg["transport"]["matching"] == "vibration" and not batch["vibration_present"].all():
            raise DataIntegrityError("Vibration-only control cannot infer with missing vibration")
        if not ((batch["vibration_present"]) | image_mask.any(1)).all(): raise DataIntegrityError("No observed modality in a bag")
        for key in ["waveform", "vision", "vision_patches", "wave_positions", "image_positions", "quality"]:
            if not torch.isfinite(batch[key]).all(): raise DataIntegrityError(f"Nonfinite model input: {key}")
        if (batch["anchor_slot"] < 0).any() or (batch["anchor_slot"] >= mask.shape[1]).any(): raise DataIntegrityError("Invalid anchor slot")
        if not mask.gather(1, batch["anchor_slot"][:, None]).all(): raise DataIntegrityError("Anchor refers to padding")
        scale = self.cfg["sequence"]["position_scale_m"]
        v = self.wave_encoder(waveform)
        if self.cfg["encoder"]["position_features"]: v = v + self.position(batch["wave_positions"][:, :, None] / scale)
        v = self.wave_context(v, src_key_padding_mask=~mask)
        z = self.image_projection(images)
        if self.cfg["encoder"]["patch_pooling"]:
            patch = self.image_projection(batch["vision_patches"])
            weights = self.patch_attention(patch).float().softmax(-2).to(patch.dtype)
            z = z + (weights * patch).sum(-2)
        image_base = z
        z = self.contextual_images(image_base, batch["image_positions"], image_mask, batch["quality"])
        b, n, d = v.shape
        aligned, masses, entropies, drift_curves, penalties, residuals, locations, variances, plans, objectives = [], [], [], [], [], [], [], [], [], []
        settings = self.cfg["transport"]
        specs, cost_batch, feasible_batch = [], [], []
        with torch.autocast(device_type=v.device.type, enabled=False):
            for i in range(b):
                vi, zi = v[i, mask[i]].float(), z[i, image_mask[i]].float()
                wp, ip = batch["wave_positions"][i, mask[i]].float(), batch["image_positions"][i, image_mask[i]].float()
                if (wp[1:] < wp[:-1]).any(): raise DataIntegrityError("Waveform positions are not in canonical track order")
                available = bool(batch["vibration_present"][i]) and len(zi) > 0
                corrected = self.warp(ip, zi, vi.mean(0)) if available and settings["learn_drift"] else ip
                position_cost = 0.5 * ((wp[:, None] - corrected[None]) / settings["position_sigma_m"]).square()
                cost = settings["feature_weight"] * (1 - F.normalize(vi, dim=-1, eps=1e-6) @ F.normalize(zi, dim=-1, eps=1e-6).T)
                if settings["geometry"]: cost = cost + settings["position_weight"] * position_cost
                allowed = (wp[:, None] - ip[None]).abs() <= settings["physical_band_m"]
                if not settings["physical_mask"]: allowed = torch.ones_like(allowed)
                if not available: allowed = torch.zeros_like(allowed)
                specs.append((corrected, cost, position_cost, allowed, available))
                wi, ii = torch.where(mask[i])[0], torch.where(image_mask[i])[0]
                indices = (wi[:, None].expand(-1, len(ii)), ii[None].expand(len(wi), -1))
                cost_batch.append(cost.new_zeros((n, images.shape[1])).index_put(indices, cost))
                feasible_batch.append(torch.zeros((n, images.shape[1]), dtype=torch.bool, device=cost.device).index_put(indices, allowed))
        batched_plan = batched_residual = None; solver_steps = 0
        if settings["matching"] == "transport" and settings["solver"] == "unbalanced":
            with torch.profiler.record_function("track_ot/batched_transport"):
                batched_plan, batched_residual, solver_steps = unbalanced_transport_batch(torch.stack(cost_batch), torch.stack(feasible_batch), mask, image_mask,
                    settings["epsilon"], settings["iterations"], settings["tolerance"], settings["marginal_penalty"])
        for i in range(b):
            wvalid, ivalid = mask[i], image_mask[i]
            vi, zi = v[i, wvalid], z[i, ivalid]
            wp, ip = batch["wave_positions"][i, wvalid].float(), batch["image_positions"][i, ivalid].float()
            corrected, cost, position_cost, allowed, available = specs[i]
            drift = corrected - ip
            aligned_i = v.new_zeros((n, d)); mass_i = v.new_zeros(n, dtype=torch.float32); entropy_i = mass_i.clone()
            loc_i, var_i = mass_i.clone(), mass_i.clone()
            zero = vi.float().sum() * 0
            penalty, residual = zero, zero
            padded_plan = mass_i.new_zeros((n, images.shape[1])); ot_objective = zero
            if len(zi) and settings["matching"] != "vibration":
                with torch.autocast(device_type=v.device.type, enabled=False):
                    if settings["matching"] == "transport" and available:
                        # The nominal band preserves the original source envelope even as drift is learned.
                        if settings["solver"] == "unbalanced":
                            plan = batched_plan[i, wvalid][:, ivalid]; residual = batched_residual[i]
                            ot_objective = unbalanced_objective(plan, cost, settings["epsilon"], settings["marginal_penalty"])
                        else:
                            plan, residual = reject_transport(cost, allowed, settings["epsilon"], settings["iterations"], settings["tolerance"], settings["reject_cost"])
                            ot_objective = (plan * cost).sum()
                        row_mass, weights, entropy = transport_summary(plan)
                        confidence = (row_mass * len(vi)).clamp(0, 1)
                        penalty = crossing_penalty(plan)
                    else:
                        # A missing-vibration fallback uses position only; no bogus vibration similarity.
                        score = -position_cost if settings["matching"] == "distance" or not available else -cost
                        weights = score.softmax(-1)
                        confidence = torch.ones(len(vi), device=v.device)
                        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(-1)
                        plan = weights / len(vi)
                    wi, ii = torch.where(wvalid)[0], torch.where(ivalid)[0]
                    padded_plan = padded_plan.index_put((wi[:, None].expand(-1, len(ii)), ii[None].expand(len(wi), -1)), plan)
                    aligned_i[wvalid] = (weights @ zi.float()).to(v.dtype)
                    mass_i[wvalid], entropy_i[wvalid] = confidence, entropy / max(math.log(max(len(zi), 2)), 1)
                    loc = weights @ corrected
                    loc_i[wvalid] = loc
                    var_i[wvalid] = (weights * (corrected[None] - loc[:, None]).square()).sum(-1)
            padded_drift = mass_i.new_zeros(images.shape[1]); padded_drift[ivalid] = drift
            aligned.append(aligned_i); masses.append(mass_i); entropies.append(entropy_i)
            drift_curves.append(padded_drift); penalties.append(penalty); residuals.append(residual)
            locations.append(loc_i); variances.append(var_i)
            plans.append(padded_plan); objectives.append(ot_objective)
        u, mass, entropy = torch.stack(aligned), torch.stack(masses), torch.stack(entropies)
        gate = self.fusion_gate(torch.cat([v, u, mass[:, :, None].to(v.dtype), entropy[:, :, None].to(v.dtype)], -1))
        gated = u * gate * mass[:, :, None]
        instance = self.instance_head(torch.cat([v, gated, v * gated, mass[:, :, None].to(v.dtype), entropy[:, :, None].to(v.dtype)], -1)).squeeze(-1)
        vib_logits, vis_logits = self.vibration_head(v).squeeze(-1), self.vision_head(u).squeeze(-1)
        # Low-mass/absent vision has a separately supervised vibration fallback.
        instance = torch.where(mass > 1e-5, instance, vib_logits)
        instance = torch.where(batch["vibration_present"][:, None], instance, vis_logits)
        if settings["matching"] == "vibration": instance = vib_logits
        idx = batch["anchor_slot"][:, None]
        pooled = (v * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True)
        return {"anchor_logit": instance.gather(1, idx).squeeze(1), "instance_logits": instance,
                "bag_logit": mil_pool(instance, mask, self.cfg["loss"]["top_fraction"]),
                "vibration_logits": vib_logits, "vision_logits": vis_logits,
                "embedding": F.normalize(self.recurrence_projection(pooled).float(), dim=-1, eps=1e-6),
                "drift_m": torch.stack(drift_curves), "crossing": torch.stack(penalties),
                "solver_residual": torch.stack(residuals), "match_mass": mass, "plans": torch.stack(plans),
                "transport_objective": torch.stack(objectives), "image_features": z,
                "image_base": image_base, "wave_features": v, "solver_iterations": solver_steps,
                "correspondence_position_m": torch.stack(locations), "correspondence_variance_m2": torch.stack(variances)}
