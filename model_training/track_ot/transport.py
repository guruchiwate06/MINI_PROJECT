"""Differentiable reject-augmented entropic transport, with checked marginals.

Dustbin augmentation permits variable partial matching. This is not fixed-mass
partial OT, KL-unbalanced OT, or a hard monotone transport solver.
"""
from __future__ import annotations

import math
import torch
from railvivid.errors import DataIntegrityError, TrainingError


def reject_transport(cost, allowed, epsilon=0.25, iterations=160, tolerance=0.005, reject_cost=1.5):
    if cost.ndim != 2 or not cost.numel() or allowed.shape != cost.shape or allowed.dtype != torch.bool:
        raise DataIntegrityError("Transport needs a nonempty N x M cost and boolean feasibility mask")
    if not torch.isfinite(cost).all(): raise DataIntegrityError("Transport cost contains NaN/Inf")
    if min(epsilon, reject_cost, tolerance) <= 0 or iterations < 1:
        raise DataIntegrityError("Invalid transport numerical settings")
    with torch.autocast(device_type=cost.device.type, enabled=False):
        cost = cost.float()
        n, m = cost.shape
        # Real marginals sum to one; each dustbin has capacity one. Total mass 2.
        top = torch.cat([cost.masked_fill(~allowed, float("inf")), cost.new_full((n, 1), reject_cost)], 1)
        bottom = torch.cat([cost.new_full((1, m), reject_cost), cost.new_zeros((1, 1))], 1)
        kernel = -torch.cat([top, bottom], 0) / epsilon
        a = torch.cat([cost.new_full((n,), 1.0 / n), cost.new_ones(1)])
        b = torch.cat([cost.new_full((m,), 1.0 / m), cost.new_ones(1)])
        loga, logb = a.log(), b.log()
        u, v = torch.zeros_like(a), torch.zeros_like(b)
        for _ in range(iterations):
            u = loga - torch.logsumexp(kernel + v[None], dim=1)
            v = logb - torch.logsumexp(kernel + u[:, None], dim=0)
        augmented = (kernel + u[:, None] + v[None]).exp()
        residual = torch.maximum((augmented.sum(1) - a).abs().max(), (augmented.sum(0) - b).abs().max())
        if not torch.isfinite(augmented).all() or float(residual.detach()) > tolerance:
            raise TrainingError(f"Transport failed marginal check ({float(residual.detach()):.6g} > {tolerance}); increase iterations/epsilon explicitly")
        return augmented[:-1, :-1], residual


def crossing_penalty(plan):
    """Soft inversion probability; no hard ordering guarantee. O(N*M) memory."""
    above = plan.cumsum(0) - plan
    above_right = above.flip(1).cumsum(1).flip(1) - above
    return (plan * above_right).sum() / plan.sum().square().clamp_min(1e-8)


def transport_summary(plan):
    row_mass = plan.sum(-1)
    normalized = plan / row_mass[:, None].clamp_min(1e-8)
    entropy = -(normalized * normalized.clamp_min(1e-8).log()).sum(-1)
    return row_mass, normalized, entropy


def unbalanced_transport(cost, allowed, epsilon=0.25, iterations=160, tolerance=0.005, marginal_penalty=2.0):
    """KL-unbalanced entropic OT. Returns plan and fixed-point residual.

    Zero-support rows/columns remain unmatched. Marginal errors are intentional,
    so convergence is tested on dual updates rather than exact marginal equality.
    """
    if cost.ndim != 2 or not cost.numel() or allowed.shape != cost.shape or allowed.dtype != torch.bool:
        raise DataIntegrityError("UOT needs nonempty matrix and boolean mask")
    if not torch.isfinite(cost).all(): raise DataIntegrityError("Nonfinite UOT cost")
    if min(epsilon, tolerance, marginal_penalty) <= 0 or iterations < 1: raise DataIntegrityError("Invalid UOT settings")
    with torch.autocast(device_type=cost.device.type, enabled=False):
        cost = cost.float()
        keep_rows, keep_cols = allowed.any(1), allowed.any(0)
        plan = cost * 0
        if not keep_rows.any(): return plan, cost.sum() * 0
        reduced = cost[keep_rows][:, keep_cols]
        feasible = allowed[keep_rows][:, keep_cols]
        kernel = (-reduced / epsilon).masked_fill(~feasible, float("-inf"))
        loga = reduced.new_full((len(reduced),), -math.log(cost.shape[0]))
        logb = reduced.new_full((reduced.shape[1],), -math.log(cost.shape[1]))
        u, v = torch.zeros_like(loga), torch.zeros_like(logb)
        exponent = marginal_penalty / (marginal_penalty + epsilon)
        residual = reduced.new_tensor(float("inf"))
        for _ in range(iterations):
            next_u = exponent * (loga - torch.logsumexp(kernel + v[None], 1))
            next_v = exponent * (logb - torch.logsumexp(kernel + next_u[:, None], 0))
            residual = torch.maximum((next_u - u).abs().max(), (next_v - v).abs().max())
            u, v = next_u, next_v
        matched = (kernel + u[:, None] + v[None]).exp()
        if not torch.isfinite(matched).all() or float(residual.detach()) > tolerance:
            raise TrainingError(f"UOT did not converge ({float(residual.detach()):.6g} > {tolerance})")
        rows = torch.where(keep_rows)[0]; cols = torch.where(keep_cols)[0]
        plan = plan.index_put((rows[:, None].expand(-1, len(cols)), cols[None].expand(len(rows), -1)), matched)
        return plan, residual


def unbalanced_objective(plan, cost, epsilon, marginal_penalty):
    def kl(mass, reference):
        return (mass * (mass.clamp_min(1e-12).log() - math.log(reference)) - mass + reference).sum()
    entropy = (plan * (plan.clamp_min(1e-12).log() - 1)).sum()
    return (plan * cost).sum() + epsilon * entropy + marginal_penalty * (kl(plan.sum(1), 1 / plan.shape[0]) + kl(plan.sum(0), 1 / plan.shape[1]))
