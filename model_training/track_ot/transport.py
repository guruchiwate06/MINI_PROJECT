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
    normalized = torch.where(row_mass[:, None] > 1e-6, plan / row_mass[:, None].clamp_min(1e-6), torch.zeros_like(plan))
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
    plan, residual, _ = unbalanced_transport_batch(cost[None], allowed[None], epsilon=epsilon, iterations=iterations,
                                                  tolerance=tolerance, marginal_penalty=marginal_penalty)
    return plan[0], residual[0]


def unbalanced_transport_batch(cost, allowed, row_mask=None, col_mask=None, epsilon=0.25,
                               iterations=160, tolerance=0.005, marginal_penalty=2.0):
    """Batched log-domain UOT with early stopping and safe empty support.

    No all-negative-infinity logsumexp is differentiated: empty rows/columns
    use finite dummy reductions and remain unmatched in the returned plan.
    """
    if cost.ndim != 3 or cost.shape != allowed.shape or allowed.dtype != torch.bool or not cost.numel():
        raise DataIntegrityError("Batched UOT needs B x N x M costs and boolean mask")
    if not torch.isfinite(cost).all(): raise DataIntegrityError("Nonfinite batched UOT cost")
    if min(epsilon, tolerance, marginal_penalty) <= 0 or iterations < 1: raise DataIntegrityError("Invalid UOT settings")
    with torch.autocast(device_type=cost.device.type, enabled=False):
        cost = cost.float()
        rows = torch.ones(cost.shape[:2], dtype=torch.bool, device=cost.device) if row_mask is None else row_mask
        cols = torch.ones((cost.shape[0], cost.shape[2]), dtype=torch.bool, device=cost.device) if col_mask is None else col_mask
        feasible = allowed & rows[:, :, None] & cols[:, None, :]
        active_r, active_c = feasible.any(-1), feasible.any(-2)
        kernel = (-cost / epsilon).masked_fill(~feasible, float("-inf"))
        loga = -rows.sum(-1).clamp_min(1).float().log()[:, None]
        logb = -cols.sum(-1).clamp_min(1).float().log()[:, None]
        u = torch.zeros_like(cost[:, :, 0]); v = torch.zeros_like(cost[:, 0, :])
        exponent = marginal_penalty / (marginal_penalty + epsilon)
        for step in range(iterations):
            row_inputs = torch.where(active_r[:, :, None], kernel + v[:, None, :], torch.zeros_like(kernel))
            next_u = torch.where(active_r, exponent * (loga - torch.logsumexp(row_inputs, -1)), torch.zeros_like(u))
            col_inputs = torch.where(active_c[:, None, :], kernel + next_u[:, :, None], torch.zeros_like(kernel))
            next_v = torch.where(active_c, exponent * (logb - torch.logsumexp(col_inputs, -2)), torch.zeros_like(v))
            if (step + 1) % 8 == 0 or step + 1 == iterations:
                residual = torch.maximum((next_u - u).abs().amax(-1), (next_v - v).abs().amax(-1))
                converged = step + 1 >= min(32, iterations) and bool((residual.detach() <= tolerance * 0.1).all())
            else: converged = False
            u, v = next_u, next_v
            if converged: break
        plan = (kernel + u[:, :, None] + v[:, None, :]).exp()
        if not torch.isfinite(plan).all() or (residual.detach() > tolerance).any():
            raise TrainingError(f"Batched UOT did not converge (max residual {float(residual.detach().max()):.6g})")
        return plan, residual, step + 1


def unbalanced_objective(plan, cost, epsilon, marginal_penalty):
    def kl(mass, reference):
        return (mass * (mass.clamp_min(1e-12).log() - math.log(reference)) - mass + reference).sum()
    entropy = (plan * (plan.clamp_min(1e-12).log() - 1)).sum()
    return (plan * cost).sum() + epsilon * entropy + marginal_penalty * (kl(plan.sum(1), 1 / plan.shape[0]) + kl(plan.sum(0), 1 / plan.shape[1]))
