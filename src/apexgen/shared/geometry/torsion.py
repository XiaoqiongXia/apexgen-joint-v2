"""Backbone dihedral extraction and circular arithmetic."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def wrap_angle(angle: Tensor) -> Tensor:
    """Wrap radians to the canonical half-open interval ``[-π, π)``."""

    return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi


def circular_difference(target: Tensor, source: Tensor) -> Tensor:
    """Return the shortest signed displacement from source to target."""

    return wrap_angle(target - source)


def dihedral(p0: Tensor, p1: Tensor, p2: Tensor, p3: Tensor) -> Tensor:
    """Compute a signed dihedral angle for broadcast-compatible points."""

    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    eps = torch.finfo(b1.dtype).eps
    b1 = b1 / torch.linalg.vector_norm(b1, dim=-1, keepdim=True).clamp_min(eps)
    v = b0 - torch.sum(b0 * b1, dim=-1, keepdim=True) * b1
    w = b2 - torch.sum(b2 * b1, dim=-1, keepdim=True) * b1
    x = torch.sum(v * w, dim=-1)
    y = torch.sum(torch.linalg.cross(b1, v, dim=-1) * w, dim=-1)
    return wrap_angle(torch.atan2(y, x))


def extract_backbone_torsions(n: Tensor, ca: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
    """Extract per-residue ``(φ, ψ, ω)`` angles and endpoint-validity masks."""

    if n.shape != ca.shape or n.shape != c.shape or n.shape[-1] != 3:
        raise ValueError("N, CA, and C coordinates must share shape (..., L, 3)")
    length = n.shape[-2]
    if length < 1:
        raise ValueError("Peptide must contain at least one residue")
    angles = torch.zeros(*n.shape[:-1], 3, dtype=n.dtype, device=n.device)
    mask = torch.zeros(*n.shape[:-1], 3, dtype=torch.bool, device=n.device)
    if length == 1:
        return angles, mask

    angles[..., 1:, 0] = dihedral(c[..., :-1, :], n[..., 1:, :], ca[..., 1:, :], c[..., 1:, :])
    angles[..., :-1, 1] = dihedral(n[..., :-1, :], ca[..., :-1, :], c[..., :-1, :], n[..., 1:, :])
    angles[..., :-1, 2] = dihedral(ca[..., :-1, :], c[..., :-1, :], n[..., 1:, :], ca[..., 1:, :])
    mask[..., 1:, 0] = True
    mask[..., :-1, 1] = True
    mask[..., :-1, 2] = True
    return angles, mask
