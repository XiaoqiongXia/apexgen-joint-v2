"""Pocket-relative origins and proper-rotation Kabsch frame labels."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from apexgen.shared.geometry.rigid import Rigid


@dataclass(frozen=True)
class PeptideFrameLabel:
    """Rigid endpoint label and fixed-geometry idealization error."""

    translation: Tensor
    rotation: Tensor
    idealization_rmsd: Tensor


def masked_centroid(points: Tensor, mask: Tensor | None = None) -> Tensor:
    """Return the centroid over the point axis, rejecting empty selections."""

    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError("points must have shape (..., N, 3)")
    if not torch.is_floating_point(points) or not torch.isfinite(points).all():
        raise ValueError("points must contain finite floating-point coordinates")
    if mask is None:
        return points.mean(dim=-2)
    if mask.shape != points.shape[:-1]:
        raise ValueError("mask must have shape (..., N)")
    weights = mask.to(dtype=points.dtype)
    count = weights.sum(dim=-1, keepdim=True)
    if torch.any(count == 0):
        raise ValueError("centroid selection must contain at least one point")
    return (points * weights[..., None]).sum(dim=-2) / count


def site_origin(core_ca: Tensor, core_mask: Tensor | None = None) -> Tensor:
    """Compute the complete core C-alpha centroid without defining PCA axes."""

    return masked_centroid(core_ca, core_mask)


def _proper_rotation(covariance: Tensor) -> Tensor:
    u, singular_values, vh = torch.linalg.svd(covariance)
    scale = singular_values[..., :1].clamp_min(torch.finfo(covariance.dtype).eps)
    if torch.any(singular_values[..., 1] <= scale[..., 0] * 1e-10):
        raise ValueError("Kabsch alignment is degenerate for collinear points")
    v = vh.transpose(-1, -2)
    determinant = torch.linalg.det(v @ u.transpose(-1, -2))
    correction = torch.ones_like(singular_values)
    correction[..., -1] = torch.where(determinant < 0, -1.0, 1.0)
    return (v * correction[..., None, :]) @ u.transpose(-1, -2)


def proper_kabsch(moving: Tensor, target: Tensor, mask: Tensor | None = None) -> Rigid:
    """Fit ``target = R @ moving + t`` with a proper Kabsch rotation."""

    if moving.shape != target.shape or moving.ndim < 2 or moving.shape[-1] != 3:
        raise ValueError("moving and target must share shape (..., N, 3)")
    if not torch.is_floating_point(moving) or not torch.is_floating_point(target):
        raise TypeError("Kabsch coordinates must be floating point")
    if not torch.isfinite(moving).all() or not torch.isfinite(target).all():
        raise ValueError("Kabsch coordinates must be finite")
    if mask is None:
        mask = torch.ones(moving.shape[:-1], dtype=torch.bool, device=moving.device)
    if mask.shape != moving.shape[:-1]:
        raise ValueError("mask must have shape (..., N)")
    if torch.any(mask.sum(dim=-1) < 3):
        raise ValueError("Kabsch alignment requires at least three points")

    moving_center = masked_centroid(moving, mask)
    target_center = masked_centroid(target, mask)
    weights = mask.to(dtype=moving.dtype)[..., None]
    moving_zero = (moving - moving_center[..., None, :]) * weights
    target_zero = (target - target_center[..., None, :]) * weights
    covariance = moving_zero.transpose(-1, -2) @ target_zero
    rotation = _proper_rotation(covariance)
    translation = target_center - (rotation @ moving_center.unsqueeze(-1)).squeeze(-1)
    return Rigid(rotation, translation)


def extract_peptide_frame_label(
    ideal_backbone: Tensor,
    experimental_backbone: Tensor,
    origin: Tensor,
    atom_mask: Tensor | None = None,
) -> PeptideFrameLabel:
    """Extract pocket-relative ``p``, proper ``R``, and idealization RMSD."""

    if ideal_backbone.shape != experimental_backbone.shape:
        raise ValueError("ideal and experimental backbones must share shape (..., L, 4, 3)")
    if ideal_backbone.ndim < 3 or ideal_backbone.shape[-2:] != (4, 3):
        raise ValueError("backbones must have shape (..., L, 4, 3)")
    if origin.shape != ideal_backbone.shape[:-3] + (3,):
        raise ValueError("origin batch shape must match the backbones")
    if atom_mask is None:
        atom_mask = torch.ones(ideal_backbone.shape[:-1], dtype=torch.bool, device=ideal_backbone.device)
    if atom_mask.shape != ideal_backbone.shape[:-1]:
        raise ValueError("atom_mask must have shape (..., L, 4)")

    ideal_ca_center = masked_centroid(ideal_backbone[..., :, 1, :], atom_mask[..., :, 1])
    experimental_ca_center = masked_centroid(
        experimental_backbone[..., :, 1, :], atom_mask[..., :, 1]
    )
    ideal_zero = ideal_backbone - ideal_ca_center[..., None, None, :]
    experimental_zero = experimental_backbone - experimental_ca_center[..., None, None, :]
    flat_shape = ideal_backbone.shape[:-3] + (-1, 3)
    flat_ideal = ideal_zero.reshape(flat_shape)
    flat_experimental = experimental_zero.reshape(flat_shape)
    flat_mask = atom_mask.reshape(atom_mask.shape[:-2] + (-1,))
    masked_ideal = torch.where(flat_mask[..., None], flat_ideal, torch.zeros_like(flat_ideal))
    masked_experimental = torch.where(
        flat_mask[..., None], flat_experimental, torch.zeros_like(flat_experimental)
    )
    covariance = masked_ideal.transpose(-1, -2) @ masked_experimental
    rotation = _proper_rotation(covariance)
    fitted = (rotation[..., None, :, :] @ flat_ideal.unsqueeze(-1)).squeeze(-1)
    residual = torch.where(
        flat_mask[..., None], fitted - flat_experimental, torch.zeros_like(fitted)
    )
    squared_error = residual.square().sum(dim=-1)
    scalar_weights = flat_mask.to(squared_error.dtype)
    rmsd = torch.sqrt(
        (squared_error * scalar_weights).sum(dim=-1) / scalar_weights.sum(dim=-1)
    )
    return PeptideFrameLabel(
        translation=experimental_ca_center - origin,
        rotation=rotation,
        idealization_rmsd=rmsd,
    )


def reconstruct_from_frame(
    ideal_backbone: Tensor,
    label: PeptideFrameLabel,
    origin: Tensor,
) -> Tensor:
    """Apply a pocket-relative frame label to a CA-centered ideal backbone."""

    translation = label.translation + origin
    return (label.rotation[..., None, None, :, :] @ ideal_backbone.unsqueeze(-1)).squeeze(-1) + translation[
        ..., None, None, :
    ]
