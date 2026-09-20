"""Differentiable SO(3) and circular helpers used by Joint-v2."""

from __future__ import annotations

import torch
from torch import Tensor

from apexgen.shared.geometry.joint_residue_constants import CHI_PI_PERIODIC, as_torch


def wrap_angle(angle: Tensor) -> Tensor:
    """Map circular angles to the idempotent half-open interval ``[-pi, pi)``."""

    return torch.remainder(angle + torch.pi, 2 * torch.pi) - torch.pi


def canonicalize_masked_angles(angle: Tensor, mask: Tensor) -> Tensor:
    """Wrap valid circular values and force every invalid slot to exact zero."""

    if mask.shape != angle.shape or mask.dtype != torch.bool:
        raise TypeError("circular angle mask must be bool and match the angle layout")
    if not torch.is_floating_point(angle):
        raise TypeError("circular angles must be floating point")
    return torch.where(mask, wrap_angle(angle), torch.zeros_like(angle))


def sidechain_pi_periodic_mask(aatype: Tensor, angle_mask: Tensor) -> Tensor:
    """Return observed chi slots whose atom naming is equivalent after a pi shift."""

    if aatype.shape != angle_mask.shape[:-1] or aatype.dtype != torch.long:
        raise TypeError("aatype must be int64 and match the sidechain angle layout")
    if angle_mask.shape[-1] != 4 or angle_mask.dtype != torch.bool:
        raise TypeError("sidechain angle mask must be bool [..., 4]")
    residue_mask = angle_mask.any(-1)
    selected = aatype[residue_mask]
    if selected.numel() and bool(((selected < 0) | (selected >= 20)).any()):
        raise ValueError("observed sidechain angles require aatype values in [0, 20)")
    safe_aatype = torch.where(residue_mask, aatype, 0)
    periodic = as_torch(
        CHI_PI_PERIODIC,
        device=aatype.device,
        dtype=torch.bool,
    )[safe_aatype]
    return periodic & angle_mask


def symmetry_aware_chi_squared_distance(
    predicted: Tensor,
    target: Tensor,
    pi_periodic: Tensor,
) -> Tensor:
    """Return squared sin/cos distance, minimizing over pi-equivalent chi targets."""

    if predicted.shape != target.shape or predicted.shape[-2:] != (4, 2):
        raise ValueError("predicted and target chi tensors must match and end in [4, 2]")
    if pi_periodic.shape != predicted.shape[:-1] or pi_periodic.dtype != torch.bool:
        raise TypeError("pi-periodic chi mask must be bool and match the chi slots")
    direct = (predicted - target).square().sum(-1)
    shifted = (predicted + target).square().sum(-1)
    return torch.where(pi_periodic, torch.minimum(direct, shifted), direct)


def symmetry_aware_chi_distance(
    predicted: Tensor,
    target: Tensor,
    pi_periodic: Tensor,
) -> Tensor:
    """Return circular chi distance in radians with pi-periodic naming symmetry."""

    if predicted.shape != target.shape or predicted.shape[-2:] != (4, 2):
        raise ValueError("predicted and target chi tensors must match and end in [4, 2]")
    if pi_periodic.shape != predicted.shape[:-1] or pi_periodic.dtype != torch.bool:
        raise TypeError("pi-periodic chi mask must be bool and match the chi slots")
    cosine = (predicted * target).sum(-1).clamp(-1.0, 1.0)
    cosine = torch.where(pi_periodic, cosine.abs(), cosine)
    return torch.acos(cosine)


def _hat(vector: Tensor) -> Tensor:
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
        vector.shape[:-1] + (3, 3)
    )


def so3_exp(vector: Tensor) -> Tensor:
    """Rodrigues exponential with finite derivatives at the zero vector."""

    if vector.shape[-1:] != (3,):
        raise ValueError("SO(3) tangent vectors must end in dimension 3")
    theta2 = vector.square().sum(-1, keepdim=True)
    # Clamp before sqrt so the inactive trigonometric branch also has a finite
    # derivative at theta=0; torch.where alone does not prevent 0*NaN backward.
    safe_theta2 = theta2.clamp_min(1e-8)
    safe_theta = safe_theta2.sqrt()
    small = theta2 <= 1e-8
    a = torch.where(
        small,
        1 - theta2 / 6 + theta2.square() / 120,
        torch.sin(safe_theta) / safe_theta,
    )
    b = torch.where(
        small,
        0.5 - theta2 / 24 + theta2.square() / 720,
        (1 - torch.cos(safe_theta)) / safe_theta2,
    )
    skew = _hat(vector)
    identity = torch.eye(3, dtype=vector.dtype, device=vector.device).expand(skew.shape)
    return identity + a[..., None] * skew + b[..., None] * (skew @ skew)


def _near_pi_axis(rotation: Tensor, vee: Tensor, cosine: Tensor) -> Tensor:
    """Recover the axis using its largest component, then sign it with the skew part.

    At an exactly symmetric pi rotation the skew part is zero: the selected
    largest axis component stays positive (argmax breaks ties deterministically).
    On either side of pi, the skew part selects the principal-log branch.
    """

    identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    symmetric = 0.5 * (rotation + rotation.transpose(-1, -2))
    axis_outer = (symmetric - cosine[..., None, None] * identity) / (1 - cosine).clamp_min(1e-8)[
        ..., None, None
    ]
    diagonal = axis_outer.diagonal(dim1=-2, dim2=-1).clamp_min(0)
    safe_root = diagonal.clamp_min(1e-12).sqrt()
    x, y, z = safe_root.unbind(-1)
    candidate_x = torch.stack((x, axis_outer[..., 0, 1] / x, axis_outer[..., 0, 2] / x), -1)
    candidate_y = torch.stack((axis_outer[..., 0, 1] / y, y, axis_outer[..., 1, 2] / y), -1)
    candidate_z = torch.stack((axis_outer[..., 0, 2] / z, axis_outer[..., 1, 2] / z, z), -1)
    candidates = torch.stack((candidate_x, candidate_y, candidate_z), dim=-2)
    choice = torch.nn.functional.one_hot(diagonal.argmax(-1), 3).to(rotation.dtype)
    axis = (candidates * choice[..., None]).sum(-2)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    alignment = (axis * vee).sum(-1, keepdim=True)
    return torch.where(alignment < 0, -axis, axis)


def so3_log(rotation: Tensor) -> Tensor:
    """Principal SO(3) logarithm, numerically stable at identity and near pi.

    The principal logarithm is discontinuous at pi; finite autograd values at
    that cut are branch derivatives, not a globally defined derivative. Path,
    loss, and sampling must use this same branch convention. Geometry callers
    use FP32 (outside autocast), or FP64 for numerical checks.
    """

    if rotation.shape[-2:] != (3, 3):
        raise ValueError("SO(3) rotations must end in shape [3, 3]")
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(-1)
    cosine = ((trace - 1) / 2).clamp(-1.0, 1.0)
    vee = torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    sine_magnitude = (0.25 * vee.square().sum(-1)).clamp_min(1e-16).sqrt()
    theta = torch.atan2(sine_magnitude, cosine)
    generic = theta[..., None] * vee / (2 * sine_magnitude[..., None])
    # log(R) = 1/2 vee + O(theta^2) around identity. This branch avoids
    # differentiating a vector norm at zero.
    small = cosine > 0.9999
    small_result = (0.5 + vee.square().sum(-1, keepdim=True) / 48) * vee
    near_pi = cosine < -0.9999
    pi_result = theta[..., None] * _near_pi_axis(rotation, vee, cosine)
    return torch.where(
        small[..., None], small_result, torch.where(near_pi[..., None], pi_result, generic)
    )
