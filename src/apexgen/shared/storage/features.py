"""SE(3)-consistent residue and residue-pair geometry features."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from apexgen.shared.geometry.rigid import Rigid
from apexgen.shared.geometry.torsion import dihedral


@dataclass(frozen=True)
class RosettaOrientations:
    omega_sin_cos: Tensor
    theta_sin_cos: Tensor
    phi_sin_cos: Tensor
    mask: Tensor


def _normalize(vector: Tensor) -> Tensor:
    epsilon = torch.finfo(vector.dtype).eps
    return vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(epsilon)


def residue_frames(n: Tensor, ca: Tensor, c: Tensor, mask: Tensor | None = None) -> tuple[Rigid, Tensor]:
    """Construct proper N-CA-C frames with origin at CA and columns x/y/z."""

    if n.shape != ca.shape or n.shape != c.shape or n.shape[-1] != 3:
        raise ValueError("N, CA, and C must share shape (..., Nres, 3)")
    x_axis = _normalize(c - ca)
    z_axis = _normalize(torch.linalg.cross(x_axis, n - ca, dim=-1))
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    rotation = torch.stack((x_axis, y_axis, z_axis), dim=-1)
    geometric_mask = (
        torch.linalg.vector_norm(c - ca, dim=-1) > 1e-8
    ) & (torch.linalg.vector_norm(torch.linalg.cross(c - ca, n - ca, dim=-1), dim=-1) > 1e-8)
    if mask is not None:
        if mask.shape != n.shape[:-1]:
            raise ValueError("frame mask must have shape (..., Nres)")
        geometric_mask = geometric_mask & mask
    return Rigid(rotation, ca), geometric_mask


def atoms_to_residue_local(frames: Rigid, atom_xyz: Tensor, atom_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Transform ``(..., Nres, Natom, 3)`` coordinates into residue frames."""

    if atom_xyz.shape[:-1] != atom_mask.shape or atom_xyz.shape[-1] != 3:
        raise ValueError("atom_xyz/mask shapes must be (..., Nres, Natom, 3)/(..., Nres, Natom)")
    if frames.translation.shape != atom_xyz.shape[:-2] + (3,):
        raise ValueError("frame batch/residue shape must match atom coordinates")
    delta = atom_xyz - frames.translation[..., :, None, :]
    local = torch.einsum("...nwi,...naw->...nai", frames.rotation, delta)
    return torch.where(atom_mask[..., None], local, torch.zeros_like(local)), atom_mask


def relative_residue_frames(frames: Rigid) -> tuple[Tensor, Tensor]:
    """Express every frame j in frame i as rotation and translation."""

    rotation_i_t = frames.rotation.transpose(-1, -2)
    relative_rotation = rotation_i_t[..., :, None, :, :] @ frames.rotation[..., None, :, :, :]
    displacement = frames.translation[..., None, :, :] - frames.translation[..., :, None, :]
    relative_translation = (
        rotation_i_t[..., :, None, :, :] @ displacement.unsqueeze(-1)
    ).squeeze(-1)
    return relative_rotation, relative_translation


def virtual_cb(n: Tensor, ca: Tensor, c: Tensor) -> Tensor:
    """Construct the AlphaFold/OpenFold-style virtual C-beta from N-CA-C."""

    b = ca - n
    c_vector = c - ca
    a = torch.linalg.cross(b, c_vector, dim=-1)
    return ca - 0.58273431 * a + 0.56802827 * b - 0.54067466 * c_vector


def distance_rbf(
    distances: Tensor,
    *,
    minimum: float = 2.0,
    maximum: float = 30.0,
    bins: int = 16,
) -> Tensor:
    """Gaussian distance expansion plus an explicit beyond-range bucket."""

    if bins < 2 or not minimum < maximum:
        raise ValueError("RBF requires at least two bins and minimum < maximum")
    centers = torch.linspace(minimum, maximum, bins, dtype=distances.dtype, device=distances.device)
    width = centers[1] - centers[0]
    gaussian = torch.exp(-((distances[..., None] - centers) / width).square())
    far_bucket = (distances > maximum).to(distances.dtype)[..., None]
    return torch.cat((gaussian, far_bucket), dim=-1)


def _angle(first: Tensor, vertex: Tensor, third: Tensor) -> Tensor:
    left = _normalize(first - vertex)
    right = _normalize(third - vertex)
    cosine = (left * right).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.acos(cosine)


def rosetta_orientations(
    n: Tensor,
    ca: Tensor,
    cb: Tensor,
    residue_mask: Tensor | None = None,
) -> RosettaOrientations:
    """Compute directed Rosetta omega/theta/phi features for all residue pairs."""

    if n.shape != ca.shape or n.shape != cb.shape or n.shape[-1] != 3:
        raise ValueError("N, CA, and CB must share shape (..., Nres, 3)")
    ca_i, ca_j = ca[..., :, None, :], ca[..., None, :, :]
    cb_i, cb_j = cb[..., :, None, :], cb[..., None, :, :]
    n_i = n[..., :, None, :].expand_as(cb_i + cb_j)
    pair_mask = ~torch.eye(n.shape[-2], dtype=torch.bool, device=n.device)
    pair_mask = pair_mask.expand(n.shape[:-2] + pair_mask.shape)
    if residue_mask is not None:
        if residue_mask.shape != n.shape[:-1]:
            raise ValueError("residue_mask must have shape (..., Nres)")
        pair_mask = pair_mask & residue_mask[..., :, None] & residue_mask[..., None, :]

    # A residue-level mask is not sufficient when callers carry partially
    # observed coordinates.  Degenerate N/CA/CB vectors make ``dihedral`` and
    # ``angle`` finite in the forward pass but can produce NaN gradients at
    # atan2(0, 0).  Exclude such pairs before evaluating the geometry.
    eps = torch.finfo(ca.dtype).eps
    residue_geometry_mask = (
        torch.linalg.vector_norm(ca - n, dim=-1) > eps
    ) & (torch.linalg.vector_norm(cb - ca, dim=-1) > eps)
    cb_i_to_j = cb[..., None, :, :] - cb[..., :, None, :]
    pair_mask = pair_mask & residue_geometry_mask[..., :, None] & residue_geometry_mask[..., None, :]
    pair_mask = pair_mask & (torch.linalg.vector_norm(cb_i_to_j, dim=-1) > eps)

    # Do not evaluate angle/dihedral functions on self pairs or padded pairs.
    # They contain coincident/zero vectors, so atan2(0, 0) can be finite in the
    # forward pass while producing NaN in the backward pass.  Masking only the
    # encoded output is too late: autograd has already traversed the degenerate
    # geometry.  Feed a fixed non-degenerate geometry to the inactive branch so
    # both forward and backward remain well-defined, while the active branch is
    # exactly unchanged.
    def constant(value: tuple[float, float, float], reference: Tensor) -> Tensor:
        tensor = torch.tensor(value, dtype=reference.dtype, device=reference.device)
        return tensor.expand_as(reference)

    def masked(actual: Tensor, inactive: Tensor) -> Tensor:
        return torch.where(pair_mask[..., None], actual, inactive)

    zero = constant((0.0, 0.0, 0.0), ca_i)
    one_x = constant((1.0, 0.0, 0.0), ca_i)
    one_xy = constant((1.0, 1.0, 0.0), ca_i)
    one_xyz = constant((1.0, 1.0, 1.0), ca_i)
    omega = dihedral(masked(ca_i, zero), masked(cb_i, one_x), masked(cb_j, one_xy), masked(ca_j, one_xyz))
    theta = dihedral(masked(n_i, zero), masked(ca_i, one_x), masked(cb_i, one_xy), masked(cb_j, one_xyz))
    phi = _angle(masked(ca_i, zero), masked(cb_i, one_x), masked(cb_j, one_xy))
    def encode(angle: Tensor) -> Tensor:
        return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)

    zeros = torch.zeros_like(encode(omega))
    return RosettaOrientations(
        omega_sin_cos=torch.where(pair_mask[..., None], encode(omega), zeros),
        theta_sin_cos=torch.where(pair_mask[..., None], encode(theta), zeros),
        phi_sin_cos=torch.where(pair_mask[..., None], encode(phi), zeros),
        mask=pair_mask,
    )
