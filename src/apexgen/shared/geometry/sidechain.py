"""Differentiable atom14 reconstruction and chi extraction."""

from __future__ import annotations

import torch
from torch import Tensor

from apexgen.shared.geometry.joint_residue_constants import (
    ATOM14_GROUP, ATOM14_LOCAL_POSITIONS, ATOM14_MASK, ATOM14_NAMES, CHI_ATOM_NAMES, CHI_EXISTS,
    RIGID_GROUP_DEFAULT_FRAMES, as_torch,
)
from apexgen.shared.geometry.torsion import dihedral


def _rotation_x(angle: Tensor) -> Tensor:
    zero, one = torch.zeros_like(angle), torch.ones_like(angle)
    cos, sin = torch.cos(angle), torch.sin(angle)
    return torch.stack((one, zero, zero, zero, cos, -sin, zero, sin, cos), dim=-1).reshape(*angle.shape, 3, 3)


def _compose(rotation: Tensor, translation: Tensor, child: Tensor) -> tuple[Tensor, Tensor]:
    """Compose a transform with homogeneous child frame(s)."""

    child_r, child_t = child[..., :3, :3], child[..., :3, 3]
    return rotation @ child_r, (rotation @ child_t.unsqueeze(-1)).squeeze(-1) + translation


def _backbone_frame(backbone: Tensor) -> tuple[Tensor, Tensor]:
    """Return local OpenFold-backbone frame with CA origin for N/CA/C coordinates."""

    n, ca, c = backbone.unbind(dim=-2)[:3]
    eps = torch.finfo(backbone.dtype).eps
    x = c - ca
    x = x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(eps)
    y = n - ca
    y = y - (x * y).sum(dim=-1, keepdim=True) * x
    y = y / torch.linalg.vector_norm(y, dim=-1, keepdim=True).clamp_min(eps)
    z = torch.linalg.cross(x, y, dim=-1)
    return torch.stack((x, y, z), dim=-1), ca


def build_atom14(backbone: Tensor, aatype: Tensor, chi: Tensor) -> tuple[Tensor, Tensor]:
    """Build atom14 coordinates from an N/CA/C/O backbone and four chi angles.

    ``aatype`` is only an endpoint/loss-side identity; callers must not feed it
    into the denoising decoder.  The returned mask is the standard atom14 mask.
    """

    if backbone.shape[-2:] != (4, 3):
        raise ValueError("backbone must have shape (..., L, 4, 3)")
    if aatype.shape != backbone.shape[:-2] or chi.shape != backbone.shape[:-2] + (4,):
        raise ValueError("aatype/chi must agree with backbone batch and residue axes")
    if aatype.dtype != torch.long or (aatype.numel() and (aatype.min() < 0 or aatype.max() >= 20)):
        raise ValueError("aatype must contain values in [0, 20)")
    base_r, base_t = _backbone_frame(backbone)
    frames = as_torch(RIGID_GROUP_DEFAULT_FRAMES, device=backbone.device, dtype=backbone.dtype)[aatype]
    group_r = {0: base_r}
    group_t = {0: base_t}
    # group 4 is defined in the backbone frame; groups 5--7 are chained from
    # the preceding chi group.  Direct torsion values are relative rotations.
    for group in range(4, 8):
        parent = 0 if group == 4 else group - 1
        default_r, default_t = _compose(group_r[parent], group_t[parent], frames[..., group, :, :])
        chi_index = group - 4
        group_r[group] = default_r @ _rotation_x(chi[..., chi_index])
        group_t[group] = default_t
    all_r = torch.stack(tuple(group_r.get(group, base_r) for group in range(8)), dim=-3)
    all_t = torch.stack(tuple(group_t.get(group, base_t) for group in range(8)), dim=-2)
    group = as_torch(ATOM14_GROUP, device=backbone.device, dtype=torch.long)[aatype]
    local = as_torch(ATOM14_LOCAL_POSITIONS, device=backbone.device, dtype=backbone.dtype)[aatype]
    gather_r = torch.gather(all_r, -3, group[..., None, None].expand(*group.shape, 3, 3))
    gather_t = torch.gather(all_t, -2, group[..., None].expand(*group.shape, 3))
    xyz = (gather_r @ local.unsqueeze(-1)).squeeze(-1) + gather_t
    # Preserve supplied backbone exactly; the rigid table is only used for CB
    # and side-chain atoms.
    xyz[..., :4, :] = backbone
    mask = as_torch(ATOM14_MASK, device=backbone.device, dtype=torch.bool)[aatype]
    return xyz * mask[..., None], mask


def extract_chi(atom14: Tensor, atom14_mask: Tensor, aatype: Tensor) -> tuple[Tensor, Tensor]:
    """Extract four chi angles and their observation mask from atom14 coordinates."""

    if atom14.shape[-2:] != (14, 3) or atom14_mask.shape != atom14.shape[:-1]:
        raise ValueError("atom14 and atom14_mask must have shape (..., L, 14, 3)/(..., L, 14)")
    if aatype.shape != atom14.shape[:-2]:
        raise ValueError("aatype must agree with atom14")
    angles = torch.zeros(*aatype.shape, 4, dtype=atom14.dtype, device=atom14.device)
    observed = torch.zeros_like(angles, dtype=torch.bool)
    for residue, chis in enumerate(CHI_ATOM_NAMES):
        selected = aatype == residue
        names = ATOM14_NAMES[residue]
        for index, atoms in enumerate(chis):
            ids = [names.index(atom) for atom in atoms]
            valid = selected
            for atom_index in ids:
                valid = valid & atom14_mask[..., atom_index]
            value = dihedral(*(atom14[..., atom_index, :] for atom_index in ids))
            angles[..., index] = torch.where(selected, value, angles[..., index])
            observed[..., index] = torch.where(selected, valid, observed[..., index])
    exists = as_torch(CHI_EXISTS, device=atom14.device, dtype=torch.bool)[aatype]
    return angles, observed & exists
