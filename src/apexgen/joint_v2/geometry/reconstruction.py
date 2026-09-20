"""Ideal backbone and endpoint atom14 reconstruction for Joint-v2."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from apexgen.shared.geometry.backbone import place_atom
from apexgen.shared.geometry.joint_residue_constants import CHI_EXISTS, as_torch
from apexgen.shared.geometry.residue_constants import BACKBONE_GEOMETRY
from apexgen.shared.geometry.sidechain import build_atom14
from apexgen.joint_v2.contracts.contract import UnifiedComplexCondition
from apexgen.joint_v2.geometry.rotations import wrap_angle
from apexgen.joint_v2.contracts.state import JointFlowState


def ideal_backbone_local(*, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
    """Return ideal local N/CA/C coordinates for the Joint-v2 frame convention."""

    geometry = BACKBONE_GEOMETRY
    return torch.tensor(
        (
            (
                geometry.n_ca * math.cos(geometry.n_ca_c),
                geometry.n_ca * math.sin(geometry.n_ca_c),
                0.0,
            ),
            (0.0, 0.0, 0.0),
            (geometry.ca_c, 0.0, 0.0),
        ),
        device=device,
        dtype=dtype,
    )


def reconstruct_backbone(state: JointFlowState) -> Tensor:
    """Place ideal N/CA/C atoms independently in every residue frame."""

    with torch.autocast(device_type=state.translation.device.type, enabled=False):
        local = ideal_backbone_local(device=state.translation.device)
        return (
            torch.einsum("bnij,aj->bnai", state.rotation.float(), local)
            + state.translation.float()[..., None, :]
        )


def reconstruct_backbone_with_oxygen(
    state: JointFlowState,
    backbone_angles_sin_cos: Tensor,
    *,
    terminal_mask: Tensor | None = None,
) -> Tensor:
    """Return frame-derived N/CA/C plus O placed from the final predicted psi.

    The supplied residue frames remain the sole source of N/CA/C.  The angle
    head contributes only psi for carbonyl-oxygen placement.  Terminal
    residues use a deterministic zero-psi ideal geometry because terminal psi
    has no native torsion supervision.
    """

    expected = state.layout + (3, 2)
    if backbone_angles_sin_cos.shape != expected:
        raise ValueError("backbone_angles_sin_cos must be [batch, graph, 3, 2]")
    if not torch.is_floating_point(backbone_angles_sin_cos):
        raise TypeError("backbone angles must be floating point")
    if terminal_mask is not None and (
        terminal_mask.shape != state.layout or terminal_mask.dtype != torch.bool
    ):
        raise TypeError("terminal_mask must be bool [batch, graph]")
    with torch.autocast(device_type=state.translation.device.type, enabled=False):
        backbone = reconstruct_backbone(state)
        psi = torch.atan2(
            backbone_angles_sin_cos.float()[..., 1, 0],
            backbone_angles_sin_cos.float()[..., 1, 1],
        )
        if terminal_mask is not None:
            psi = torch.where(terminal_mask, 0.0, psi)
        geometry = BACKBONE_GEOMETRY
        oxygen = place_atom(
            backbone[..., 0, :],
            backbone[..., 1, :],
            backbone[..., 2, :],
            geometry.c_o,
            geometry.ca_c_o,
            wrap_angle(psi + math.pi),
        )
        return torch.cat((backbone, oxygen[..., None, :]), dim=-2)


def reconstruct_atom14(
    state: JointFlowState,
    condition: UnifiedComplexCondition,
    backbone_angles_sin_cos: Tensor,
    sidechain_angles_sin_cos: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build endpoint atom14 coordinates from final sequence, frames and chi.

    Returns unified-graph ``(atom14, atom14_mask, chi, chi_mask)`` tensors.
    Pocket and padding entries are forced to zero/false and are never used as
    generated peptide coordinates.
    """

    if state.layout != condition.layout:
        raise ValueError("state and condition layouts differ")
    expected = state.layout + (4, 2)
    if sidechain_angles_sin_cos.shape != expected:
        raise ValueError("sidechain_angles_sin_cos must be [batch, graph, 4, 2]")
    if not torch.is_floating_point(sidechain_angles_sin_cos):
        raise TypeError("sidechain angles must be floating point")
    peptide = condition.peptide_mask
    predicted_aatype = state.sequence_logits.argmax(-1)
    safe_aatype = torch.where(peptide, predicted_aatype, 0)
    chi = torch.atan2(
        sidechain_angles_sin_cos.float()[..., 0],
        sidechain_angles_sin_cos.float()[..., 1],
    )
    terminal = peptide & (
        condition.sequence_index
        == condition.peptide_mask.sum(-1)[:, None] - 1
    )
    backbone = reconstruct_backbone_with_oxygen(
        state,
        backbone_angles_sin_cos,
        terminal_mask=terminal,
    )
    atom14, atom14_mask = build_atom14(backbone, safe_aatype, chi)
    chi_mask = as_torch(CHI_EXISTS, device=state.translation.device, dtype=torch.bool)[
        safe_aatype
    ]
    atom14_mask = atom14_mask & peptide[..., None]
    chi_mask = chi_mask & peptide[..., None]
    atom14 = torch.where(atom14_mask[..., None], atom14, 0.0)
    chi = torch.where(chi_mask, chi, 0.0)
    return atom14, atom14_mask, chi, chi_mask


def pack_peptide_backbone(
    backbone: Tensor,
    condition: UnifiedComplexCondition,
    *,
    maximum_length: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Pack unified peptide nodes into sequence order without GPU scalar sync."""

    if backbone.shape[:2] != condition.layout or backbone.shape[-2:] != (3, 3):
        raise ValueError("backbone must be [batch, graph, N_CA_C, xyz]")
    if maximum_length is None:
        maximum_length = int(condition.peptide_mask.shape[1])
    if maximum_length <= 0:
        raise ValueError("maximum_length must be positive")
    packed = backbone.new_zeros(backbone.shape[0], maximum_length, 3, 3)
    mask = (
        torch.arange(maximum_length, device=backbone.device)[None]
        < condition.peptide_mask.sum(-1)[:, None]
    )
    batch_index = torch.arange(backbone.shape[0], device=backbone.device)[:, None]
    batch_index = batch_index.expand_as(condition.peptide_mask)
    peptide = condition.peptide_mask
    packed[batch_index[peptide], condition.sequence_index[peptide]] = backbone[peptide]
    return packed, mask
