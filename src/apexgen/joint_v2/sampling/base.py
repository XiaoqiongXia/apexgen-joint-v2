"""Endpoint-independent peptide base for sequence--structure Joint-v2."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from apexgen.joint_v2.contracts.contract import UnifiedComplexCondition
from apexgen.joint_v2.contracts.state import (
    AMINO_ACID_TYPES,
    JointFlowState,
    center_sequence_logits,
)


def quaternion_to_rotation(quaternion: Tensor) -> Tensor:
    """Convert normalized scalar-first quaternions to rotation matrices."""

    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def sample_haar_rotations(
    shape: tuple[int, ...] | torch.Size,
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> Tensor:
    """Sample Haar-uniform SO(3) matrices via normalized Gaussian quaternions."""

    quaternion = torch.randn(*shape, 4, dtype=torch.float32, device=device, generator=generator)
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    quaternion = torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)
    return quaternion_to_rotation(quaternion)


def sample_base_state(
    condition: UnifiedComplexCondition,
    *,
    translation_sigma_angstrom: float = 5.0,
    generator: torch.Generator | None = None,
) -> JointFlowState:
    """Sample peptide translation, rotation and centered sequence bases."""

    if (
        isinstance(translation_sigma_angstrom, bool)
        or not isinstance(translation_sigma_angstrom, (int, float))
        or not math.isfinite(float(translation_sigma_angstrom))
        or translation_sigma_angstrom <= 0
    ):
        raise ValueError("translation sigma must be finite and positive")
    peptide = condition.peptide_mask
    noise = torch.randn(
        *condition.layout,
        3,
        dtype=torch.float32,
        device=condition.residue_mask.device,
        generator=generator,
    )
    random_rotation = sample_haar_rotations(
        condition.layout,
        device=condition.residue_mask.device,
        generator=generator,
    )
    sequence_noise = center_sequence_logits(
        torch.randn(
            *condition.layout,
            AMINO_ACID_TYPES,
            dtype=torch.float32,
            device=condition.residue_mask.device,
            generator=generator,
        )
    )
    return JointFlowState(
        translation=torch.where(
            peptide[..., None],
            noise * float(translation_sigma_angstrom),
            condition.pocket_translation,
        ),
        rotation=torch.where(peptide[..., None, None], random_rotation, condition.pocket_rotation),
        sequence_logits=torch.where(peptide[..., None], sequence_noise, 0.0),
    )
