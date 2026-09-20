"""Dynamic sequence--structure state for Joint-v2 endpoint refinement."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch import Tensor


AMINO_ACID_TYPES = 20
UNKNOWN_AATYPE = AMINO_ACID_TYPES
SEQUENCE_EPSILON = 0.05
POCKET_ATOM_SLOTS = 38
BACKBONE_ANGLE_NAMES = ("phi", "psi", "omega")
BACKBONE_ANGLE_SLOTS = len(BACKBONE_ANGLE_NAMES)
SIDECHAIN_ANGLE_NAMES = ("chi1", "chi2", "chi3", "chi4")
SIDECHAIN_ANGLE_SLOTS = len(SIDECHAIN_ANGLE_NAMES)
TORSION_ANGLE_NAMES = BACKBONE_ANGLE_NAMES + SIDECHAIN_ANGLE_NAMES
TORSION_ANGLE_SLOTS = len(TORSION_ANGLE_NAMES)
DEBUG_INVARIANTS_ENV = "APEXGEN_JOINT_V2_DEBUG_INVARIANTS"


def debug_invariants_enabled() -> bool:
    """Return whether expensive value-level tensor checks are enabled."""

    value = os.environ.get(DEBUG_INVARIANTS_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{DEBUG_INVARIANTS_ENV} must be 0 or 1")
    return value == "1"


@dataclass(frozen=True)
class JointFlowState:
    """Unified-complex frames and centered sequence logits.

    Public translations are measured in angstrom and rotations map local
    coordinates to global coordinates. Geometry is deliberately stored in
    FP32 even when the learned network runs under BF16 autocast. Sequence
    logits are also stored and updated in FP32; all non-peptide entries are
    zero and only peptide entries are dynamic.
    """

    translation: Tensor  # float32 [B, N, 3]
    rotation: Tensor  # float32 [B, N, 3, 3]
    sequence_logits: Tensor  # float32 [B, N, 20], centered over amino-acid type

    def __post_init__(self) -> None:
        if self.translation.ndim != 3 or self.translation.shape[-1] != 3:
            raise ValueError("translation must be float32 [batch, graph, 3]")
        if self.rotation.shape != self.translation.shape[:-1] + (3, 3):
            raise ValueError("rotation must be float32 [batch, graph, 3, 3]")
        if self.sequence_logits.shape != self.translation.shape[:-1] + (AMINO_ACID_TYPES,):
            raise ValueError("sequence_logits must be float32 [batch, graph, 20]")
        if any(
            value.dtype != torch.float32
            for value in (self.translation, self.rotation, self.sequence_logits)
        ):
            raise TypeError("JointFlowState tensors must be stored in float32")
        if len({value.device for value in self.__dict__.values()}) != 1:
            raise ValueError("JointFlowState tensors must be on one device")
        if debug_invariants_enabled():
            self.validate()

    @property
    def layout(self) -> torch.Size:
        return self.translation.shape[:-1]

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "JointFlowState":
        return JointFlowState(
            translation=self.translation.to(
                device=device, dtype=torch.float32, non_blocking=non_blocking
            ),
            rotation=self.rotation.to(
                device=device, dtype=torch.float32, non_blocking=non_blocking
            ),
            sequence_logits=self.sequence_logits.to(
                device=device, dtype=torch.float32, non_blocking=non_blocking
            ),
        )

    def detach(self) -> "JointFlowState":
        return JointFlowState(
            self.translation.detach(),
            self.rotation.detach(),
            self.sequence_logits.detach(),
        )

    def validate(self, residue_mask: Tensor | None = None) -> None:
        """Validate finite proper rotations and centered logits at a boundary."""

        if residue_mask is None:
            residue_mask = torch.ones(self.layout, dtype=torch.bool, device=self.translation.device)
        if residue_mask.shape != self.layout or residue_mask.dtype != torch.bool:
            raise TypeError("residue_mask must be bool with the state layout")
        finite = (
            torch.isfinite(self.translation).all()
            & torch.isfinite(self.rotation).all()
            & torch.isfinite(self.sequence_logits).all()
        )
        identity = torch.eye(3, dtype=torch.float32, device=self.rotation.device)
        orthogonal_error = self.rotation.transpose(-1, -2) @ self.rotation - identity
        determinant = torch.linalg.det(self.rotation)
        invalid_rotation = (
            orthogonal_error.abs().amax(dim=(-2, -1))[residue_mask] > 2e-3
        ).any() | ((determinant[residue_mask] - 1.0).abs() > 2e-3).any()
        if not bool(finite.detach().cpu()):
            raise ValueError("JointFlowState tensors must be finite")
        if bool(invalid_rotation.detach().cpu()):
            raise ValueError("valid JointFlowState rotations must be proper SO(3) matrices")
        centered_error = self.sequence_logits.sum(-1).abs()
        if bool((centered_error[residue_mask] > 1e-4).any().detach().cpu()):
            raise ValueError("JointFlowState sequence logits must be centered")

    def validate_geometry(self, residue_mask: Tensor | None = None) -> None:
        """Compatibility spelling for callers validating the whole joint state."""

        self.validate(residue_mask)


def center_sequence_logits(value: Tensor) -> Tensor:
    """Project FP32 logits onto the zero-mean amino-acid logit hyperplane."""

    if value.ndim < 1 or value.shape[-1] != AMINO_ACID_TYPES:
        raise ValueError("sequence logits must end in 20 amino-acid channels")
    value = value.float()
    return value - value.mean(-1, keepdim=True)


def sequence_endpoint_logits(
    aatype: Tensor,
    peptide_mask: Tensor,
    *,
    epsilon: float = SEQUENCE_EPSILON,
) -> Tensor:
    """Map native amino-acid identities to finite centered endpoint logits."""

    if aatype.shape != peptide_mask.shape or aatype.dtype != torch.long:
        raise TypeError("aatype must be int64 with the peptide mask layout")
    if peptide_mask.dtype != torch.bool:
        raise TypeError("peptide_mask must be bool")
    if not 0.0 < float(epsilon) < 1.0:
        raise ValueError("sequence epsilon must lie in (0, 1)")
    selected = aatype[peptide_mask]
    if selected.numel() and bool(((selected < 0) | (selected >= AMINO_ACID_TYPES)).any()):
        raise ValueError("peptide aatype values must lie in [0, 20)")
    probability = torch.full(
        (*aatype.shape, AMINO_ACID_TYPES),
        float(epsilon) / AMINO_ACID_TYPES,
        dtype=torch.float32,
        device=aatype.device,
    )
    safe_aatype = torch.where(peptide_mask, aatype, 0)
    probability.scatter_(
        -1,
        safe_aatype[..., None],
        1.0 - float(epsilon) + float(epsilon) / AMINO_ACID_TYPES,
    )
    endpoint = center_sequence_logits(probability.log())
    return torch.where(peptide_mask[..., None], endpoint, 0.0)


def sequence_probabilities(sequence_logits: Tensor) -> Tensor:
    """Map finite centered logits back to the interior simplex."""

    if sequence_logits.ndim < 1 or sequence_logits.shape[-1] != AMINO_ACID_TYPES:
        raise ValueError("sequence logits must end in 20 amino-acid channels")
    if sequence_logits.dtype != torch.float32:
        raise TypeError("sequence logits must be float32")
    return sequence_logits.softmax(-1)


def decode_sequence_logits(sequence_logits: Tensor) -> Tensor:
    """Deterministically decode a joint sequence endpoint by argmax."""

    return sequence_probabilities(sequence_logits).argmax(-1)


def identity_state(
    layout: tuple[int, int] | torch.Size,
    *,
    device: torch.device | str | None = None,
) -> JointFlowState:
    """Create zero translations, identity rotations and zero sequence logits."""

    batch, graph = int(layout[0]), int(layout[1])
    translation = torch.zeros(batch, graph, 3, dtype=torch.float32, device=device)
    rotation = torch.eye(3, dtype=torch.float32, device=device).expand(batch, graph, 3, 3).clone()
    sequence_logits = torch.zeros(
        batch, graph, AMINO_ACID_TYPES, dtype=torch.float32, device=device
    )
    return JointFlowState(
        translation=translation,
        rotation=rotation,
        sequence_logits=sequence_logits,
    )
