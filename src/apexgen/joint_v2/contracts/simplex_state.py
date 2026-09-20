"""Explicit probability-simplex state; never reinterpret it as centered logits."""

from dataclasses import dataclass

import torch
from torch import Tensor

from apexgen.joint_v2.contracts.state import JointFlowState, center_sequence_logits


@dataclass(frozen=True)
class SimplexJointState:
    translation: Tensor
    rotation: Tensor
    sequence_probabilities: Tensor

    @property
    def layout(self):
        return self.translation.shape[:-1]

    def to(self, device):
        return SimplexJointState(
            *(
                v.to(device=device, dtype=torch.float32)
                for v in (self.translation, self.rotation, self.sequence_probabilities)
            )
        )

    def detach(self):
        return SimplexJointState(
            *(v.detach() for v in (self.translation, self.rotation, self.sequence_probabilities))
        )

    def network_state(self):
        """Fixed linear input adapter p -> p-mean(p); no log, softmax, or scale."""
        return JointFlowState(
            self.translation, self.rotation, center_sequence_logits(self.sequence_probabilities)
        )

    def validate(self, condition):
        if self.layout != condition.layout or self.sequence_probabilities.shape != (
            *self.layout,
            20,
        ):
            raise ValueError("Simplex state layout mismatch")
        if self.sequence_probabilities.dtype != torch.float32:
            raise TypeError("Simplex probabilities must be FP32")
        self.network_state().validate(condition.residue_mask)
        p = self.sequence_probabilities
        mask = condition.peptide_mask
        if not bool(torch.isfinite(p).all()) or bool((p < 0).any()):
            raise ValueError("Simplex probabilities must be finite and nonnegative")
        if not bool(
            torch.allclose(p[mask].sum(-1), torch.ones_like(p[mask][..., 0]), atol=2e-5, rtol=0)
        ):
            raise ValueError("Peptide simplex rows must sum to one")
        if bool(p[~mask].count_nonzero()):
            raise ValueError("Pocket/padding simplex rows must be zero")
        if not torch.equal(self.translation[~mask], condition.pocket_translation[~mask]):
            raise ValueError("Observed translations changed")
        if not torch.equal(self.rotation[~mask], condition.pocket_rotation[~mask]):
            raise ValueError("Observed rotations changed")
