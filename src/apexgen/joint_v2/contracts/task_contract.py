"""Explicit observation boundary for exploratory task-factorization experiments."""

from dataclasses import dataclass
import hashlib
import json

from torch import Tensor
import torch

from apexgen.joint_v2.contracts.contract import UnifiedComplexCondition
from apexgen.joint_v2.contracts.state import JointFlowState


TASKS = ("J", "S", "G_s", "G_0")
TASK_CONTRACT = {
    "schema": "apexgen.joint_v2.task_factorization.v1",
    "tasks": {"J": "joint", "S": "fixed_frames", "G_s": "fixed_sequence", "G_0": "no_sequence"},
    "geometry_target": "sidecar_frame_and_matching_ideal_N_CA_C",
    "auxiliary_angles": "disabled",
    "sequence_objective": "centered_logit_mse_plus_soft_ce",
    "fixed_observations": "clamp_each_refinement_and_solver_step",
    "encoder": "pocket_only_static_contract",
    "initialization": "scratch_only_v1",
}
TASK_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(TASK_CONTRACT, sort_keys=True).encode()
).hexdigest()


@dataclass(frozen=True)
class TaskObservation:
    """Only task-permitted observations; never carries loss-only native labels."""

    task: str
    pocket: UnifiedComplexCondition
    translation: Tensor | None = None
    rotation: Tensor | None = None
    sequence_logits: Tensor | None = None

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f"unknown task {self.task}")
        for field, required, shape in (
            ("translation", self.task == "S", self.pocket.layout + (3,)),
            ("rotation", self.task == "S", self.pocket.layout + (3, 3)),
            ("sequence_logits", self.task == "G_s", self.pocket.layout + (20,)),
        ):
            value = getattr(self, field)
            if required != (value is not None):
                raise ValueError(f"task {self.task}: observation {field} is required={required}")
            if value is not None and (
                value.shape != shape
                or value.dtype != torch.float32
                or value.device != self.pocket.residue_mask.device
            ):
                raise ValueError(f"invalid observed {field} layout/dtype/device")

    def clamp(self, state: JointFlowState) -> JointFlowState:
        if state.layout != self.pocket.layout:
            raise ValueError("task state layout differs from pocket")
        p = self.pocket.peptide_mask
        translation = self.translation if self.task == "S" else state.translation
        rotation = self.rotation if self.task == "S" else state.rotation
        sequence = self.sequence_logits if self.task == "G_s" else state.sequence_logits
        if self.task == "G_0":
            sequence = torch.zeros_like(state.sequence_logits)
        return JointFlowState(
            torch.where(p[..., None], translation, self.pocket.pocket_translation),
            torch.where(p[..., None, None], rotation, self.pocket.pocket_rotation),
            torch.where(p[..., None], sequence, 0.0),
        )
