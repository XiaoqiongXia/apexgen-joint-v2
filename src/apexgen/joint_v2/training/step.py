"""Pure on-path training step for joint sequence--structure Joint-v2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from apexgen.joint_v2.contracts.contract import (
    JointEndpointPrediction,
    PeptideNativeTargets,
    UnifiedComplexCondition,
)
from apexgen.joint_v2.sampling.flow import conditional_path, sample_training_time
from apexgen.joint_v2.training.loss import JointEndpointLossWeights, joint_endpoint_loss
from apexgen.joint_v2.contracts.state import JointFlowState
from apexgen.shared.training.precision import network_autocast


@dataclass(frozen=True)
class JointV2TrainingOutput:
    time: Tensor
    state: JointFlowState
    prediction: JointEndpointPrediction
    losses: dict[str, Tensor]


def joint_endpoint_training_step(
    model: nn.Module,
    base: JointFlowState,
    condition: UnifiedComplexCondition,
    targets: PeptideNativeTargets,
    *,
    weights: JointEndpointLossWeights | None = None,
    generator: torch.Generator | None = None,
    network_precision: str = "float32",
    training_time_max: float = 1.0,
) -> JointV2TrainingOutput:
    """Sample an on-path time below ``training_time_max`` and supervise the endpoint."""

    time = sample_training_time(
        base.layout[0],
        maximum=training_time_max,
        device=base.translation.device,
        generator=generator,
    )
    state = conditional_path(
        base=base,
        targets=targets,
        condition=condition,
        time=time,
    )
    with network_autocast(state.translation.device, network_precision):
        prediction = model(state, time, condition, return_intermediates=True)
    losses = joint_endpoint_loss(prediction, condition, targets, weights=weights)
    return JointV2TrainingOutput(time, state, prediction, losses)
