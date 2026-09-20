"""On-path interpolation and joint endpoint-parameterized rollout."""

from __future__ import annotations

from collections.abc import Callable
import math

import torch
from torch import Tensor

from apexgen.joint_v2.contracts.contract import (
    JointEndpointPrediction,
    PeptideNativeTargets,
    UnifiedComplexCondition,
)
from apexgen.joint_v2.geometry.rotations import so3_exp, so3_log
from apexgen.joint_v2.sampling.clocks import sequence_time
from apexgen.joint_v2.contracts.state import (
    JointFlowState,
    center_sequence_logits,
    debug_invariants_enabled,
)


INTEGRATION_STEPS = 20
INTEGRATION_STEP_SIZE = 1.0 / INTEGRATION_STEPS
MAX_MODEL_TIME = (INTEGRATION_STEPS - 1) / INTEGRATION_STEPS


def _batch_time(time: Tensor, state: JointFlowState) -> Tensor:
    if time.shape != (state.layout[0],):
        raise ValueError("time must be [batch]")
    time = time.to(device=state.translation.device, dtype=torch.float32)
    if debug_invariants_enabled() and not bool(
        (torch.isfinite(time) & (time >= 0.0) & (time <= 1.0)).all().detach().cpu()
    ):
        raise ValueError("time must contain finite values in [0, 1]")
    return time


def _geodesic_interpolate(start: Tensor, end: Tensor, fraction: Tensor) -> Tensor:
    relative = start.transpose(-1, -2) @ end
    return start @ so3_exp(fraction * so3_log(relative))


def conditional_path(
    *,
    base: JointFlowState,
    targets: PeptideNativeTargets,
    condition: UnifiedComplexCondition,
    time: Tensor,
    sequence_time_power: float = 1.0,
) -> JointFlowState:
    """Construct the translation/SO(3)/centered-logit on-path state ``x_t``."""

    if base.layout != condition.layout or targets.layout != condition.layout:
        raise ValueError("base, condition and target layouts differ")
    time = _batch_time(time, base)
    fraction = time[:, None, None]
    peptide = condition.peptide_mask
    endpoint = targets.endpoint_state(condition)
    translation = (1.0 - fraction) * base.translation + fraction * targets.endpoint_translation
    rotation = _geodesic_interpolate(base.rotation, targets.endpoint_rotation, fraction)
    sequence_fraction = sequence_time(time, sequence_time_power)[:, None, None]
    sequence = center_sequence_logits(
        (1.0 - sequence_fraction) * base.sequence_logits
        + sequence_fraction * endpoint.sequence_logits
    )
    return JointFlowState(
        translation=torch.where(peptide[..., None], translation, condition.pocket_translation),
        rotation=torch.where(peptide[..., None, None], rotation, condition.pocket_rotation),
        sequence_logits=torch.where(peptide[..., None], sequence, 0.0),
    )


def endpoint_step(
    state: JointFlowState,
    endpoint: JointEndpointPrediction | JointFlowState,
    *,
    condition: UnifiedComplexCondition,
    time: Tensor,
    next_time: Tensor,
    sequence_time_power: float = 1.0,
) -> JointFlowState:
    """Move the proper fraction of the remaining path toward a predicted endpoint."""

    if state.layout != condition.layout:
        raise ValueError("state and condition layouts differ")
    time = _batch_time(time, state)
    next_time = _batch_time(next_time, state)
    if debug_invariants_enabled() and not bool(
        ((next_time >= time) & (next_time <= 1.0)).all().detach().cpu()
    ):
        raise ValueError("next_time must lie between time and one")
    endpoint_translation = endpoint.translation.float()
    endpoint_rotation = endpoint.rotation.float()
    endpoint_sequence = endpoint.sequence_logits.float()
    if (
        endpoint_translation.shape != state.translation.shape
        or endpoint_rotation.shape != state.rotation.shape
        or endpoint_sequence.shape != state.sequence_logits.shape
    ):
        raise ValueError("endpoint and state layouts differ")
    remaining = (1.0 - time).clamp_min(torch.finfo(torch.float32).eps)
    alpha = ((next_time - time) / remaining).clamp(0.0, 1.0)[:, None, None]
    proposed_translation = state.translation + alpha * (endpoint_translation - state.translation)
    proposed_rotation = _geodesic_interpolate(state.rotation, endpoint_rotation, alpha)
    clock = sequence_time(time, sequence_time_power)
    next_clock = sequence_time(next_time, sequence_time_power)
    sequence_alpha = (
        (next_clock - clock) / (1 - clock).clamp_min(torch.finfo(torch.float32).eps)
    ).clamp(0, 1)[:, None, None]
    proposed_sequence = center_sequence_logits(
        state.sequence_logits + sequence_alpha * (endpoint_sequence - state.sequence_logits)
    )
    peptide = condition.peptide_mask
    return JointFlowState(
        translation=torch.where(
            peptide[..., None], proposed_translation, condition.pocket_translation
        ),
        rotation=torch.where(
            peptide[..., None, None], proposed_rotation, condition.pocket_rotation
        ),
        sequence_logits=torch.where(peptide[..., None], proposed_sequence, 0.0),
    )


def sample_training_time(
    batch_size: int,
    *,
    maximum: float = 1.0,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample FP32 ``Uniform[0, maximum)`` times for on-path training."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, (int, float))
        or not math.isfinite(maximum)
        or not 0.0 < maximum <= 1.0
    ):
        raise ValueError("maximum training time must lie in (0, 1]")
    return torch.rand(batch_size, dtype=torch.float32, device=device, generator=generator) * maximum


def integrate_endpoints(
    initial: JointFlowState,
    condition: UnifiedComplexCondition,
    endpoint_fn: Callable[[JointFlowState, Tensor], JointEndpointPrediction],
) -> JointFlowState:
    """Run the locked 20 endpoint queries at ``t=0.00,...,0.95``."""

    state = initial
    batch = initial.layout[0]
    for index in range(INTEGRATION_STEPS):
        time = torch.full(
            (batch,),
            index * INTEGRATION_STEP_SIZE,
            dtype=torch.float32,
            device=initial.translation.device,
        )
        next_time = torch.full_like(time, (index + 1) * INTEGRATION_STEP_SIZE)
        state = endpoint_step(
            state,
            endpoint_fn(state, time),
            condition=condition,
            time=time,
            next_time=next_time,
        )
    return state
