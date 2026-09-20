"""Stepwise error decomposition for Joint-v2 endpoint rollouts."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from apexgen.joint_v2.data.batch import JointV2Batch
from apexgen.joint_v2.contracts.contract import JointEndpointPrediction
from apexgen.joint_v2.sampling.flow import INTEGRATION_STEPS, conditional_path, endpoint_step
from apexgen.joint_v2.geometry.rotations import so3_log
from apexgen.joint_v2.contracts.state import JointFlowState
from apexgen.joint_v2.evaluation.validation import block_endpoint_metrics_per_sample


def state_error_metrics_per_sample(
    observed: JointFlowState,
    expected: JointFlowState,
    batch: JointV2Batch,
) -> dict[str, dict[str, float]]:
    """Measure state error independently for each sample's peptide residues."""

    if observed.layout != batch.condition.layout or expected.layout != batch.condition.layout:
        raise ValueError("state and batch layouts differ")
    rows: dict[str, dict[str, float]] = {}
    for index, sample_id in enumerate(batch.sample_ids):
        peptide = batch.condition.peptide_mask[index]
        translation = (
            observed.translation[index, peptide] - expected.translation[index, peptide]
        )
        relative_rotation = (
            expected.rotation[index, peptide].transpose(-1, -2)
            @ observed.rotation[index, peptide]
        )
        sequence_delta = (
            observed.sequence_logits[index, peptide]
            - expected.sequence_logits[index, peptide]
        )
        rows[sample_id] = {
            "translation_rmse_angstrom": float(
                translation.square().sum(-1).mean().sqrt()
            ),
            "rotation_mean_degrees": float(
                so3_log(relative_rotation).norm(dim=-1).mean() * (180.0 / math.pi)
            ),
            "sequence_logit_rmse": float(sequence_delta.square().mean().sqrt()),
            "sequence_accuracy": float(
                (
                    observed.sequence_logits[index, peptide].argmax(-1)
                    == expected.sequence_logits[index, peptide].argmax(-1)
                )
                .float()
                .mean()
            ),
        }
    return rows


def _masked_vector_alignment(
    predicted: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    coordinate_rms: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if predicted.shape != target.shape or predicted.shape[:2] != mask.shape:
        raise ValueError("velocity vectors and peptide mask have inconsistent layouts")
    weight = mask[..., None].to(torch.float32)
    predicted = predicted.float() * weight
    target = target.float() * weight
    reduce_dimensions = tuple(range(1, predicted.ndim))
    dot = (predicted * target).sum(dim=reduce_dimensions)
    predicted_norm = predicted.square().sum(dim=reduce_dimensions).sqrt()
    target_norm = target.square().sum(dim=reduce_dimensions).sqrt()
    denominator = predicted_norm * target_norm
    epsilon = torch.finfo(torch.float32).eps
    cosine = torch.where(
        denominator > epsilon,
        dot / denominator.clamp_min(epsilon),
        torch.zeros_like(dot),
    ).clamp(-1.0, 1.0)
    ratio = torch.where(
        target_norm > epsilon,
        predicted_norm / target_norm.clamp_min(epsilon),
        torch.zeros_like(target_norm),
    )
    count = mask.float().sum(-1).clamp_min(1.0)
    if coordinate_rms:
        count = count * predicted.shape[-1]
    predicted_rms = (predicted.square().sum(dim=reduce_dimensions) / count).sqrt()
    target_rms = (target.square().sum(dim=reduce_dimensions) / count).sqrt()
    return cosine, ratio, predicted_rms, target_rms


def endpoint_velocity_metrics_per_sample(
    state: JointFlowState,
    predicted_endpoint: JointEndpointPrediction | JointFlowState,
    target_endpoint: JointFlowState,
    batch: JointV2Batch,
    time: Tensor,
) -> dict[str, Tensor]:
    """Compare predicted and target endpoint velocities from the same state."""

    if state.layout != batch.condition.layout or target_endpoint.layout != state.layout:
        raise ValueError("state, endpoint and batch layouts differ")
    if time.shape != (state.layout[0],):
        raise ValueError("time must be [batch]")
    remaining = (1.0 - time.float()).clamp_min(torch.finfo(torch.float32).eps)
    scalar = remaining[:, None, None]
    predicted_translation = (
        predicted_endpoint.translation.float() - state.translation.float()
    ) / scalar
    target_translation = (
        target_endpoint.translation.float() - state.translation.float()
    ) / scalar
    predicted_sequence = (
        predicted_endpoint.sequence_logits.float() - state.sequence_logits.float()
    ) / scalar
    target_sequence = (
        target_endpoint.sequence_logits.float() - state.sequence_logits.float()
    ) / scalar
    predicted_rotation = so3_log(
        state.rotation.float().transpose(-1, -2) @ predicted_endpoint.rotation.float()
    ) / scalar
    target_rotation = so3_log(
        state.rotation.float().transpose(-1, -2) @ target_endpoint.rotation.float()
    ) / scalar
    peptide = batch.condition.peptide_mask
    translation = _masked_vector_alignment(
        predicted_translation,
        target_translation,
        peptide,
        coordinate_rms=False,
    )
    rotation = _masked_vector_alignment(
        predicted_rotation,
        target_rotation,
        peptide,
        coordinate_rms=False,
    )
    sequence = _masked_vector_alignment(
        predicted_sequence,
        target_sequence,
        peptide,
        coordinate_rms=True,
    )
    degrees = 180.0 / math.pi
    return {
        "translation_velocity_cosine": translation[0],
        "translation_velocity_magnitude_ratio": translation[1],
        "translation_predicted_velocity_rms_angstrom": translation[2],
        "translation_target_velocity_rms_angstrom": translation[3],
        "rotation_velocity_cosine": rotation[0],
        "rotation_velocity_magnitude_ratio": rotation[1],
        "rotation_predicted_velocity_rms_degrees": rotation[2] * degrees,
        "rotation_target_velocity_rms_degrees": rotation[3] * degrees,
        "sequence_velocity_cosine": sequence[0],
        "sequence_velocity_magnitude_ratio": sequence[1],
        "sequence_predicted_velocity_rmse": sequence[2],
        "sequence_target_velocity_rmse": sequence[3],
    }


def _sample_tensor_rows(
    values: dict[str, Tensor],
    sample_ids: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    return {
        sample_id: {name: float(value[index]) for name, value in values.items()}
        for index, sample_id in enumerate(sample_ids)
    }


def _sample_block_rows(
    prediction: JointEndpointPrediction,
    batch: JointV2Batch,
) -> dict[str, dict[str, list[float]]]:
    metrics = block_endpoint_metrics_per_sample(prediction, batch)
    return {
        sample_id: {
            name: value[index].detach().cpu().tolist()
            for name, value in metrics.items()
        }
        for index, sample_id in enumerate(batch.sample_ids)
    }


def _frame_endpoint_from_block(
    prediction: JointEndpointPrediction,
    block: int | None,
) -> JointFlowState:
    if block is None:
        translation = prediction.translation
        rotation = prediction.rotation
    else:
        if prediction.block_translation is None or prediction.block_rotation is None:
            raise ValueError("fixed-block endpoint selection requires block frames")
        blocks = prediction.block_translation.shape[1]
        if block <= 0 or block > blocks:
            raise ValueError(f"frame endpoint block must lie in [1, {blocks}]")
        translation = prediction.block_translation[:, block - 1]
        rotation = prediction.block_rotation[:, block - 1]
    return JointFlowState(translation, rotation, prediction.sequence_logits)


@torch.no_grad()
def closed_loop_step_error_profile(
    model: nn.Module,
    batch: JointV2Batch,
    base: JointFlowState,
    *,
    frame_endpoint_block: int | None = None,
) -> list[dict[str, Any]]:
    """Decompose every rollout step into on-path, off-path and local errors."""

    runtime = model.module if hasattr(model, "module") else model
    runtime.eval()
    encoding = runtime.encode_complex(batch.condition)
    endpoint_target = batch.targets.endpoint_state(batch.condition)
    state = base
    rows: list[dict[str, Any]] = []
    for index in range(INTEGRATION_STEPS):
        time = torch.full(
            (len(batch.sample_ids),),
            index / INTEGRATION_STEPS,
            dtype=torch.float32,
            device=base.translation.device,
        )
        next_time = torch.full_like(time, (index + 1) / INTEGRATION_STEPS)
        oracle_state = conditional_path(
            base=base,
            targets=batch.targets,
            condition=batch.condition,
            time=time,
        )
        oracle_next = conditional_path(
            base=base,
            targets=batch.targets,
            condition=batch.condition,
            time=next_time,
        )
        rollout_input = state
        rollout_prediction = runtime.decode(
            rollout_input,
            time,
            batch.condition,
            encoding,
            return_intermediates=True,
        )
        oracle_prediction = runtime.decode(
            oracle_state,
            time,
            batch.condition,
            encoding,
            return_intermediates=True,
        )
        rollout_endpoint = _frame_endpoint_from_block(
            rollout_prediction,
            frame_endpoint_block,
        )
        oracle_endpoint = _frame_endpoint_from_block(
            oracle_prediction,
            frame_endpoint_block,
        )
        target_recovery_next = endpoint_step(
            rollout_input,
            endpoint_target,
            condition=batch.condition,
            time=time,
            next_time=next_time,
        )
        state = endpoint_step(
            rollout_input,
            rollout_endpoint,
            condition=batch.condition,
            time=time,
            next_time=next_time,
        )
        velocity = endpoint_velocity_metrics_per_sample(
            rollout_input,
            rollout_endpoint,
            endpoint_target,
            batch,
            time,
        )
        rows.append(
            {
                "step": index + 1,
                "query_time": index / INTEGRATION_STEPS,
                "next_time": (index + 1) / INTEGRATION_STEPS,
                "frame_endpoint_block": frame_endpoint_block,
                "rollout_input_path_deviation_before_step": state_error_metrics_per_sample(
                    rollout_input, oracle_state, batch
                ),
                "oracle_state_endpoint_error": state_error_metrics_per_sample(
                    oracle_endpoint, endpoint_target, batch
                ),
                "rollout_state_endpoint_error": state_error_metrics_per_sample(
                    rollout_endpoint, endpoint_target, batch
                ),
                "rollout_vs_oracle_endpoint_gap": state_error_metrics_per_sample(
                    rollout_endpoint, oracle_endpoint, batch
                ),
                "endpoint_velocity_alignment": _sample_tensor_rows(
                    velocity, batch.sample_ids
                ),
                "rollout_step_vs_target_recovery_step": state_error_metrics_per_sample(
                    state, target_recovery_next, batch
                ),
                "rollout_path_deviation_after_step": state_error_metrics_per_sample(
                    state, oracle_next, batch
                ),
                "oracle_state_block_endpoint_error": _sample_block_rows(
                    oracle_prediction, batch
                ),
                "rollout_state_block_endpoint_error": _sample_block_rows(
                    rollout_prediction, batch
                ),
            }
        )
    return rows
