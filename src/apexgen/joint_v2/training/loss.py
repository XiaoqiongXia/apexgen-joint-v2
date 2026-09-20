"""Sample-first sequence--structure endpoint losses for Joint-v2."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from apexgen.joint_v2.contracts.contract import (
    JointEndpointPrediction,
    PeptideNativeTargets,
    UnifiedComplexCondition,
)
from apexgen.joint_v2.geometry.rotations import (
    sidechain_pi_periodic_mask,
    so3_log,
    symmetry_aware_chi_squared_distance,
)
from apexgen.joint_v2.geometry.reconstruction import reconstruct_backbone
from apexgen.joint_v2.contracts.state import JointFlowState, SEQUENCE_EPSILON, sequence_endpoint_logits


@dataclass(frozen=True)
class JointEndpointLossWeights:
    trajectory_fape: float = 1.0
    final_translation: float = 1.0
    final_rotation: float = 1.0
    final_backbone: float = 1.0
    backbone_angle: float = 1.0
    backbone_angle_norm: float = 0.02
    sidechain_angle: float = 1.0
    sidechain_angle_norm: float = 0.02
    sequence_logit: float = 1.0
    sequence_soft_ce: float = 1.0
    sequence_total: float = 1.0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0.0
            ):
                raise ValueError(f"loss weight {name} must be finite and non-negative")
        if not any(value > 0 for value in self.__dict__.values()):
            raise ValueError("at least one endpoint loss weight must be positive")


def _masked_mean_per_sample(value: Tensor, mask: Tensor) -> Tensor:
    if value.shape != mask.shape or mask.dtype != torch.bool:
        raise TypeError("loss mask must be bool and match the loss tensor")
    weight = mask.to(value.dtype)
    count = weight.flatten(1).sum(-1)
    return (value * weight).flatten(1).sum(-1) / count.clamp_min(1.0)


def _masked_pair_mean_per_block(value: Tensor, mask: Tensor) -> Tensor:
    if value.ndim != 4 or mask.shape != value.shape[:1] + value.shape[2:]:
        raise ValueError("pair loss must be [batch, blocks, graph, graph]")
    weight = mask[:, None].to(value.dtype)
    count = weight.sum(dim=(-2, -1))
    return (value * weight).sum(dim=(-2, -1)) / count.clamp_min(1.0)


def _masked_mean_per_block(value: Tensor, mask: Tensor) -> Tensor:
    if value.ndim != 3 or mask.shape != value.shape[:1] + value.shape[2:]:
        raise ValueError("block loss must be [batch, blocks, graph]")
    weight = mask[:, None].to(value.dtype)
    count = weight.sum(-1)
    return (value * weight).sum(-1) / count.clamp_min(1.0)


def trajectory_fape_per_block(
    prediction: JointEndpointPrediction,
    condition: UnifiedComplexCondition,
    targets: PeptideNativeTargets,
    *,
    length_scale_angstrom: float = 10.0,
    distance_epsilon_angstrom_squared: float = 1e-4,
    peptide_clamp_angstrom: float | None = None,
    cross_clamp_angstrom: float | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return origin-point peptide/cross FAPE as ``[batch, blocks]`` tensors."""

    if prediction.block_translation is None or prediction.block_rotation is None:
        raise ValueError("trajectory FAPE requires intermediate block frames")
    if length_scale_angstrom <= 0 or distance_epsilon_angstrom_squared <= 0:
        raise ValueError("FAPE scale and epsilon must be positive")
    endpoint = targets.endpoint_state(condition)
    pred_translation = prediction.block_translation.float()
    pred_rotation = prediction.block_rotation.float()
    target_translation = endpoint.translation[:, None]
    target_rotation = endpoint.rotation[:, None]

    with torch.autocast(device_type=pred_translation.device.type, enabled=False):
        pred_displacement = pred_translation[:, :, None, :, :] - pred_translation[:, :, :, None, :]
        target_displacement = (
            target_translation[:, :, None, :, :] - target_translation[:, :, :, None, :]
        )
        pred_local = torch.einsum("bknji,bknmj->bknmi", pred_rotation, pred_displacement)
        target_local = torch.einsum("bknji,bknmj->bknmi", target_rotation, target_displacement)
        distance = torch.sqrt(
            (pred_local - target_local).square().sum(-1) + distance_epsilon_angstrom_squared
        )

    peptide = condition.peptide_mask
    pocket = condition.pocket_mask
    peptide_pair = peptide[:, :, None] & peptide[:, None, :]
    cross_pair = (pocket[:, :, None] & peptide[:, None, :]) | (
        peptide[:, :, None] & pocket[:, None, :]
    )
    peptide_distance = distance
    cross_distance = distance
    if peptide_clamp_angstrom is not None:
        peptide_distance = peptide_distance.clamp_max(peptide_clamp_angstrom)
    if cross_clamp_angstrom is not None:
        cross_distance = cross_distance.clamp_max(cross_clamp_angstrom)
    peptide_fape = _masked_pair_mean_per_block(
        peptide_distance / length_scale_angstrom,
        peptide_pair,
    )
    cross_fape = _masked_pair_mean_per_block(
        cross_distance / length_scale_angstrom,
        cross_pair,
    )
    return 0.5 * (peptide_fape + cross_fape), peptide_fape, cross_fape


def trajectory_fape_per_sample(
    prediction: JointEndpointPrediction,
    condition: UnifiedComplexCondition,
    targets: PeptideNativeTargets,
    **kwargs,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return block-averaged origin-point peptide/cross FAPE per sample."""

    values = trajectory_fape_per_block(prediction, condition, targets, **kwargs)
    return tuple(value.mean(1) for value in values)


def joint_endpoint_loss_per_sample(
    prediction: JointEndpointPrediction,
    condition: UnifiedComplexCondition,
    targets: PeptideNativeTargets,
    *,
    weights: JointEndpointLossWeights | None = None,
) -> dict[str, Tensor]:
    """Compute normalized endpoint terms without reducing the batch dimension."""

    condition.validate_model_input()
    weights = weights or JointEndpointLossWeights()
    peptide = condition.peptide_mask
    with torch.autocast(device_type=prediction.translation.device.type, enabled=False):
        trajectory, peptide_fape, cross_fape = trajectory_fape_per_sample(
            prediction, condition, targets
        )
        translation_error = (
            prediction.translation.float() - targets.endpoint_translation
        ).square().sum(-1) / 100.0
        translation = _masked_mean_per_sample(translation_error, peptide)

        relative_rotation = (
            targets.endpoint_rotation.transpose(-1, -2) @ prediction.rotation.float()
        )
        rotation_error = so3_log(relative_rotation).square().sum(-1) / (math.pi**2)
        rotation = _masked_mean_per_sample(rotation_error, peptide)

        predicted_backbone = reconstruct_backbone(
            JointFlowState(
                prediction.translation.float(),
                prediction.rotation.float(),
                prediction.sequence_logits.float(),
            )
        )
        backbone_error = (predicted_backbone - targets.backbone_xyz).square().sum(-1) / 100.0
        backbone = _masked_mean_per_sample(backbone_error, targets.backbone_atom_mask)

        angle_error = (
            prediction.backbone_angles_sin_cos.float() - targets.backbone_angles_sin_cos
        ).square().sum(-1) / 4.0
        angle = _masked_mean_per_sample(angle_error, targets.backbone_angle_mask)
        raw_angle_norm = torch.linalg.vector_norm(
            prediction.unnormalized_backbone_angles.float(), dim=-1
        )
        angle_norm = _masked_mean_per_sample(
            (raw_angle_norm - 1.0).abs(), targets.backbone_angle_mask
        )

        sidechain_periodic = sidechain_pi_periodic_mask(
            targets.endpoint_aatype,
            targets.sidechain_angle_mask,
        )
        sidechain_error = symmetry_aware_chi_squared_distance(
            prediction.sidechain_angles_sin_cos.float(),
            targets.sidechain_angles_sin_cos,
            sidechain_periodic,
        ) / 4.0
        sidechain_angle = _masked_mean_per_sample(
            sidechain_error,
            targets.sidechain_angle_mask,
        )
        raw_sidechain_norm = torch.linalg.vector_norm(
            prediction.unnormalized_sidechain_angles.float(), dim=-1
        )
        sidechain_angle_norm = _masked_mean_per_sample(
            (raw_sidechain_norm - 1.0).abs(),
            targets.sidechain_angle_mask,
        )

        target_logits = sequence_endpoint_logits(
            targets.endpoint_aatype,
            peptide,
            epsilon=SEQUENCE_EPSILON,
        )
        target_probability = target_logits.softmax(-1)
        final_sequence = prediction.sequence_logits.float()
        sequence_logit = _masked_mean_per_sample(
            (final_sequence - target_logits).square().mean(-1),
            peptide,
        )
        sequence_soft_ce = _masked_mean_per_sample(
            -(target_probability * final_sequence.log_softmax(-1)).sum(-1),
            peptide,
        )
        sequence_objective = (
            weights.sequence_logit * sequence_logit
            + weights.sequence_soft_ce * sequence_soft_ce
        )
        sequence_accuracy = _masked_mean_per_sample(
            (prediction.sequence_logits.argmax(-1) == targets.endpoint_aatype).float(),
            peptide,
        )
        sequence_hard_nll = _masked_mean_per_sample(
            -prediction.sequence_logits.float().log_softmax(-1).gather(
                -1,
                torch.where(peptide, targets.endpoint_aatype, 0)[..., None],
            )[..., 0],
            peptide,
        )
        final_log_probability = prediction.sequence_logits.float().log_softmax(-1)
        sequence_entropy = _masked_mean_per_sample(
            -(final_log_probability.exp() * final_log_probability).sum(-1),
            peptide,
        )

        per_sample_total = (
            weights.trajectory_fape * trajectory
            + weights.final_translation * translation
            + weights.final_rotation * rotation
            + weights.final_backbone * backbone
            + weights.backbone_angle * angle
            + weights.backbone_angle_norm * angle_norm
            + weights.sidechain_angle * sidechain_angle
            + weights.sidechain_angle_norm * sidechain_angle_norm
            + weights.sequence_total * sequence_objective
        )
        per_sample = {
            "trajectory_fape": trajectory,
            "peptide_fape": peptide_fape,
            "cross_fape": cross_fape,
            "final_translation": translation,
            "final_rotation": rotation,
            "final_backbone_n_ca_c": backbone,
            "backbone_angle": angle,
            "backbone_angle_norm": angle_norm,
            "sidechain_angle": sidechain_angle,
            "sidechain_angle_norm": sidechain_angle_norm,
            "sequence_logit": sequence_logit,
            "sequence_logit_rmse": sequence_logit.sqrt(),
            "sequence_soft_ce": sequence_soft_ce,
            "sequence_accuracy": sequence_accuracy,
            "sequence_hard_nll": sequence_hard_nll,
            "sequence_perplexity": sequence_hard_nll.exp(),
            "sequence_entropy": sequence_entropy,
            "total": per_sample_total,
        }
    return per_sample


def joint_endpoint_loss(
    prediction: JointEndpointPrediction,
    condition: UnifiedComplexCondition,
    targets: PeptideNativeTargets,
    *,
    weights: JointEndpointLossWeights | None = None,
) -> dict[str, Tensor]:
    """Compute normalized per-sample terms, combine them, then average the batch."""

    per_sample = joint_endpoint_loss_per_sample(
        prediction,
        condition,
        targets,
        weights=weights,
    )
    return {name: value.mean() for name, value in per_sample.items()}
