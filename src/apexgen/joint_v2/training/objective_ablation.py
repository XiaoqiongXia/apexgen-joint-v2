"""Exploratory objective ablations with fixed architecture, paths and solver."""

import math

import torch

from apexgen.joint_v2.geometry.rotations import so3_log
from apexgen.joint_v2.runtime.lineage import canonical_sha256
from apexgen.joint_v2.model.task_factorization import task_losses, task_objective_weights

ARMS = ("original", "sequence", "rotation", "both", "minimal")
OBJECTIVE_CONTRACT = dict(
    schema="apexgen.joint_v2.objective_ablation.v2",
    minimal="only final_translation, rotation_tangent and sequence_logit; unit weights",
    sequence="remove shared output soft-CE; retain centered endpoint MSE",
    rotation="mean_peptide ||Log(R_t^T R_hat)-Log(R_t^T R_1)||^2 / pi^2",
    time_weight="unchanged endpoint weighting; tangent displacement is (1-t)^2 weighted velocity loss",
    unchanged="J random position model, path and solver; historical four arms retain FAPE/backbone/translation",
    original="return original summed total unchanged; extra metrics do not enter backward",
)
OBJECTIVE_CONTRACT_SHA256 = canonical_sha256(OBJECTIVE_CONTRACT)


def objective_weights(arm):
    if arm not in ARMS:
        raise ValueError(f"unknown objective arm {arm}")
    if arm == "minimal":
        return dict(final_translation=1.0, rotation_tangent=1.0, sequence_logit=1.0)
    return {
        (
            "rotation_tangent" if key == "final_rotation" and arm in {"rotation", "both"} else key
        ): weight
        for key, weight in task_objective_weights("J").items()
        if key != "sequence_soft_ce" or arm not in {"sequence", "both"}
    }


def rotation_tangent_per_sample(predicted_rotation, target_rotation, current_rotation, mask):
    with torch.autocast(device_type=predicted_rotation.device.type, enabled=False):
        current_inverse = current_rotation.float().transpose(-1, -2)
        predicted = so3_log(current_inverse @ predicted_rotation.float())
        target = so3_log(current_inverse @ target_rotation.float())
        error = (predicted - target).square().sum(-1) / math.pi**2
        return torch.where(mask, error, 0.0).sum(-1) / mask.sum(-1).clamp_min(1)


def objective_losses(prediction, batch, state, arm):
    weights = objective_weights(arm)
    values = task_losses(prediction, batch, "J")
    values["original_total"] = values["total"]
    values["rotation_tangent"] = rotation_tangent_per_sample(
        prediction.rotation,
        batch.targets.endpoint_rotation,
        state.rotation,
        batch.condition.peptide_mask,
    )
    if arm != "original":
        values["total"] = sum(values[name] * weight for name, weight in weights.items())
    return values


def path_information(prediction, batch, state, time):
    """Comparable diagnostics in the actual sampler's current tangent space."""
    mask = batch.condition.peptide_mask
    target = batch.targets.endpoint_state(batch.condition)
    scale = (1 - time)[:, None, None]
    sequence_pred = (prediction.sequence_logits - state.sequence_logits) / scale
    sequence_true = (target.sequence_logits - state.sequence_logits) / scale
    inverse = state.rotation.transpose(-1, -2)
    rotation_pred = so3_log(inverse @ prediction.rotation) / scale
    rotation_true = so3_log(inverse @ target.rotation) / scale
    rows = []
    for b, m in enumerate(mask):
        row = dict(
            input_sequence_accuracy=float(
                (state.sequence_logits[b, m].argmax(-1) == batch.targets.endpoint_aatype[b, m])
                .float()
                .mean()
            )
        )
        for name, pred, true in [
            ("sequence", sequence_pred, sequence_true),
            ("rotation", rotation_pred, rotation_true),
        ]:
            pred, true = pred[b, m].flatten(), true[b, m].flatten()
            row[f"{name}_velocity_rmse"] = float((pred - true).square().mean().sqrt())
            row[f"{name}_velocity_cosine"] = float(
                torch.nn.functional.cosine_similarity(pred[None], true[None])
            )
        rows.append(row)
    return rows
