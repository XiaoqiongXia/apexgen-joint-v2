"""Exploratory joint Dirichlet/translation/SO(3) runtime with an explicit contract."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from apexgen.joint_v2.contracts.simplex_state import SimplexJointState
from apexgen.joint_v2.contracts.task_contract import TaskObservation
from apexgen.joint_v2.geometry.rotations import so3_exp, so3_log
from apexgen.joint_v2.runtime.lineage import canonical_sha256
from apexgen.joint_v2.sampling.base import sample_haar_rotations
from apexgen.joint_v2.sampling.dirichlet import (
    DirichletField,
    alpha_at,
    conditional_simplex,
    sample_dirichlet,
)
from apexgen.joint_v2.training.simplex_weighting import SimplexLossWeights
from apexgen.shared.training.precision import network_autocast


CONTRACT = dict(
    schema="apexgen.joint_v2.simplex_codesign.v3",
    state="FP32 translation, rotation and 20-component simplex; zero sequence outside peptide",
    sequence_input="fixed linear p-1/20 adapter on peptide; no log or likelihood correction",
    path="Dir(1+(alpha(t)-1)*onehot(y)); alpha(t)=1+(alpha_max-1)*t",
    base="translation N(0,25 I), Haar rotation, independent Dir(1,...,1)",
    time="Uniform[0,1); same normalized time for geometry and sequence",
    objective=dict(
        final_translation="mean ||t_hat-t_true||^2 in angstrom^2; no fixed divisor",
        rotation_tangent="mean ||Log(R_t^T R_hat)-Log(R_t^T R_true)||^2 in radian^2; no fixed divisor",
        sequence_ce="mean negative log categorical probability of native amino acid",
    ),
    weights=dict(final_translation=1.0, rotation_tangent=1.0, sequence_ce=1.0),
    weighting="explicit positive fixed weights in run manifest; unit defaults; raw and weighted losses logged separately",
    sequence_solver="alpha-time exponential midpoint; posterior frozen per model step; C substeps <=0.05",
    coefficient="float64 derivative h=1e-4; FP32 hybrid log/linear grid 513 x 4097; x<1e-5 asymptotic; alpha in [1,32]",
    geometry_solver="endpoint Euler translation; geodesic endpoint rotation",
    rotation_gradients="full backpropagation across decoder blocks; no rotation detach",
    readout="extra categorical query at t=1; finite alpha_max is not a one-hot endpoint",
    backbone="independent residue R,t to N/CA/C; no covalent rebuilding/projection",
    compatibility="new model/checkpoint contract; legacy centered-state checkpoints rejected",
)
CONTRACT_SHA256 = canonical_sha256(CONTRACT)


def sample_simplex_base(condition, *, generator=None):
    mask = condition.peptide_mask
    device = mask.device
    translation = 5 * torch.randn(*condition.layout, 3, device=device, generator=generator)
    rotation = sample_haar_rotations(condition.layout, device=device, generator=generator)
    p = sample_dirichlet(torch.ones(*condition.layout, 20, device=device), generator=generator)
    return SimplexJointState(
        torch.where(mask[..., None], translation, condition.pocket_translation),
        torch.where(mask[..., None, None], rotation, condition.pocket_rotation),
        torch.where(mask[..., None], p, 0.0),
    )


def simplex_training_path(base, batch, time, *, alpha_max=8.0, generator=None):
    """Sample conditional marginals, not a straight segment to a one-hot vertex."""
    mask, targets = batch.condition.peptide_mask, batch.targets
    fraction = time[:, None, None]
    translation = (1 - fraction) * base.translation + fraction * targets.endpoint_translation
    rotation = base.rotation @ so3_exp(
        fraction * so3_log(base.rotation.transpose(-1, -2) @ targets.endpoint_rotation)
    )
    p = conditional_simplex(
        targets.endpoint_aatype, mask, time, alpha_max=alpha_max, generator=generator
    )
    return SimplexJointState(
        torch.where(mask[..., None], translation, batch.condition.pocket_translation),
        torch.where(mask[..., None, None], rotation, batch.condition.pocket_rotation),
        p,
    )


def simplex_losses(prediction, batch, state, *, weights=None):
    """Three raw per-example objectives plus explicit weighted contributions and total."""
    weights = SimplexLossWeights() if weights is None else weights
    mask = batch.condition.peptide_mask
    count = mask.sum(-1).clamp_min(1)
    with torch.autocast(device_type=mask.device.type, enabled=False):
        error = (
            (prediction.translation.float() - batch.targets.endpoint_translation.float())
            .square()
            .sum(-1)
        )
        translation = torch.where(mask, error, 0.0).sum(-1) / count
        inverse = state.rotation.float().transpose(-1, -2)
        rotation_error = (
            (
                so3_log(inverse @ prediction.rotation.float())
                - so3_log(inverse @ batch.targets.endpoint_rotation.float())
            )
            .square()
            .sum(-1)
        )
        rotation = torch.where(mask, rotation_error, 0.0).sum(-1) / count
        labels = torch.where(mask, batch.targets.endpoint_aatype, 0)
        ce = F.cross_entropy(
            prediction.sequence_logits.float().transpose(1, 2), labels, reduction="none"
        )
        ce = torch.where(mask, ce, 0.0).sum(-1) / count
    raw = dict(
        final_translation=translation,
        rotation_tangent=rotation,
        sequence_ce=ce,
    )
    weighted = {f"weighted_{name}": value * weights.as_dict()[name] for name, value in raw.items()}
    return dict(**raw, **weighted, total=sum(weighted.values()))


def simplex_fm_losses(
    model, batch, generator, *, alpha_max=8.0, precision="bfloat16", weights=None
):
    device = batch.condition.peptide_mask.device
    base = sample_simplex_base(batch.condition, generator=generator)
    time = torch.rand(len(batch.sample_ids), device=device, generator=generator)
    state = simplex_training_path(base, batch, time, alpha_max=alpha_max, generator=generator)
    with network_autocast(device, precision):
        prediction = model(state, time, TaskObservation("J", batch.condition))
    return simplex_losses(prediction, batch, state, weights=weights)


def simplex_step(state, prediction, condition, time, next_time, field):
    if time.shape != (state.layout[0],) or next_time.shape != time.shape:
        raise ValueError("Sampling times must be [batch]")
    alpha, next_alpha = alpha_at(time, field.alpha_max), alpha_at(next_time, field.alpha_max)
    if bool(((next_time < time) | (time >= 1)).any()):
        raise ValueError("Sampling requires 0 <= time <= next_time <= 1 and time < 1")
    mask = condition.peptide_mask
    fraction = ((next_time - time) / (1 - time))[:, None, None]
    translation = state.translation + fraction * (
        prediction.translation.float() - state.translation
    )
    rotation = state.rotation @ so3_exp(
        fraction * so3_log(state.rotation.transpose(-1, -2) @ prediction.rotation.float())
    )
    p = torch.zeros_like(state.sequence_probabilities)
    posterior = prediction.sequence_logits.float()[mask].softmax(-1)
    p[mask] = field.step(
        state.sequence_probabilities[mask],
        posterior,
        alpha[:, None].expand(mask.shape)[mask][:, None],
        next_alpha[:, None].expand(mask.shape)[mask][:, None],
    )
    return SimplexJointState(
        torch.where(mask[..., None], translation, condition.pocket_translation),
        torch.where(mask[..., None, None], rotation, condition.pocket_rotation),
        p,
    )


@dataclass(frozen=True)
class SimplexSample:
    state: SimplexJointState
    sequence_logits: torch.Tensor
    states: tuple
    predictions: tuple


@torch.no_grad()
def sample_simplex(
    model,
    condition,
    *,
    generator=None,
    alpha_max=8.0,
    steps=20,
    precision="float32",
    field=None,
    keep_trace=False,
):
    """No supervision targets accepted. Finite-alpha classification follows integration."""
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("Positive sampling step count required")
    if model.training:
        raise ValueError("Set model.eval() before sampling")
    field = (
        field
        if field is not None
        else DirichletField(alpha_max, device=condition.peptide_mask.device)
    )
    if field.alpha_max != alpha_max:
        raise ValueError("Coefficient field alpha_max mismatch")
    state = sample_simplex_base(condition, generator=generator)
    state.validate(condition)
    observation = TaskObservation("J", condition)
    device = state.translation.device
    states, predictions = [], []
    with network_autocast(device, precision):
        encoding = model.encode_complex(observation)
    for index in range(steps + 1):
        time = torch.full((state.layout[0],), index / steps, device=device)
        with network_autocast(device, precision):
            prediction = model.decode(state, time, observation, encoding)
        if prediction.sequence_logits.shape != state.sequence_probabilities.shape:
            raise ValueError("Categorical logits shape differs from simplex state")
        if not bool(torch.isfinite(prediction.sequence_logits[condition.peptide_mask]).all()):
            raise FloatingPointError(
                f"Nonfinite categorical logits at sampling time {index / steps}"
            )
        if keep_trace:
            states.append(state.to("cpu"))
            predictions.append(
                dict(
                    translation=prediction.translation.float().cpu(),
                    rotation=prediction.rotation.float().cpu(),
                    sequence_logits=prediction.sequence_logits.float().cpu(),
                )
            )
        if index < steps:
            state = simplex_step(
                state,
                prediction,
                condition,
                time,
                torch.full_like(time, (index + 1) / steps),
                field,
            )
            state.validate(condition)
    logits = torch.where(condition.peptide_mask[..., None], prediction.sequence_logits.float(), 0.0)
    return SimplexSample(state, logits, tuple(states), tuple(predictions))
