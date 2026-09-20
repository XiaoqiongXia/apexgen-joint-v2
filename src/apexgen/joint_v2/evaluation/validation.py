"""Random-time validation and fixed joint endpoint rollouts."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from apexgen.shared.geometry.joint_residue_constants import (
    ATOM14_AMBIGUITY_SWAP_INDEX,
    ATOM14_DISTANCE_LOWER_BOUND,
    as_torch,
)
from apexgen.shared.geometry.torsion import extract_backbone_torsions
from apexgen.joint_v2.sampling.base import sample_base_state
from apexgen.joint_v2.data.batch import JointV2Batch
from apexgen.joint_v2.contracts.contract import JointEndpointPrediction
from apexgen.joint_v2.sampling.flow import MAX_MODEL_TIME, conditional_path, integrate_endpoints
from apexgen.joint_v2.geometry.rotations import (
    sidechain_pi_periodic_mask,
    so3_log,
    symmetry_aware_chi_distance,
    wrap_angle,
)
from apexgen.joint_v2.training.loss import (
    JointEndpointLossWeights,
    joint_endpoint_loss,
    trajectory_fape_per_block,
)
from apexgen.joint_v2.geometry.reconstruction import (
    pack_peptide_backbone,
    reconstruct_atom14,
    reconstruct_backbone,
)
from apexgen.joint_v2.contracts.state import JointFlowState
from apexgen.joint_v2.training.step import joint_endpoint_training_step
from apexgen.joint_v2.evaluation.generation_quality import (
    generation_metrics, metric_groups, summarize_generation_candidates,
)
from apexgen.shared.training.precision import network_autocast


_LOSS_NAMES = (
    "total",
    "trajectory_fape",
    "peptide_fape",
    "cross_fape",
    "final_translation",
    "final_rotation",
    "final_backbone_n_ca_c",
    "backbone_angle",
    "backbone_angle_norm",
    "sidechain_angle",
    "sidechain_angle_norm",
    "sequence_logit",
    "sequence_logit_rmse",
    "sequence_soft_ce",
    "sequence_accuracy",
    "sequence_hard_nll",
    "sequence_perplexity",
    "sequence_entropy",
)


def _model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def block_endpoint_metrics_per_sample(
    prediction: JointEndpointPrediction,
    batch: JointV2Batch,
    *,
    peptide_clamp_angstrom: float = 10.0,
    cross_clamp_angstrom: float = 30.0,
) -> dict[str, Tensor]:
    """Return sample-first metrics for every refinement block."""

    if prediction.block_translation is None or prediction.block_rotation is None:
        raise ValueError("block metrics require intermediate frames")
    peptide = batch.condition.peptide_mask[:, None]
    weight = peptide.to(torch.float32)
    count = weight.sum(-1).clamp_min(1.0)
    translation_error = (
        (prediction.block_translation - batch.targets.endpoint_translation[:, None])
        .square()
        .sum(-1)
    )
    translation_rmsd = ((translation_error * weight).sum(-1) / count).sqrt()
    relative_rotation = (
        batch.targets.endpoint_rotation[:, None].transpose(-1, -2) @ prediction.block_rotation
    )
    rotation_degrees = (
        (so3_log(relative_rotation).norm(dim=-1) * weight).sum(-1) / count * (180.0 / math.pi)
    )
    sequence_recovery = (
        prediction.sequence_logits.argmax(-1)
        == batch.targets.endpoint_aatype
    ).float()
    final_sequence_recovery = (sequence_recovery[:, None] * weight).sum(-1) / count
    sequence_recovery = final_sequence_recovery.expand(
        -1, prediction.block_translation.shape[1]
    )
    fape, peptide_fape, cross_fape = trajectory_fape_per_block(
        prediction, batch.condition, batch.targets
    )
    clamped, clamped_peptide, clamped_cross = trajectory_fape_per_block(
        prediction,
        batch.condition,
        batch.targets,
        peptide_clamp_angstrom=peptide_clamp_angstrom,
        cross_clamp_angstrom=cross_clamp_angstrom,
    )
    return {
        "translation_rmsd_angstrom": translation_rmsd,
        "rotation_mean_degrees": rotation_degrees,
        "sequence_recovery": sequence_recovery,
        "fape": fape,
        "peptide_fape": peptide_fape,
        "cross_fape": cross_fape,
        "clamped_fape": clamped,
        "clamped_peptide_fape": clamped_peptide,
        "clamped_cross_fape": clamped_cross,
    }


@dataclass(frozen=True)
class JointV2RolloutCandidate:
    sample_index: int
    sample_id: str
    base_index: int
    base_seed: int
    angle_query_time: float
    backbone: Tensor
    aatype: Tensor
    atom14: Tensor
    atom14_mask: Tensor
    chi_radians: Tensor
    chi_mask: Tensor
    metrics: dict[str, float]

    @property
    def generation_quality(self):
        return metric_groups(self.metrics)['generation_quality']

    @property
    def reconstruction_diagnostics(self):
        return metric_groups(self.metrics)['reconstruction_diagnostics']


@torch.no_grad()
def validate_joint_v2_epoch(
    model: nn.Module,
    batches: Iterable[JointV2Batch],
    *,
    device: torch.device | str,
    translation_sigma_angstrom: float,
    seed: int,
    network_precision: str = "float32",
    weights: JointEndpointLossWeights | None = None,
    peptide_clamp_angstrom: float = 10.0,
    cross_clamp_angstrom: float = 30.0,
    training_time_max: float = 1.0,
) -> dict[str, float]:
    """Evaluate every validation sample with a deterministic random base/time."""

    runtime = _model(model)
    runtime.eval()
    target_device = torch.device(device)
    generator = torch.Generator(device=target_device).manual_seed(seed)
    totals = {name: 0.0 for name in _LOSS_NAMES}
    diagnostic_totals = {
        "clamped_fape": 0.0,
        "clamped_peptide_fape": 0.0,
        "clamped_cross_fape": 0.0,
        "backbone_angle_circular_mae_degrees": 0.0,
        "sidechain_angle_circular_mae_degrees": 0.0,
        **{f"chi{index}_circular_mae_degrees": 0.0 for index in range(1, 5)},
    }
    block_totals: dict[str, Tensor] | None = None
    time_total = 0.0
    sample_count = 0
    for cpu_batch in batches:
        batch = cpu_batch.to(target_device, non_blocking=target_device.type == "cuda")
        size = len(batch.sample_ids)
        base = sample_base_state(
            batch.condition,
            translation_sigma_angstrom=translation_sigma_angstrom,
            generator=generator,
        )
        output = joint_endpoint_training_step(
            runtime,
            base,
            batch.condition,
            batch.targets,
            weights=weights,
            generator=generator,
            network_precision=network_precision,
            training_time_max=training_time_max,
        )
        for name in _LOSS_NAMES:
            totals[name] += float(output.losses[name]) * size
        block_metrics = block_endpoint_metrics_per_sample(
            output.prediction,
            batch,
            peptide_clamp_angstrom=peptide_clamp_angstrom,
            cross_clamp_angstrom=cross_clamp_angstrom,
        )
        summed_blocks = {
            name: value.sum(0).double().cpu() for name, value in block_metrics.items()
        }
        if block_totals is None:
            block_totals = summed_blocks
        else:
            for name, value in summed_blocks.items():
                if value.shape != block_totals[name].shape:
                    raise ValueError("validation refinement-block count changed between batches")
                block_totals[name] += value
        diagnostic_totals["clamped_fape"] += float(
            block_metrics["clamped_fape"].mean(1).sum()
        )
        diagnostic_totals["clamped_peptide_fape"] += float(
            block_metrics["clamped_peptide_fape"].mean(1).sum()
        )
        diagnostic_totals["clamped_cross_fape"] += float(
            block_metrics["clamped_cross_fape"].mean(1).sum()
        )
        target_angles = batch.targets.backbone_angles_sin_cos
        angle_mask = batch.targets.backbone_angle_mask
        angle_distance = torch.acos(
            (output.prediction.backbone_angles_sin_cos * target_angles).sum(-1).clamp(-1.0, 1.0)
        )
        angle_weight = angle_mask.float()
        angle_per_sample = (
            (angle_distance * angle_weight).flatten(1).sum(-1)
            / angle_weight.flatten(1).sum(-1).clamp_min(1.0)
            * (180.0 / math.pi)
        )
        diagnostic_totals["backbone_angle_circular_mae_degrees"] += float(angle_per_sample.sum())
        sidechain_mask = batch.targets.sidechain_angle_mask
        sidechain_periodic = sidechain_pi_periodic_mask(
            batch.targets.endpoint_aatype,
            sidechain_mask,
        )
        sidechain_distance = symmetry_aware_chi_distance(
            output.prediction.sidechain_angles_sin_cos,
            batch.targets.sidechain_angles_sin_cos,
            sidechain_periodic,
        )
        sidechain_weight = sidechain_mask.float()
        sidechain_per_sample = (
            (sidechain_distance * sidechain_weight).flatten(1).sum(-1)
            / sidechain_weight.flatten(1).sum(-1).clamp_min(1.0)
            * (180.0 / math.pi)
        )
        diagnostic_totals["sidechain_angle_circular_mae_degrees"] += float(
            sidechain_per_sample.sum()
        )
        for index in range(sidechain_mask.shape[-1]):
            slot_mask = sidechain_mask[..., index]
            slot_weight = slot_mask.float()
            slot_per_sample = (
                (sidechain_distance[..., index] * slot_weight).sum(-1)
                / slot_weight.sum(-1).clamp_min(1.0)
                * (180.0 / math.pi)
            )
            diagnostic_totals[f"chi{index + 1}_circular_mae_degrees"] += float(
                slot_per_sample.sum()
            )
        time_total += float(output.time.sum())
        sample_count += size
    if sample_count == 0:
        raise ValueError("validation split is empty")
    if block_totals is None:
        raise RuntimeError("validation produced no refinement-block metrics")
    per_block = {
        f"metric_block_{block + 1:02d}_{name}": float(values[block] / sample_count)
        for name, values in block_totals.items()
        for block in range(values.shape[0])
    }
    return {
        **{f"loss_{name}": value / sample_count for name, value in totals.items()},
        **{f"metric_{name}": value / sample_count for name, value in diagnostic_totals.items()},
        **per_block,
        "sampled_time_mean": time_total / sample_count,
        "sample_count": float(sample_count),
    }


def _backbone_clashes(
    backbone: Tensor,
    condition: JointV2Batch,
    row: int,
    *,
    threshold: float,
) -> dict[str, float]:
    peptide_atoms = backbone.reshape(-1, 3).float()
    length, atoms = backbone.shape[:2]
    residue = torch.arange(length, device=backbone.device).repeat_interleave(atoms)
    atom = torch.arange(atoms, device=backbone.device).repeat(length)
    distance = torch.cdist(peptide_atoms, peptide_atoms)
    upper = torch.triu(torch.ones_like(distance, dtype=torch.bool), diagonal=1)
    adjacent_cn = (residue[None] == residue[:, None] + 1) & (atom[:, None] == 2) & (atom[None] == 0)
    internal = distance[upper & (residue[:, None] != residue[None]) & ~adjacent_cn]

    static = condition.condition
    pocket_atoms = static.pocket_atom_xyz[row][static.pocket_atom_mask[row]].float()
    interface = torch.cdist(peptide_atoms, pocket_atoms) if pocket_atoms.numel() else None
    internal_count = float((internal < threshold).sum())
    interface_count = float((interface < threshold).sum()) if interface is not None else 0.0
    return {
        "peptide_internal_clash_pair_count_below_2A": internal_count,
        "peptide_pocket_clash_pair_count_below_2A": interface_count,
        "peptide_internal_clash_free": float(internal_count == 0),
        "peptide_pocket_clash_free": float(interface_count == 0),
        "peptide_internal_minimum_nonbonded_distance_angstrom": (
            float(internal.min()) if internal.numel() else threshold
        ),
        "peptide_pocket_minimum_distance_angstrom": (
            float(interface.min()) if interface is not None and interface.numel() else threshold
        ),
    }


def _all_atom_clashes(
    atom14: Tensor,
    atom14_mask: Tensor,
    aatype: Tensor,
    batch: JointV2Batch,
    row: int,
    *,
    threshold: float,
) -> dict[str, float]:
    """Count peptide inter-residue and peptide--pocket heavy-atom clashes."""

    length, atoms = atom14_mask.shape
    if aatype.shape != (length,) or aatype.dtype != torch.long:
        raise TypeError("aatype must be int64 [length]")
    intra_distance = torch.cdist(atom14.float(), atom14.float())
    intra_lower_bound = as_torch(
        ATOM14_DISTANCE_LOWER_BOUND,
        device=atom14.device,
        dtype=torch.float32,
    )[aatype]
    intra_upper = torch.triu(
        torch.ones(atoms, atoms, dtype=torch.bool, device=atom14.device), diagonal=1
    )
    intra_valid = atom14_mask[..., :, None] & atom14_mask[..., None, :] & intra_upper
    intra_margin = intra_distance - intra_lower_bound
    intra_violations = intra_valid & (intra_margin < 0.0)
    intra_count = float(intra_violations.sum())
    residue = torch.arange(length, device=atom14.device)[:, None].expand(length, atoms)
    atom = torch.arange(atoms, device=atom14.device)[None].expand(length, atoms)
    peptide_atoms = atom14[atom14_mask].float()
    residue = residue[atom14_mask]
    atom = atom[atom14_mask]
    distance = torch.cdist(peptide_atoms, peptide_atoms)
    upper = torch.triu(torch.ones_like(distance, dtype=torch.bool), diagonal=1)
    same_residue = residue[:, None] == residue[None]
    adjacent_cn = (
        ((residue[:, None] + 1 == residue[None]) & (atom[:, None] == 2) & (atom[None] == 0))
        | ((residue[None] + 1 == residue[:, None]) & (atom[None] == 2) & (atom[:, None] == 0))
    )
    internal = distance[upper & ~same_residue & ~adjacent_cn]

    static = batch.condition
    pocket_atoms = static.pocket_atom_xyz[row][static.pocket_atom_mask[row]].float()
    interface = torch.cdist(peptide_atoms, pocket_atoms) if pocket_atoms.numel() else None
    internal_count = float((internal < threshold).sum())
    interface_count = float((interface < threshold).sum()) if interface is not None else 0.0
    return {
        "all_atom_peptide_intra_residue_violation_pair_count": intra_count,
        "all_atom_peptide_intra_residue_violation_free": float(intra_count == 0),
        "all_atom_peptide_intra_residue_minimum_distance_margin_angstrom": (
            float(intra_margin[intra_valid].min()) if bool(intra_valid.any()) else threshold
        ),
        "all_atom_peptide_inter_residue_clash_pair_count_below_2A": internal_count,
        "all_atom_peptide_pocket_clash_pair_count_below_2A": interface_count,
        "all_atom_peptide_inter_residue_clash_free": float(internal_count == 0),
        "all_atom_peptide_pocket_clash_free": float(interface_count == 0),
        "all_atom_peptide_total_clash_or_violation_pair_count": (
            intra_count + internal_count
        ),
        "all_atom_peptide_total_clash_or_violation_free": float(
            intra_count + internal_count == 0
        ),
        "all_atom_peptide_inter_residue_minimum_nonbonded_distance_angstrom": (
            float(internal.min()) if internal.numel() else threshold
        ),
        "all_atom_peptide_pocket_minimum_distance_angstrom": (
            float(interface.min()) if interface is not None and interface.numel() else threshold
        ),
    }


def _angle_radians(first: Tensor, center: Tensor, last: Tensor) -> Tensor:
    left = first - center
    right = last - center
    cosine = torch.nn.functional.cosine_similarity(left, right, dim=-1).clamp(-1.0, 1.0)
    return torch.acos(cosine)


def _kabsch_rmsd(predicted: Tensor, native: Tensor, mask: Tensor) -> Tensor:
    predicted = predicted[mask]
    native = native[mask]
    if predicted.shape[0] < 3:
        return predicted.new_zeros(())
    predicted = predicted - predicted.mean(0)
    native = native - native.mean(0)
    left, _, right = torch.linalg.svd(predicted.transpose(0, 1) @ native)
    correction = torch.eye(3, device=predicted.device)
    correction[-1, -1] = torch.where(
        torch.linalg.det(left @ right) < 0,
        predicted.new_tensor(-1.0),
        predicted.new_tensor(1.0),
    )
    aligned = predicted @ (left @ correction @ right)
    return (aligned - native).square().sum(-1).mean().sqrt()


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    weight = mask.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _masked_mean_if_any(value: Tensor, mask: Tensor) -> Tensor | None:
    """Return a masked mean, or ``None`` instead of a misleading empty-set zero."""

    if not bool(mask.any()):
        return None
    return value[mask].mean()


def _symmetry_corrected_atom14_rmsd(
    predicted: Tensor,
    predicted_mask: Tensor,
    native: Tensor,
    native_mask: Tensor,
    native_aatype: Tensor,
    comparable_residue: Tensor,
    atom_selection: Tensor,
) -> tuple[Tensor | None, Tensor]:
    """Return RMSD/count after choosing each native residue's best atom naming."""

    swap = as_torch(
        ATOM14_AMBIGUITY_SWAP_INDEX,
        device=native.device,
        dtype=torch.long,
    )[native_aatype]
    alternative = torch.gather(native, -2, swap[..., None].expand(-1, -1, 3))
    alternative_mask = torch.gather(native_mask, -1, swap)
    direct_mask = (
        predicted_mask
        & native_mask
        & comparable_residue[..., None]
        & atom_selection
    )
    alternative_valid = (
        predicted_mask
        & alternative_mask
        & comparable_residue[..., None]
        & atom_selection
    )
    direct_sse = ((predicted - native).square().sum(-1) * direct_mask).sum(-1)
    alternative_sse = ((predicted - alternative).square().sum(-1) * alternative_valid).sum(-1)
    direct_count = direct_mask.sum(-1)
    alternative_count = alternative_valid.sum(-1)
    direct_mse = direct_sse / direct_count.clamp_min(1)
    alternative_mse = alternative_sse / alternative_count.clamp_min(1)
    use_alternative = (alternative_count == direct_count) & (alternative_mse < direct_mse)
    selected_sse = torch.where(use_alternative, alternative_sse, direct_sse)
    selected_count = torch.where(use_alternative, alternative_count, direct_count)
    total_count = selected_count.sum()
    if not bool(total_count > 0):
        return None, total_count
    rmsd = (selected_sse.sum() / total_count.clamp_min(1)).sqrt()
    return rmsd, total_count


def _rollout_rows(
    generated: JointFlowState,
    initial: JointFlowState,
    batch: JointV2Batch,
    *,
    clash_distance_angstrom: float,
    angle_sin_cos: Tensor | None = None,
    sidechain_angle_sin_cos: Tensor | None = None,
) -> tuple[list[dict[str, float]], Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    if angle_sin_cos is None or sidechain_angle_sin_cos is None:
        raise ValueError("full-atom rollout validation requires final backbone and chi angles")
    backbone = reconstruct_backbone(generated)
    packed, residue_mask = pack_peptide_backbone(
        backbone,
        batch.condition,
        maximum_length=max(batch.peptide_lengths),
    )
    atom14, atom14_mask, chi, chi_mask = reconstruct_atom14(
        generated,
        batch.condition,
        angle_sin_cos,
        sidechain_angle_sin_cos,
    )
    maximum_length = packed.shape[1]
    packed_atom14 = atom14.new_zeros(atom14.shape[0], maximum_length, 14, 3)
    packed_atom14_mask = torch.zeros(
        atom14.shape[0], maximum_length, 14, dtype=torch.bool, device=atom14.device
    )
    packed_chi = chi.new_zeros(chi.shape[0], maximum_length, 4)
    packed_chi_mask = torch.zeros(
        chi.shape[0], maximum_length, 4, dtype=torch.bool, device=chi.device
    )
    rows: list[dict[str, float]] = []
    for row, length in enumerate(batch.peptide_lengths):
        peptide = batch.condition.peptide_mask[row]
        packed_atom14[row, :length] = atom14[row, peptide]
        packed_atom14_mask[row, :length] = atom14_mask[row, peptide]
        packed_chi[row, :length] = chi[row, peptide]
        packed_chi_mask[row, :length] = chi_mask[row, peptide]
        predicted = backbone[row, peptide]
        native = batch.targets.backbone_xyz[row, peptide]
        native_mask = batch.targets.backbone_atom_mask[row, peptide]
        backbone_rmsd = (
            ((predicted - native).square().sum(-1) * native_mask).sum()
            / native_mask.sum().clamp_min(1)
        ).sqrt()
        rotation_error = so3_log(
            batch.targets.endpoint_rotation[row, peptide].transpose(-1, -2)
            @ generated.rotation[row, peptide]
        ).norm(dim=-1)
        translation_rmse = (
            generated.translation[row, peptide]
            .sub(batch.targets.endpoint_translation[row, peptide])
            .square()
            .sum(-1)
            .mean()
            .sqrt()
        )
        cn = torch.linalg.vector_norm(
            packed[row, : length - 1, 2] - packed[row, 1:length, 0], dim=-1
        )
        native_cn_mask = native_mask[:-1, 2] & native_mask[1:, 0]
        native_ca_c_n_mask = native_mask[:-1, 1:].all(-1) & native_mask[1:, 0]
        native_c_n_ca_mask = native_mask[:-1, 2] & native_mask[1:, :2].all(-1)
        native_cn = torch.linalg.vector_norm(native[:-1, 2] - native[1:, 0], dim=-1)
        ca_c_n = _angle_radians(predicted[:-1, 1], predicted[:-1, 2], predicted[1:, 0])
        c_n_ca = _angle_radians(predicted[:-1, 2], predicted[1:, 0], predicted[1:, 1])
        native_ca_c_n = _angle_radians(native[:-1, 1], native[:-1, 2], native[1:, 0])
        native_c_n_ca = _angle_radians(native[:-1, 2], native[1:, 0], native[1:, 1])
        pocket = batch.condition.pocket_mask[row]
        pocket_change = max(
            float(
                (generated.translation[row, pocket] - initial.translation[row, pocket]).abs().max()
            ),
            float((generated.rotation[row, pocket] - initial.rotation[row, pocket]).abs().max()),
            float(
                (generated.sequence_logits[row, pocket] - initial.sequence_logits[row, pocket])
                .abs()
                .max()
            ),
        )
        metrics = {
            "translation_rmse_angstrom": float(translation_rmse),
            "rotation_mean_degrees": float(rotation_error.mean() * 180.0 / math.pi),
            "backbone_n_ca_c_rmsd_angstrom": float(backbone_rmsd),
            "peptide_kabsch_rmsd_angstrom": float(_kabsch_rmsd(predicted, native, native_mask)),
            "cn_bond_mean_angstrom": float(_masked_mean(cn, native_cn_mask)),
            "cn_bond_mae_to_native_angstrom": float(
                _masked_mean((cn - native_cn).abs(), native_cn_mask)
            ),
            "ca_c_n_angle_mae_to_native_degrees": float(
                _masked_mean((ca_c_n - native_ca_c_n).abs(), native_ca_c_n_mask)
                * 180.0
                / math.pi
            ),
            "c_n_ca_angle_mae_to_native_degrees": float(
                _masked_mean((c_n_ca - native_c_n_ca).abs(), native_c_n_ca_mask)
                * 180.0
                / math.pi
            ),
            "pocket_max_abs_change": pocket_change,
            "sequence_accuracy": float(
                (generated.sequence_logits[row, peptide].argmax(-1)
                 == batch.targets.endpoint_aatype[row, peptide]).float().mean()
            ),
        }
        sequence_nll = -generated.sequence_logits[row, peptide].log_softmax(-1).gather(
            -1, batch.targets.endpoint_aatype[row, peptide][..., None]
        )[..., 0].mean()
        metrics["sequence_hard_nll"] = float(sequence_nll)
        metrics["sequence_perplexity"] = float(sequence_nll.exp())
        predicted_angles = angle_sin_cos[row, peptide]
        target_angles = batch.targets.backbone_angles_sin_cos[row, peptide]
        angle_mask = batch.targets.backbone_angle_mask[row, peptide]
        cosine = (predicted_angles * target_angles).sum(-1).clamp(-1.0, 1.0)
        metrics["backbone_angle_circular_mae_degrees"] = float(
            _masked_mean(torch.acos(cosine), angle_mask) * 180.0 / math.pi
        )
        frame_angles, frame_mask = extract_backbone_torsions(
            predicted[:, 0], predicted[:, 1], predicted[:, 2]
        )
        head_angles = torch.atan2(predicted_angles[..., 0], predicted_angles[..., 1])
        consistency_mask = angle_mask & frame_mask
        metrics["angle_head_frame_consistency_mae_degrees"] = float(
            _masked_mean(wrap_angle(head_angles - frame_angles).abs(), consistency_mask)
            * 180.0
            / math.pi
        )

        predicted_aatype = generated.sequence_logits[row, peptide].argmax(-1)
        native_aatype = batch.targets.endpoint_aatype[row, peptide]
        sequence_match = predicted_aatype == native_aatype
        predicted_sidechain = sidechain_angle_sin_cos[row, peptide]
        target_sidechain = batch.targets.sidechain_angles_sin_cos[row, peptide]
        sidechain_mask = batch.targets.sidechain_angle_mask[row, peptide]
        sidechain_periodic = sidechain_pi_periodic_mask(native_aatype, sidechain_mask)
        sidechain_distance = symmetry_aware_chi_distance(
            predicted_sidechain,
            target_sidechain,
            sidechain_periodic,
        )
        metrics["sidechain_angle_valid_count"] = float(sidechain_mask.sum())
        comparable_chi = sidechain_mask & sequence_match[..., None]
        metrics["sidechain_angle_comparable_count"] = float(comparable_chi.sum())
        sidechain_mae = _masked_mean_if_any(sidechain_distance, comparable_chi)
        if sidechain_mae is not None:
            metrics["sidechain_angle_circular_mae_degrees"] = float(
                sidechain_mae * 180.0 / math.pi
            )
        for index in range(sidechain_mask.shape[-1]):
            valid_slot = sidechain_mask[..., index]
            comparable_slot = comparable_chi[..., index]
            metrics[f"chi{index + 1}_valid_count"] = float(valid_slot.sum())
            metrics[f"chi{index + 1}_comparable_count"] = float(comparable_slot.sum())
            slot_mae = _masked_mean_if_any(
                sidechain_distance[..., index], comparable_slot
            )
            if slot_mae is not None:
                metrics[f"chi{index + 1}_circular_mae_degrees"] = float(
                    slot_mae * 180.0 / math.pi
                )

        recovered_chi = (sidechain_distance <= math.radians(20.0)) & comparable_chi
        comparable_rotamer = comparable_chi.any(-1)
        recovered_rotamer = (~comparable_chi | recovered_chi).all(-1) & comparable_rotamer
        metrics["chi_rotamer_comparable_count"] = float(comparable_chi.sum())
        metrics["residue_rotamer_comparable_count"] = float(comparable_rotamer.sum())
        chi_recovery = _masked_mean_if_any(recovered_chi.float(), comparable_chi)
        if chi_recovery is not None:
            metrics["chi_rotamer_recovery_20deg"] = float(chi_recovery)
        residue_recovery = _masked_mean_if_any(
            recovered_rotamer.float(), comparable_rotamer
        )
        if residue_recovery is not None:
            metrics["residue_rotamer_recovery_20deg"] = float(residue_recovery)

        native_atom14 = batch.targets.atom14_xyz[row, peptide]
        native_atom14_mask = batch.targets.atom14_mask[row, peptide]
        all_atoms = torch.ones(14, dtype=torch.bool, device=predicted.device)
        sidechain_atoms = torch.arange(14, device=predicted.device) >= 4
        full_rmsd, full_count = _symmetry_corrected_atom14_rmsd(
            atom14[row, peptide],
            atom14_mask[row, peptide],
            native_atom14,
            native_atom14_mask,
            native_aatype,
            sequence_match,
            all_atoms,
        )
        sidechain_rmsd, sidechain_count = _symmetry_corrected_atom14_rmsd(
            atom14[row, peptide],
            atom14_mask[row, peptide],
            native_atom14,
            native_atom14_mask,
            native_aatype,
            sequence_match,
            sidechain_atoms,
        )
        metrics["all_atom_comparable_atom_count"] = float(full_count)
        metrics["sidechain_comparable_atom_count"] = float(sidechain_count)
        if full_rmsd is not None:
            metrics["all_atom_rmsd_symmetry_corrected_angstrom"] = float(full_rmsd)
        if sidechain_rmsd is not None:
            metrics["sidechain_heavy_atom_rmsd_symmetry_corrected_angstrom"] = float(
                sidechain_rmsd
            )
        metrics["all_atom_comparable_residue_fraction"] = float(sequence_match.float().mean())
        metrics.update(
            _backbone_clashes(packed[row, :length], batch, row, threshold=clash_distance_angstrom)
        )
        metrics.update(
            _all_atom_clashes(
                atom14[row, peptide],
                atom14_mask[row, peptide],
                predicted_aatype,
                batch,
                row,
                threshold=clash_distance_angstrom,
            )
        )
        metrics.update(generation_metrics(
            predicted, atom14[row, peptide], atom14_mask[row, peptide],
            predicted_aatype, batch.condition, row,
        ))
        rows.append(metrics)
    predicted_aatype = generated.sequence_logits.argmax(-1)
    packed_aatype = torch.zeros(
        predicted_aatype.shape[0], packed.shape[1], dtype=torch.long, device=packed.device
    )
    for row, length in enumerate(batch.peptide_lengths):
        packed_aatype[row, :length] = predicted_aatype[row, batch.condition.peptide_mask[row]]
    return (
        rows,
        packed,
        residue_mask,
        packed_aatype,
        packed_atom14,
        packed_atom14_mask,
        packed_chi,
        packed_chi_mask,
    )


def joint_v2_rollout_candidates(
    generated: JointFlowState,
    initial: JointFlowState,
    batch: JointV2Batch,
    *,
    base_index: int,
    base_seed: int,
    angle_query_time: float,
    clash_distance_angstrom: float = 2.0,
    angle_sin_cos: Tensor | None = None,
    sidechain_angle_sin_cos: Tensor | None = None,
) -> list[JointV2RolloutCandidate]:
    if not math.isfinite(angle_query_time) or not 0.0 <= angle_query_time <= 1.0:
        raise ValueError("angle_query_time must be finite and lie in [0, 1]")
    (
        rows,
        backbones,
        residue_masks,
        aatypes,
        atom14,
        atom14_masks,
        chi,
        chi_masks,
    ) = _rollout_rows(
        generated,
        initial,
        batch,
        clash_distance_angstrom=clash_distance_angstrom,
        angle_sin_cos=angle_sin_cos,
        sidechain_angle_sin_cos=sidechain_angle_sin_cos,
    )
    return [
        JointV2RolloutCandidate(
            sample_index=index,
            sample_id=batch.sample_ids[index],
            base_index=base_index,
            base_seed=base_seed,
            angle_query_time=float(angle_query_time),
            backbone=backbones[index, residue_masks[index]].detach().cpu(),
            aatype=aatypes[index, residue_masks[index]].detach().cpu(),
            atom14=atom14[index, residue_masks[index]].detach().cpu(),
            atom14_mask=atom14_masks[index, residue_masks[index]].detach().cpu(),
            chi_radians=chi[index, residue_masks[index]].detach().cpu(),
            chi_mask=chi_masks[index, residue_masks[index]].detach().cpu(),
            metrics=metrics,
        )
        for index, metrics in enumerate(rows)
    ]


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate an empty rollout panel")
    result = {"candidate_count": float(len(rows))}
    names = sorted({name for row in rows for name in row})
    for name in names:
        defined = [row[name] for row in rows if name in row]
        values = torch.tensor(defined, dtype=torch.float64)
        result[f"{name}_defined_candidate_count"] = float(len(defined))
        result[f"{name}_mean"] = float(values.mean())
        result[f"{name}_p50"] = float(torch.quantile(values, 0.5))
        result[f"{name}_p95"] = float(torch.quantile(values, 0.95))
        result[f"{name}_max"] = float(values.max())
    return result


@torch.no_grad()
def validate_joint_v2_rollout_panel(
    model: nn.Module,
    batch: JointV2Batch,
    *,
    translation_sigma_angstrom: float,
    seed: int,
    bases_per_target: int = 4,
    network_precision: str = "float32",
    clash_distance_angstrom: float = 2.0,
    artifact_callback: Callable[[JointV2RolloutCandidate], None] | None = None,
) -> dict:
    if isinstance(bases_per_target, bool) or not isinstance(bases_per_target, int) or bases_per_target < 1:
        raise ValueError('bases_per_target must be a positive integer')
    runtime = _model(model)
    runtime.eval()
    with network_autocast(batch.condition.residue_mask.device, network_precision):
        encoding = runtime.encode_complex(batch.condition)
    rows: list[dict[str, float]] = []
    all_candidates = []
    for base_index in range(bases_per_target):
        base_seed = seed + base_index
        generator = torch.Generator(device=batch.condition.residue_mask.device).manual_seed(
            base_seed
        )
        initial = sample_base_state(
            batch.condition,
            translation_sigma_angstrom=translation_sigma_angstrom,
            generator=generator,
        )

        last_prediction: JointEndpointPrediction | None = None

        def endpoint(state: JointFlowState, time: Tensor):
            nonlocal last_prediction
            with network_autocast(state.translation.device, network_precision):
                last_prediction = runtime.decode(
                    state,
                    time,
                    batch.condition,
                    encoding,
                    return_intermediates=False,
                )
            return last_prediction

        generated = integrate_endpoints(initial, batch.condition, endpoint)
        candidates = joint_v2_rollout_candidates(
            generated,
            initial,
            batch,
            base_index=base_index,
            base_seed=base_seed,
            angle_query_time=MAX_MODEL_TIME,
            clash_distance_angstrom=clash_distance_angstrom,
            angle_sin_cos=(
                None if last_prediction is None else last_prediction.backbone_angles_sin_cos
            ),
            sidechain_angle_sin_cos=(
                None if last_prediction is None else last_prediction.sidechain_angles_sin_cos
            ),
        )
        rows.extend(candidate.metrics for candidate in candidates)
        all_candidates.extend(candidates)
        if artifact_callback is not None:
            for candidate in candidates:
                artifact_callback(candidate)
    return {
        **_aggregate(rows),
        "target_count": float(len(batch.sample_ids)),
        "bases_per_target": float(bases_per_target),
        "clash_distance_angstrom": float(clash_distance_angstrom),
        "generation_quality": summarize_generation_candidates(all_candidates),
        "reconstruction_diagnostics": _aggregate([c.reconstruction_diagnostics for c in all_candidates]),
        "legacy_flat_metrics_role": "mixed_diagnostics_not_a_generation_ranking_score",
    }


@torch.no_grad()
def validate_joint_v2_batch(
    model: nn.Module,
    batch: JointV2Batch,
    *,
    translation_sigma_angstrom: float,
    seed: int,
    network_precision: str = "float32",
    times: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 0.95),
    weights: JointEndpointLossWeights | None = None,
) -> dict[str, float]:
    """Fixed-time loss grid plus one closed-loop rollout diagnostic."""

    runtime = _model(model)
    runtime.eval()
    generator = torch.Generator(device=batch.condition.residue_mask.device).manual_seed(seed)
    initial = sample_base_state(
        batch.condition,
        translation_sigma_angstrom=translation_sigma_angstrom,
        generator=generator,
    )
    with network_autocast(initial.translation.device, network_precision):
        encoding = runtime.encode_complex(batch.condition)
    rows: list[dict[str, Tensor]] = []
    for value in times:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("validation time must lie in [0, 1]")
        time = torch.full(
            (initial.layout[0],),
            float(value),
            dtype=torch.float32,
            device=initial.translation.device,
        )
        state = conditional_path(
            base=initial, targets=batch.targets, condition=batch.condition, time=time
        )
        with network_autocast(state.translation.device, network_precision):
            prediction = runtime.decode(
                state, time, batch.condition, encoding, return_intermediates=True
            )
        rows.append(
            joint_endpoint_loss(prediction, batch.condition, batch.targets, weights=weights)
        )
    grid = {
        f"grid_{name}": float(torch.stack([row[name] for row in rows]).mean())
        for name in _LOSS_NAMES
    }

    def endpoint(state: JointFlowState, time: Tensor):
        with network_autocast(state.translation.device, network_precision):
            return runtime.decode(
                state, time, batch.condition, encoding, return_intermediates=False
            )

    last_prediction: JointEndpointPrediction | None = None

    def endpoint_with_angles(state: JointFlowState, time: Tensor):
        nonlocal last_prediction
        last_prediction = endpoint(state, time)
        return last_prediction

    generated = integrate_endpoints(initial, batch.condition, endpoint_with_angles)
    rollout_rows, *_ = _rollout_rows(
        generated,
        initial,
        batch,
        clash_distance_angstrom=2.0,
        angle_sin_cos=(
            None if last_prediction is None else last_prediction.backbone_angles_sin_cos
        ),
        sidechain_angle_sin_cos=(
            None if last_prediction is None else last_prediction.sidechain_angles_sin_cos
        ),
    )
    return {**grid, **{f"rollout_{k}": v for k, v in _aggregate(rollout_rows).items()}}
