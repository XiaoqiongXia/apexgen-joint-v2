"""Per-objective gradient attribution for Joint-v2 endpoint refinement."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import torch
from torch import Tensor, nn

from apexgen.joint_v2.data.batch import JointV2Batch
from apexgen.joint_v2.contracts.contract import JointEndpointPrediction
from apexgen.joint_v2.training.loss import joint_endpoint_loss_per_sample


WEIGHTED_OBJECTIVES = {
    "trajectory_fape": 1.0,
    "final_translation": 1.0,
    "final_rotation": 1.0,
    "final_backbone_n_ca_c": 1.0,
    "backbone_angle": 1.0,
    "backbone_angle_norm": 0.02,
    "sidechain_angle": 1.0,
    "sidechain_angle_norm": 0.02,
    "sequence_logit": 1.0,
    "sequence_soft_ce": 1.0,
}

_BASE_PARAMETER_PREFIXES = {
    "encoder": ("encoder.",),
    "shared_input": ("decoder.structure_module.linear_in.",),
    "shared_norms": (
        "decoder.structure_module.layer_norm_s.",
        "decoder.structure_module.layer_norm_z.",
        "decoder.structure_module.layer_norm_ipa.",
    ),
    "sequence_projection": (
        "decoder.structure_module.sequence_norm.",
        "decoder.structure_module.sequence_projection.",
    ),
    "time_conditioning": (
        "decoder.structure_module.time_conditioner.",
        "decoder.structure_module.time_film.",
    ),
    "ipa": ("decoder.structure_module.ipa.",),
    "transition": ("decoder.structure_module.transition.",),
    "backbone_update": ("decoder.structure_module.backbone_update.",),
    "sequence_update": (
        "decoder.structure_module.sequence_update.",
        "decoder.structure_module.final_sequence_resnet.",
    ),
    "sequence_latent_update": (
        "decoder.structure_module.sequence_latent_transition.",
    ),
    "angle_head": ("decoder.structure_module.angle_resnet.",),
}


def _cosine(dot: float, first_norm: float, second_norm: float) -> float | None:
    if first_norm == 0.0 or second_norm == 0.0:
        return None
    value = dot / (first_norm * second_norm)
    return max(-1.0, min(1.0, value))


def _gradient_gram(
    gradients: list[tuple[Tensor | None, ...]],
    indices: tuple[int, ...],
) -> list[list[float]]:
    reference = next(
        (
            gradient[index]
            for gradient in gradients
            for index in indices
            if gradient[index] is not None
        ),
        None,
    )
    if reference is None:
        return [[0.0] * len(gradients) for _ in gradients]
    gram = torch.zeros(
        len(gradients), len(gradients), dtype=torch.float32, device=reference.device
    )
    for index in indices:
        present = [gradient[index] for gradient in gradients]
        template = next((value for value in present if value is not None), None)
        if template is None:
            continue
        matrix = torch.stack(
            [torch.zeros_like(template) if value is None else value for value in present]
        ).flatten(1)
        gram = gram + matrix @ matrix.transpose(0, 1)
    result = gram.double().cpu().tolist()
    if any(not math.isfinite(value) for row in result for value in row):
        raise RuntimeError("non-finite gradient Gram matrix")
    return result


def _parameter_groups(names: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
    groups = {
        group: tuple(
            index
            for index, name in enumerate(names)
            if any(name.startswith(prefix) for prefix in prefixes)
        )
        for group, prefixes in _BASE_PARAMETER_PREFIXES.items()
    }
    missing = [name for name, indices in groups.items() if not indices]
    if missing:
        raise RuntimeError(f"gradient parameter groups are empty: {missing}")
    shared_names = {
        "shared_input",
        "shared_norms",
        "sequence_projection",
        "sequence_latent_update",
        "time_conditioning",
        "ipa",
        "transition",
    }
    groups["shared_refinement"] = tuple(
        sorted({index for name in shared_names for index in groups[name]})
    )
    groups["all_profiled"] = tuple(sorted({index for indices in groups.values() for index in indices}))
    return groups


def joint_objective_gradient_attribution(
    model: nn.Module,
    prediction: JointEndpointPrediction,
    batch: JointV2Batch,
) -> dict[str, Any]:
    """Return weighted per-loss norms, alignments, and conflicts for one case."""

    parameter_items = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and any(
            name.startswith(prefix)
            for prefixes in _BASE_PARAMETER_PREFIXES.values()
            for prefix in prefixes
        )
    )
    if not parameter_items:
        raise RuntimeError("gradient attribution selected no model parameters")
    parameter_names = tuple(name for name, _ in parameter_items)
    parameters = tuple(parameter for _, parameter in parameter_items)
    groups = _parameter_groups(parameter_names)
    per_sample_losses = joint_endpoint_loss_per_sample(
        prediction,
        batch.condition,
        batch.targets,
    )
    batch_size = len(batch.sample_ids)
    requests = [
        (sample_index, objective)
        for sample_index in range(batch_size)
        for objective in WEIGHTED_OBJECTIVES
    ]
    gradients: dict[int, dict[str, tuple[Tensor | None, ...]]] = {
        index: {} for index in range(batch_size)
    }
    weighted_values: dict[str, dict[str, float]] = {
        sample_id: {} for sample_id in batch.sample_ids
    }
    for request_index, (sample_index, objective) in enumerate(requests):
        weighted = WEIGHTED_OBJECTIVES[objective] * per_sample_losses[objective][sample_index]
        weighted_values[batch.sample_ids[sample_index]][objective] = float(weighted.detach())
        gradient = torch.autograd.grad(
            weighted,
            parameters,
            retain_graph=request_index + 1 < len(requests),
            allow_unused=True,
        )
        gradients[sample_index][objective] = tuple(gradient)

    result_groups: dict[str, Any] = {}
    objectives = tuple(WEIGHTED_OBJECTIVES)
    ordered_gradients = [
        gradients[sample_index][objective]
        for sample_index in range(batch_size)
        for objective in objectives
    ]
    objective_count = len(objectives)
    for group_name, indices in groups.items():
        gram = _gradient_gram(ordered_gradients, indices)
        sample_rows: dict[str, Any] = {}
        norm_by_sample: dict[int, dict[str, float]] = {}
        for sample_index, sample_id in enumerate(batch.sample_ids):
            offset = sample_index * objective_count
            squared_norms = {
                name: gram[offset + index][offset + index]
                for index, name in enumerate(objectives)
            }
            norms = {name: math.sqrt(max(0.0, value)) for name, value in squared_norms.items()}
            total_squared = sum(
                gram[offset + first][offset + second]
                for first in range(objective_count)
                for second in range(objective_count)
            )
            total_norm = math.sqrt(max(0.0, total_squared))
            norm_by_sample[sample_index] = norms
            alignments = {
                name: _cosine(
                    sum(
                        gram[offset + index][offset + other]
                        for other in range(objective_count)
                    ),
                    norms[name],
                    total_norm,
                )
                for index, name in enumerate(objectives)
            }
            pairwise = {
                f"{first}__{second}": _cosine(
                    gram[offset + first_index][offset + second_index],
                    norms[first],
                    norms[second],
                )
                for (first_index, first), (second_index, second) in combinations(
                    enumerate(objectives), 2
                )
            }
            norm_sum = sum(norms.values())
            sample_rows[sample_id] = {
                "gradient_l2": norms,
                "total_gradient_l2": total_norm,
                "cancellation_ratio": None if norm_sum == 0.0 else total_norm / norm_sum,
                "alignment_to_total": alignments,
                "pairwise_cosine": pairwise,
            }
        cross_sample = {}
        if batch_size == 2:
            for objective_index, objective in enumerate(objectives):
                cross_sample[objective] = _cosine(
                    gram[objective_index][objective_count + objective_index],
                    norm_by_sample[0][objective],
                    norm_by_sample[1][objective],
                )
            total_cross_dot = sum(
                gram[first][objective_count + second]
                for first in range(objective_count)
                for second in range(objective_count)
            )
            cross_sample["total"] = _cosine(
                total_cross_dot,
                sample_rows[batch.sample_ids[0]]["total_gradient_l2"],
                sample_rows[batch.sample_ids[1]]["total_gradient_l2"],
            )
        result_groups[group_name] = {
            "parameter_count": sum(parameters[index].numel() for index in indices),
            "samples": sample_rows,
            "cross_sample_cosine": cross_sample,
        }
    return {
        "weighted_loss_values": weighted_values,
        "parameter_groups": result_groups,
    }
