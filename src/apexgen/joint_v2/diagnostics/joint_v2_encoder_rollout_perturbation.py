"""Closed-loop rollout diagnostics under role-aligned encoder shuffling."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from apexgen.joint_v2.diagnostics.joint_v2_rollout_profile import integrate_capped_time_solver
from apexgen.joint_v2.diagnostics.joint_v2_time_profile import sample_diagnostic_base_state
from apexgen.joint_v2.data.batch import JointV2Batch
from apexgen.joint_v2.contracts.contract import UnifiedComplexCondition, UnifiedComplexEncoding
from apexgen.joint_v2.geometry.rotations import so3_log
from apexgen.joint_v2.contracts.state import JointFlowState
from apexgen.joint_v2.evaluation.validation import _rollout_rows
from apexgen.shared.training.precision import network_autocast


ARM_NAMES = ("baseline", "shuffle_single", "shuffle_pair", "shuffle_single_pair")
FINAL_METRICS = (
    "translation_rmse_angstrom",
    "rotation_mean_degrees",
    "backbone_n_ca_c_rmsd_angstrom",
    "peptide_kabsch_rmsd_angstrom",
    "sequence_accuracy",
    "sequence_hard_nll",
)
TRACE_METRICS = (
    "translation_rmse_angstrom",
    "rotation_mean_degrees",
    "sequence_accuracy",
    "sequence_hard_nll",
)


def paired_donors(condition: UnifiedComplexCondition) -> tuple[int, ...]:
    """Pair similar-length samples into a deterministic derangement."""

    batch = condition.layout[0]
    if batch < 2 or batch % 2:
        raise ValueError("encoder shuffling requires a positive even batch")
    ordered = sorted(
        range(batch),
        key=lambda index: (
            int(condition.peptide_mask[index].sum()),
            int(condition.pocket_mask[index].sum()),
            index,
        ),
    )
    donors = [-1] * batch
    for offset in range(0, batch, 2):
        left, right = ordered[offset : offset + 2]
        donors[left], donors[right] = right, left
    return tuple(donors)


def _role_map(source_mask: Tensor, target_count: int) -> Tensor:
    source = source_mask.nonzero(as_tuple=False).flatten()
    if source.numel() == 0 or target_count <= 0:
        raise ValueError("role-aligned encoding shuffle requires non-empty roles")
    positions = torch.linspace(
        0, source.numel() - 1, target_count, device=source.device
    ).round().long()
    return source[positions]


def transplant_encoding(
    encoding: UnifiedComplexEncoding,
    condition: UnifiedComplexCondition,
    donors: tuple[int, ...],
    *,
    shuffle_single: bool,
    shuffle_pair: bool,
) -> UnifiedComplexEncoding:
    """Transplant encoder tensors while leaving direct condition inputs unchanged."""

    if not shuffle_single and not shuffle_pair:
        raise ValueError("at least one encoder tensor must be shuffled")
    if len(donors) != condition.layout[0]:
        raise ValueError("donor count differs from batch size")
    single, pair = encoding.single.clone(), encoding.pair.clone()
    for target, donor in enumerate(donors):
        if donor == target or donor < 0 or donor >= len(donors):
            raise ValueError("donors must form an in-batch derangement")
        target_pocket = condition.pocket_mask[target].nonzero(as_tuple=False).flatten()
        target_peptide = condition.peptide_mask[target].nonzero(as_tuple=False).flatten()
        target_nodes = torch.cat((target_pocket, target_peptide))
        donor_nodes = torch.cat(
            (
                _role_map(condition.pocket_mask[donor], target_pocket.numel()),
                _role_map(condition.peptide_mask[donor], target_peptide.numel()),
            )
        )
        if shuffle_single:
            single[target, target_nodes] = encoding.single[donor, donor_nodes]
        if shuffle_pair:
            donor_pair = encoding.pair[donor].index_select(0, donor_nodes).index_select(
                1, donor_nodes
            )
            pair[target, target_nodes[:, None], target_nodes[None, :]] = donor_pair
    return UnifiedComplexEncoding(single=single, pair=pair)


def encoder_perturbation_arms(
    encoding: UnifiedComplexEncoding,
    condition: UnifiedComplexCondition,
) -> tuple[dict[str, UnifiedComplexEncoding], tuple[int, ...]]:
    donors = paired_donors(condition)
    return {
        "baseline": encoding,
        "shuffle_single": transplant_encoding(
            encoding, condition, donors, shuffle_single=True, shuffle_pair=False
        ),
        "shuffle_pair": transplant_encoding(
            encoding, condition, donors, shuffle_single=False, shuffle_pair=True
        ),
        "shuffle_single_pair": transplant_encoding(
            encoding, condition, donors, shuffle_single=True, shuffle_pair=True
        ),
    }, donors


def _trace_metrics(state: JointFlowState, batch: JointV2Batch) -> dict[str, Tensor]:
    peptide = batch.condition.peptide_mask
    weight = peptide.float()
    count = weight.sum(-1).clamp_min(1.0)
    translation = (
        state.translation.sub(batch.targets.endpoint_translation).square().sum(-1) * weight
    ).sum(-1).div(count).sqrt()
    rotation = so3_log(
        batch.targets.endpoint_rotation.transpose(-1, -2) @ state.rotation
    ).norm(dim=-1)
    rotation = (rotation * weight).sum(-1) / count * (180.0 / math.pi)
    target_aatype = batch.targets.endpoint_aatype
    sequence_accuracy = (
        (state.sequence_logits.argmax(-1) == target_aatype).float() * weight
    ).sum(-1) / count
    safe_target = torch.where(peptide, target_aatype, 0)
    sequence_nll = -state.sequence_logits.log_softmax(-1).gather(
        -1, safe_target[..., None]
    )[..., 0]
    sequence_nll = (sequence_nll * weight).sum(-1) / count
    return {
        "translation_rmse_angstrom": translation,
        "rotation_mean_degrees": rotation,
        "sequence_accuracy": sequence_accuracy,
        "sequence_hard_nll": sequence_nll,
    }


def _target_means(
    rows: list[dict[str, Any]], sample_ids: tuple[str, ...], metrics: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = {
        sample_id: {metric: [] for metric in metrics} for sample_id in sample_ids
    }
    for row in rows:
        sample_id = row["sample_id"]
        for metric in metrics:
            grouped[sample_id][metric].append(float(row[metric]))
    return {
        sample_id: {
            metric: float(np.mean(values))
            for metric, values in metric_rows.items()
        }
        for sample_id, metric_rows in grouped.items()
    }


def _bootstrap(values: list[float], seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.RandomState(seed)
    means = generator.choice(array, size=(10_000, len(array)), replace=True).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "target_count": len(values),
        "bootstrap_replicates": 10_000,
    }


def _summaries(
    target_means: dict[str, dict[str, dict[str, float]]],
    sample_ids: tuple[str, ...],
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summaries, deltas = {}, {}
    for arm_index, arm in enumerate(ARM_NAMES):
        summaries[arm] = {}
        for metric_index, metric in enumerate(FINAL_METRICS):
            values = [target_means[arm][sample_id][metric] for sample_id in sample_ids]
            summaries[arm][metric] = _bootstrap(
                values, seed + 10_000 * arm_index + 100 * metric_index
            )
        if arm == "baseline":
            continue
        deltas[arm] = {}
        for metric_index, metric in enumerate(FINAL_METRICS):
            values = [
                target_means[arm][sample_id][metric]
                - target_means["baseline"][sample_id][metric]
                for sample_id in sample_ids
            ]
            deltas[arm][metric] = _bootstrap(
                values, seed + 1_000_000 + 10_000 * arm_index + 100 * metric_index
            )
    return summaries, deltas


@torch.no_grad()
def evaluate_encoder_rollout_perturbation(
    model: nn.Module,
    batch: JointV2Batch,
    *,
    device: torch.device | str,
    translation_sigma_angstrom: float,
    seed: int,
    bases_per_sample: int,
    network_precision: str,
    clash_distance_angstrom: float,
) -> dict[str, Any]:
    """Compare matched closed-loop rollouts under correct and shuffled encodings."""

    if bases_per_sample <= 0:
        raise ValueError("bases_per_sample must be positive")
    runtime = model.module if hasattr(model, "module") else model
    runtime.eval()
    target_device = torch.device(device)
    batch = batch.to(target_device, non_blocking=target_device.type == "cuda")
    with network_autocast(target_device, network_precision):
        encoding = runtime.encode_complex(batch.condition)
    arms, donors = encoder_perturbation_arms(encoding, batch.condition)
    final_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARM_NAMES}
    trace_rows: dict[str, dict[int, list[dict[str, Any]]]] = {
        arm: defaultdict(list) for arm in ARM_NAMES
    }
    base_seed_rows = []
    for base_index in range(bases_per_sample):
        initial, row_seeds = sample_diagnostic_base_state(
            batch,
            global_seed=seed,
            base_index=base_index,
            translation_sigma_angstrom=translation_sigma_angstrom,
        )
        base_seed_rows.extend(
            {"sample_id": sample_id, "base_index": base_index, "seed": row_seed}
            for sample_id, row_seed in zip(batch.sample_ids, row_seeds, strict=True)
        )
        for arm in ARM_NAMES:
            last_prediction = None

            def endpoint(state: JointFlowState, time: Tensor):
                nonlocal last_prediction
                with network_autocast(target_device, network_precision):
                    last_prediction = runtime.decode(
                        state,
                        time,
                        batch.condition,
                        arms[arm],
                        return_intermediates=False,
                    )
                return last_prediction

            def trace(index: int, query: float, next_value: float, state: JointFlowState) -> None:
                metrics = _trace_metrics(state, batch)
                for sample_index, sample_id in enumerate(batch.sample_ids):
                    trace_rows[arm][index].append(
                        {
                            "sample_id": sample_id,
                            "base_index": base_index,
                            "query_time": query,
                            "next_time": next_value,
                            **{
                                name: float(values[sample_index])
                                for name, values in metrics.items()
                            },
                        }
                    )

            generated = integrate_capped_time_solver(
                initial, batch.condition, endpoint, trace_callback=trace
            )
            if last_prediction is None:
                raise RuntimeError("rollout produced no endpoint prediction")
            rows, *_ = _rollout_rows(
                generated,
                initial,
                batch,
                clash_distance_angstrom=clash_distance_angstrom,
                angle_sin_cos=last_prediction.backbone_angles_sin_cos,
                sidechain_angle_sin_cos=last_prediction.sidechain_angles_sin_cos,
            )
            final_rows[arm].extend(
                {
                    "sample_id": sample_id,
                    "base_index": base_index,
                    **{metric: row[metric] for metric in FINAL_METRICS},
                }
                for sample_id, row in zip(batch.sample_ids, rows, strict=True)
            )

    sample_ids = tuple(batch.sample_ids)
    final_target_means = {
        arm: _target_means(rows, sample_ids, FINAL_METRICS)
        for arm, rows in final_rows.items()
    }
    final_summary, final_delta = _summaries(
        final_target_means, sample_ids, seed=seed + 200_000_000
    )
    trace = []
    for step in range(20):
        step_target_means = {
            arm: _target_means(trace_rows[arm][step], sample_ids, TRACE_METRICS)
            for arm in ARM_NAMES
        }
        step_summary, step_delta = {}, {}
        for arm_index, arm in enumerate(ARM_NAMES):
            step_summary[arm] = {}
            for metric_index, metric in enumerate(TRACE_METRICS):
                values = [step_target_means[arm][sample_id][metric] for sample_id in sample_ids]
                step_summary[arm][metric] = float(np.mean(values))
            if arm == "baseline":
                continue
            step_delta[arm] = {
                metric: float(
                    np.mean(
                        [
                            step_target_means[arm][sample_id][metric]
                            - step_target_means["baseline"][sample_id][metric]
                            for sample_id in sample_ids
                        ]
                    )
                )
                for metric in TRACE_METRICS
            }
        trace.append(
            {
                "step": step + 1,
                "query_time": step / 20,
                "next_time": (step + 1) / 20,
                "target_macro_mean": step_summary,
                "paired_delta_perturbed_minus_baseline": step_delta,
            }
        )
    return {
        "arms": list(ARM_NAMES),
        "sample_ids": list(sample_ids),
        "sample_count": len(sample_ids),
        "bases_per_sample": bases_per_sample,
        "candidate_count_per_arm": len(sample_ids) * bases_per_sample,
        "base_seed_rows": base_seed_rows,
        "donor_rows": [
            {
                "target_sample_id": sample_ids[target],
                "donor_sample_id": sample_ids[donor],
            }
            for target, donor in enumerate(donors)
        ],
        "final_target_means": final_target_means,
        "final_target_macro_summary": final_summary,
        "final_paired_delta_perturbed_minus_baseline": final_delta,
        "trace": trace,
    }
