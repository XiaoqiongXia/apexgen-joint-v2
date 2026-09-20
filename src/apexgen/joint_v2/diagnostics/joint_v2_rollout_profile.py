"""Full-validation closed-loop profile for the locked endpoint solver."""

from __future__ import annotations

import math
import time as wall_time
from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from apexgen.joint_v2.diagnostics.joint_v2_time_profile import (
    _device,
    sample_diagnostic_base_state,
    validate_time_profile_checkpoint,
    write_diagnostic_json,
)
from apexgen.joint_v2.data.batch import JointV2Batch, collate_joint_v2_records
from apexgen.joint_v2.runtime.config import validate_joint_v2_config, validate_joint_v2_data_config
from apexgen.joint_v2.contracts.contract import JOINT_V2_CONTRACT_SHA256, require_joint_v2_contract
from apexgen.joint_v2.data.dataset import JointV2Dataset, JointV2DatasetView
from apexgen.joint_v2.sampling.flow import (
    INTEGRATION_STEP_SIZE,
    INTEGRATION_STEPS,
    endpoint_step,
)
from apexgen.joint_v2.geometry.rotations import so3_log
from apexgen.joint_v2.runtime.lineage import (
    canonical_sha256,
    joint_v2_data_views_identity,
    joint_v2_dataset_identity,
    sha256_file,
)
from apexgen.joint_v2.model.network import build_joint_v2_model
from apexgen.joint_v2.runtime.run import load_joint_v2_run_directory
from apexgen.joint_v2.contracts.state import JointFlowState
from apexgen.joint_v2.training.trainer import validated_checkpoint_model_state
from apexgen.joint_v2.evaluation.validation import _rollout_rows
from apexgen.joint_v2.evaluation.validation import block_endpoint_metrics_per_sample
from apexgen.shared.training.precision import network_autocast


ROLLOUT_PROFILE_SCHEMA_VERSION = "apexgen.joint_v2.sequence_structure_endpoint.rollout_profile.v1"
SOLVER_NAMES = ("endpoint_20",)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def solver_specification(name: str) -> tuple[str, int]:
    if name not in SOLVER_NAMES:
        raise ValueError(f"unsupported endpoint solver: {name}")
    return "endpoint_fractional_geodesic", INTEGRATION_STEPS


def capped_time_schedule(steps: int = INTEGRATION_STEPS) -> tuple[tuple[float, float], ...]:
    if steps != INTEGRATION_STEPS:
        raise ValueError("frame-endpoint v1 fixes the integration step count to 20")
    return tuple(
        (index * INTEGRATION_STEP_SIZE, (index + 1) * INTEGRATION_STEP_SIZE)
        for index in range(steps)
    )


def integrate_capped_time_solver(
    initial: JointFlowState,
    condition,
    endpoint_fn: Callable[[JointFlowState, Tensor], Any],
    *,
    solver: str = "endpoint_20",
    trace_callback: Callable[[int, float, float, JointFlowState], None] | None = None,
) -> JointFlowState:
    """Run the single supported solver and optionally observe every new state."""

    solver_specification(solver)
    state = initial
    for index, (query, next_value) in enumerate(capped_time_schedule()):
        time = torch.full(
            (initial.layout[0],), query, dtype=torch.float32, device=initial.translation.device
        )
        next_time = torch.full_like(time, next_value)
        state = endpoint_step(
            state,
            endpoint_fn(state, time),
            condition=condition,
            time=time,
            next_time=next_time,
        )
        if trace_callback is not None:
            trace_callback(index, query, next_value, state)
    return state


def _distribution(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p95": float(torch.quantile(tensor, 0.95)),
        "max": float(tensor.max()),
    }


@torch.no_grad()
def validate_joint_v2_rollout_profile(
    model: nn.Module,
    batches: Iterable[JointV2Batch],
    *,
    device: torch.device | str,
    solver: str,
    translation_sigma_angstrom: float,
    seed: int,
    bases_per_sample: int,
    network_precision: str,
    clash_distance_angstrom: float,
    peptide_clamp_angstrom: float,
    cross_clamp_angstrom: float,
) -> dict[str, Any]:
    """Roll out every validation sample/base and aggregate final and step metrics."""

    solver_specification(solver)
    if bases_per_sample <= 0:
        raise ValueError("bases_per_sample must be positive")
    runtime = model.module if hasattr(model, "module") else model
    runtime.eval()
    target_device = torch.device(device)
    final_rows: list[dict[str, float]] = []
    trace_rows: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    block_trace_rows: dict[int, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    seed_rows: list[tuple[str, int, int]] = []
    sample_count = 0
    for cpu_batch in batches:
        batch = cpu_batch.to(target_device, non_blocking=target_device.type == "cuda")
        sample_count += len(batch.sample_ids)
        with network_autocast(target_device, network_precision):
            encoding = runtime.encode_complex(batch.condition)
        for base_index in range(bases_per_sample):
            initial, row_seeds = sample_diagnostic_base_state(
                batch,
                global_seed=seed,
                base_index=base_index,
                translation_sigma_angstrom=translation_sigma_angstrom,
            )
            seed_rows.extend(
                (sample_id, base_index, row_seed)
                for sample_id, row_seed in zip(batch.sample_ids, row_seeds, strict=True)
            )

            last_prediction = None
            current_block_metrics = None

            def endpoint(state: JointFlowState, time: Tensor):
                nonlocal current_block_metrics, last_prediction
                with network_autocast(target_device, network_precision):
                    last_prediction = runtime.decode(
                        state,
                        time,
                        batch.condition,
                        encoding,
                        return_intermediates=True,
                    )
                current_block_metrics = block_endpoint_metrics_per_sample(
                    last_prediction,
                    batch,
                    peptide_clamp_angstrom=peptide_clamp_angstrom,
                    cross_clamp_angstrom=cross_clamp_angstrom,
                )
                return last_prediction

            def trace(index: int, query: float, next_value: float, state: JointFlowState) -> None:
                peptide = batch.condition.peptide_mask
                weight = peptide.float()
                count = weight.sum(-1).clamp_min(1.0)
                translation_error = (
                    state.translation.sub(batch.targets.endpoint_translation).square().sum(-1)
                )
                translation = ((translation_error * weight).sum(-1) / count).sqrt()
                rotation_error = so3_log(
                    batch.targets.endpoint_rotation.transpose(-1, -2) @ state.rotation
                ).norm(dim=-1)
                rotation = (rotation_error * weight).sum(-1) / count
                trace_rows[index]["query_time"].append(query)
                trace_rows[index]["next_time"].append(next_value)
                trace_rows[index]["translation_rmse_angstrom"].extend(
                    float(value) for value in translation
                )
                trace_rows[index]["rotation_mean_degrees"].extend(
                    float(value * 180.0 / math.pi) for value in rotation
                )
                if current_block_metrics is None:
                    raise RuntimeError("rollout block metrics were not produced")
                for name, values in current_block_metrics.items():
                    for block in range(values.shape[1]):
                        block_trace_rows[index][block][name].extend(
                            float(value) for value in values[:, block]
                        )

            generated = integrate_capped_time_solver(
                initial,
                batch.condition,
                endpoint,
                solver=solver,
                trace_callback=trace,
            )
            rows, *_ = _rollout_rows(
                generated,
                initial,
                batch,
                clash_distance_angstrom=clash_distance_angstrom,
                angle_sin_cos=(
                    None if last_prediction is None else last_prediction.backbone_angles_sin_cos
                ),
                sidechain_angle_sin_cos=(
                    None if last_prediction is None else last_prediction.sidechain_angles_sin_cos
                ),
            )
            final_rows.extend(rows)
    if sample_count == 0:
        raise ValueError("rollout dataset is empty")
    statistics = {}
    for name in sorted({name for row in final_rows for name in row}):
        defined = [row[name] for row in final_rows if name in row]
        statistics[name] = {
            **_distribution(defined),
            "defined_candidate_count": len(defined),
        }
    trace = [
        {
            "step": index + 1,
            **{name: sum(values) / len(values) for name, values in metrics.items()},
            "refinement_blocks": [
                {
                    "block": block + 1,
                    **{name: sum(values) / len(values) for name, values in block_metrics.items()},
                }
                for block, block_metrics in sorted(block_trace_rows[index].items())
            ],
        }
        for index, metrics in sorted(trace_rows.items())
    ]
    base_bank_sha256 = canonical_sha256(
        {"schema_version": "sequence_structure_endpoint_base_bank.v1", "rows": sorted(seed_rows)}
    )
    return {
        "solver": solver,
        "sample_count": sample_count,
        "bases_per_sample": bases_per_sample,
        "candidate_count": len(final_rows),
        "base_bank_sha256": base_bank_sha256,
        "statistics": statistics,
        "trace": trace,
    }


def _manifest_dataset_view(
    source: JointV2Dataset,
    *,
    selection: dict[str, Any] | None,
    name: str,
) -> JointV2DatasetView:
    """Rebuild an ordered train/validation view and verify its recorded identities."""

    if selection is None:
        return JointV2DatasetView(source, tuple(range(len(source))))
    row = selection.get(name)
    if not isinstance(row, dict):
        raise ValueError(f"run data selection has no {name} view")
    indices = row.get("source_indices")
    sample_ids = row.get("sample_ids")
    if (
        not isinstance(indices, list)
        or not indices
        or any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
        or not isinstance(sample_ids, list)
        or any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids)
        or row.get("source_count") != len(source)
        or row.get("selected_count") != len(indices)
        or len(sample_ids) != len(indices)
    ):
        raise ValueError(f"run data selection {name} view is invalid")
    view = JointV2DatasetView(source, tuple(indices))
    observed_ids = [str(record.get("sample_id")) for record in view.rows]
    if observed_ids != sample_ids:
        raise ValueError(f"run data selection {name} sample IDs differ from the dataset")
    return view


def run_joint_v2_rollout_profile_diagnostic(
    *,
    run_dir: Path,
    checkpoint_name: Path,
    output_path: Path,
    solver: str,
    bases_per_sample: int,
    batch_size: int | None,
    num_workers: int,
    seed: int,
    model_state: str,
    device_name: str,
    exploratory: bool,
    view: str = "validation",
) -> dict[str, Any]:
    if not exploratory:
        raise RuntimeError("Joint-v2 rollout diagnostics are exploratory; pass --exploratory")
    solver_specification(solver)
    if num_workers < 0 or model_state not in {"ema", "raw"} or view not in {
        "train",
        "validation",
    }:
        raise ValueError("invalid rollout diagnostic options")
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"rollout output already exists: {output_path}")
    started = wall_time.monotonic()
    run_dir = run_dir.resolve()
    run = load_joint_v2_run_directory(run_dir, expected_binding={})
    require_joint_v2_contract(run, source="rollout-profile run")
    config, data_config = run.get("resolved_config"), run.get("resolved_data_config")
    if not isinstance(config, dict) or not isinstance(data_config, dict):
        raise ValueError("run lacks resolved configs")
    validate_joint_v2_config(config)
    validate_joint_v2_data_config(data_config)
    config_sha256, data_config_sha256 = canonical_sha256(config), canonical_sha256(data_config)
    checkpoint = (
        checkpoint_name if checkpoint_name.is_absolute() else run_dir / checkpoint_name
    ).resolve()
    if checkpoint.parent != run_dir or not checkpoint.is_file():
        raise ValueError("checkpoint must be a direct child of the run directory")
    checkpoint_sha256 = sha256_file(checkpoint)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_time_profile_checkpoint(
        checkpoint_payload,
        run,
        config_sha256=config_sha256,
        data_config_sha256=data_config_sha256,
        data_config=data_config,
    )
    pocket_root, target_root = Path(data_config["pocket_root"]), Path(data_config["target_root"])
    dataset_identity = joint_v2_dataset_identity(pocket_root, target_root)
    train_source = JointV2Dataset(pocket_root, target_root, data_config["train_split"])
    validation_source = JointV2Dataset(
        pocket_root, target_root, data_config["validation_split"]
    )
    selection = run.get("data_selection")
    if selection is not None:
        if (
            not isinstance(selection, dict)
            or run.get("data_selection_sha256") != canonical_sha256(selection)
        ):
            raise ValueError("run data selection digest is invalid")
    train_dataset = _manifest_dataset_view(
        train_source,
        selection=selection,
        name="train",
    )
    validation_dataset = _manifest_dataset_view(
        validation_source,
        selection=selection,
        name="validation",
    )
    try:
        views = joint_v2_data_views_identity(
            train_dataset.rows,
            validation_dataset.rows,
            train_split=data_config["train_split"],
            validation_split=data_config["validation_split"],
        )
        if (
            dataset_identity["identity_sha256"] != run["dataset_identity_sha256"]
            or views["identity_sha256"] != run["data_view_identity_sha256"]
        ):
            raise ValueError("current dataset or data view differs from run binding")
        training = config["training"]
        effective_batch_size = training["batch_size"] if batch_size is None else batch_size
        device = _device(device_name)
        evaluation_dataset = train_dataset if view == "train" else validation_dataset
        loader = DataLoader(
            evaluation_dataset,
            batch_size=effective_batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=device.type == "cuda",
            collate_fn=collate_joint_v2_records,
        )
        model = build_joint_v2_model(config).to(device)
        model.load_state_dict(
            validated_checkpoint_model_state(
                checkpoint_payload, model, model_state_view=model_state
            )
        )
        result = validate_joint_v2_rollout_profile(
            model,
            loader,
            device=device,
            solver=solver,
            translation_sigma_angstrom=config["flow"]["base_translation_sigma_angstrom"],
            seed=seed,
            bases_per_sample=bases_per_sample,
            network_precision=config["precision"]["network"],
            clash_distance_angstrom=training["rollout_clash_distance_angstrom"],
            peptide_clamp_angstrom=config["loss"]["trajectory_fape"][
                "validation_peptide_clamp_angstrom"
            ],
            cross_clamp_angstrom=config["loss"]["trajectory_fape"][
                "validation_cross_clamp_angstrom"
            ],
        )
    finally:
        train_dataset.close()
        validation_dataset.close()
    payload = {
        "schema_version": ROLLOUT_PROFILE_SCHEMA_VERSION,
        "joint_v2_contract_sha256": JOINT_V2_CONTRACT_SHA256,
        "formal": False,
        "created_at_utc": _utc_now(),
        "duration_seconds": wall_time.monotonic() - started,
        "run_dir": str(run_dir),
        "run_id": run["run_id"],
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": checkpoint_sha256,
        "optimizer_step": int(checkpoint_payload["optimizer_step"]),
        "model_state_view": model_state,
        "config_sha256": config_sha256,
        "data_config_sha256": data_config_sha256,
        "dataset_identity_sha256": dataset_identity["identity_sha256"],
        "data_view_identity_sha256": views["identity_sha256"],
        "validation_split": data_config["validation_split"],
        "evaluation_view": view,
        "evaluation_split": (
            data_config["train_split"] if view == "train" else data_config["validation_split"]
        ),
        "effective_batch_size": effective_batch_size,
        "num_workers": num_workers,
        "seed": seed,
        **result,
    }
    write_diagnostic_json(output_path, payload)
    return payload
