"""Deterministic endpoint-loss profile over a fixed flow-time grid."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time as wall_time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from apexgen.joint_v2.sampling.base import sample_base_state
from apexgen.joint_v2.data.batch import JointV2Batch, collate_joint_v2_records
from apexgen.joint_v2.runtime.config import validate_joint_v2_config, validate_joint_v2_data_config
from apexgen.joint_v2.contracts.contract import JOINT_V2_CONTRACT_SHA256, require_joint_v2_contract
from apexgen.joint_v2.data.dataset import JointV2Dataset
from apexgen.joint_v2.sampling.flow import conditional_path
from apexgen.joint_v2.geometry.rotations import (
    sidechain_pi_periodic_mask,
    symmetry_aware_chi_distance,
)
from apexgen.joint_v2.runtime.lineage import (
    canonical_sha256,
    joint_v2_data_views_identity,
    joint_v2_dataset_identity,
    sha256_file,
)
from apexgen.joint_v2.training.loss import JointEndpointLossWeights, joint_endpoint_loss
from apexgen.joint_v2.model.network import build_joint_v2_model
from apexgen.joint_v2.runtime.run import load_joint_v2_run_directory
from apexgen.joint_v2.contracts.state import JointFlowState
from apexgen.joint_v2.training.trainer import validated_checkpoint_model_state
from apexgen.joint_v2.evaluation.validation import block_endpoint_metrics_per_sample
from apexgen.shared.training.precision import network_autocast


TIME_PROFILE_SCHEMA_VERSION = "apexgen.joint_v2.sequence_structure_endpoint.time_profile.v1"
BASE_SEED_SCHEMA_VERSION = "apexgen.joint_v2.sequence_structure_endpoint.diagnostic_base_seed.v1"
_CHECKPOINT_SCHEMA_VERSION = "apexgen.joint_v2.sequence_structure_endpoint.checkpoint.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def diagnostic_base_seed(global_seed: int, base_index: int, sample_id: str) -> int:
    """Derive a batching/order-independent 63-bit seed for one sample/base."""

    payload = f"{BASE_SEED_SCHEMA_VERSION}\0{global_seed}\0{base_index}\0{sample_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _slice_condition(batch: JointV2Batch, row: int):
    cls = type(batch.condition)
    return cls(**{name: value[row : row + 1] for name, value in batch.condition.__dict__.items()})


def sample_diagnostic_base_state(
    batch: JointV2Batch,
    *,
    global_seed: int,
    base_index: int,
    translation_sigma_angstrom: float,
) -> tuple[JointFlowState, tuple[int, ...]]:
    """Sample each row independently so the base bank survives rebatching."""

    states, seeds = [], []
    for row, sample_id in enumerate(batch.sample_ids):
        condition = _slice_condition(batch, row)
        seed = diagnostic_base_seed(global_seed, base_index, sample_id)
        generator = torch.Generator(device=condition.residue_mask.device).manual_seed(seed)
        states.append(
            sample_base_state(
                condition,
                translation_sigma_angstrom=translation_sigma_angstrom,
                generator=generator,
            )
        )
        seeds.append(seed)
    return JointFlowState(
        translation=torch.cat([state.translation for state in states]),
        rotation=torch.cat([state.rotation for state in states]),
        sequence_logits=torch.cat([state.sequence_logits for state in states]),
    ), tuple(seeds)


def _weights(config: dict[str, Any]) -> JointEndpointLossWeights:
    loss = config["loss"]
    return JointEndpointLossWeights(
        trajectory_fape=loss["trajectory_fape"]["weight"],
        final_translation=loss["final_translation"]["weight"],
        final_rotation=loss["final_rotation"]["weight"],
        final_backbone=loss["final_backbone_n_ca_c"]["weight"],
        backbone_angle=loss["backbone_angle"]["weight"],
        backbone_angle_norm=loss["backbone_angle_norm"]["weight"],
        sidechain_angle=loss["sidechain_angle"]["weight"],
        sidechain_angle_norm=loss["sidechain_angle_norm"]["weight"],
        sequence_logit=loss["sequence"]["logit_weight"],
        sequence_soft_ce=loss["sequence"]["soft_ce_weight"],
        sequence_total=loss["sequence"]["total_weight"],
    )


@torch.no_grad()
def validate_joint_v2_time_profile(
    model: nn.Module,
    batches: Iterable[JointV2Batch],
    *,
    device: torch.device | str,
    translation_sigma_angstrom: float,
    seed: int,
    times: Iterable[float],
    bases_per_sample: int,
    network_precision: str,
    weights: JointEndpointLossWeights | None = None,
    peptide_clamp_angstrom: float = 10.0,
    cross_clamp_angstrom: float = 30.0,
) -> dict[str, Any]:
    """Report mean endpoint losses for every fixed ``t`` over a fixed base bank."""

    if bases_per_sample <= 0:
        raise ValueError("bases_per_sample must be positive")
    time_grid = tuple(float(value) for value in times)
    if not time_grid or any(not 0.0 <= value <= 1.0 for value in time_grid):
        raise ValueError("time grid must be non-empty and contained in [0, 1]")
    runtime = model.module if hasattr(model, "module") else model
    runtime.eval()
    target_device = torch.device(device)
    names = (
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
        "backbone_angle_circular_mae_degrees",
        "sidechain_angle_circular_mae_degrees",
    )
    totals = {time: {name: 0.0 for name in names} for time in time_grid}
    block_totals: dict[float, dict[str, Tensor] | None] = {time: None for time in time_grid}
    count = {time: 0 for time in time_grid}
    base_seed_rows: list[tuple[str, int, int]] = []
    sample_count = 0
    for cpu_batch in batches:
        batch = cpu_batch.to(target_device, non_blocking=target_device.type == "cuda")
        size = len(batch.sample_ids)
        sample_count += size
        with network_autocast(target_device, network_precision):
            encoding = runtime.encode_complex(batch.condition)
        for base_index in range(bases_per_sample):
            base, row_seeds = sample_diagnostic_base_state(
                batch,
                global_seed=seed,
                base_index=base_index,
                translation_sigma_angstrom=translation_sigma_angstrom,
            )
            base_seed_rows.extend(
                (sample_id, base_index, row_seed)
                for sample_id, row_seed in zip(batch.sample_ids, row_seeds, strict=True)
            )
            for time_value in time_grid:
                time = torch.full((size,), time_value, dtype=torch.float32, device=target_device)
                state = conditional_path(
                    base=base,
                    targets=batch.targets,
                    condition=batch.condition,
                    time=time,
                )
                with network_autocast(target_device, network_precision):
                    prediction = runtime.decode(
                        state,
                        time,
                        batch.condition,
                        encoding,
                        return_intermediates=True,
                    )
                losses = joint_endpoint_loss(
                    prediction, batch.condition, batch.targets, weights=weights
                )
                target_angles = batch.targets.backbone_angles_sin_cos
                angle_mask = batch.targets.backbone_angle_mask
                angle_distance = torch.acos(
                    (prediction.backbone_angles_sin_cos * target_angles).sum(-1).clamp(-1.0, 1.0)
                )
                angle_weight = angle_mask.float()
                angle_per_sample = (
                    (angle_distance * angle_weight).flatten(1).sum(-1)
                    / angle_weight.flatten(1).sum(-1).clamp_min(1.0)
                    * (180.0 / torch.pi)
                )
                losses = {
                    **losses,
                    "backbone_angle_circular_mae_degrees": angle_per_sample.mean(),
                }
                sidechain_mask = batch.targets.sidechain_angle_mask
                sidechain_distance = symmetry_aware_chi_distance(
                    prediction.sidechain_angles_sin_cos,
                    batch.targets.sidechain_angles_sin_cos,
                    sidechain_pi_periodic_mask(
                        batch.targets.endpoint_aatype,
                        sidechain_mask,
                    ),
                )
                sidechain_weight = sidechain_mask.float()
                sidechain_per_sample = (
                    (sidechain_distance * sidechain_weight).flatten(1).sum(-1)
                    / sidechain_weight.flatten(1).sum(-1).clamp_min(1.0)
                    * (180.0 / torch.pi)
                )
                losses["sidechain_angle_circular_mae_degrees"] = (
                    sidechain_per_sample.mean()
                )
                for name in names:
                    totals[time_value][name] += float(losses[name]) * size
                block_metrics = block_endpoint_metrics_per_sample(
                    prediction,
                    batch,
                    peptide_clamp_angstrom=peptide_clamp_angstrom,
                    cross_clamp_angstrom=cross_clamp_angstrom,
                )
                summed = {
                    name: value.sum(0).double().cpu() for name, value in block_metrics.items()
                }
                if block_totals[time_value] is None:
                    block_totals[time_value] = summed
                else:
                    for name, value in summed.items():
                        block_totals[time_value][name] += value
                count[time_value] += size
    if sample_count == 0:
        raise ValueError("validation split is empty")
    rows = [
        {
            "time": time_value,
            **{name: totals[time_value][name] / count[time_value] for name in names},
            "blocks": [
                {
                    "block": block + 1,
                    **{
                        name: float(value[block] / count[time_value])
                        for name, value in block_totals[time_value].items()
                    },
                }
                for block in range(next(iter(block_totals[time_value].values())).shape[0])
            ],
        }
        for time_value in time_grid
    ]
    base_bank_sha256 = canonical_sha256(
        {"schema_version": BASE_SEED_SCHEMA_VERSION, "rows": sorted(base_seed_rows)}
    )
    return {
        "sample_count": sample_count,
        "bases_per_sample": bases_per_sample,
        "base_bank_sha256": base_bank_sha256,
        "rows": rows,
    }


def validate_time_profile_checkpoint(
    checkpoint: dict[str, Any],
    run: dict[str, Any],
    *,
    config_sha256: str,
    data_config_sha256: str,
    data_config: dict[str, Any],
) -> None:
    if checkpoint.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("diagnostic checkpoint schema mismatch")
    require_joint_v2_contract(checkpoint, source="diagnostic checkpoint")
    for name, expected in (
        ("run_id", run["run_id"]),
        ("config_sha256", config_sha256),
        ("dataset_identity_sha256", run["dataset_identity_sha256"]),
        ("data_view_identity_sha256", run["data_view_identity_sha256"]),
    ):
        if checkpoint.get(name) != expected:
            raise ValueError(f"diagnostic checkpoint binding mismatch: {name}")
    extra = checkpoint.get("extra", {})
    if (
        extra.get("data_config_sha256") != data_config_sha256
        or extra.get("resolved_data_config") != data_config
    ):
        raise ValueError("diagnostic checkpoint data configuration mismatch")


def write_diagnostic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def run_joint_v2_time_profile_diagnostic(
    *,
    run_dir: Path,
    checkpoint_name: Path,
    output_path: Path,
    times: Iterable[float],
    bases_per_sample: int,
    batch_size: int | None,
    num_workers: int,
    seed: int | None,
    model_state: str,
    device_name: str,
    exploratory: bool,
) -> dict[str, Any]:
    """Load a provenance-bound exploratory run and write an endpoint time profile."""

    if not exploratory:
        raise RuntimeError("Joint-v2 diagnostics are exploratory; pass --exploratory")
    if num_workers < 0 or model_state not in {"ema", "raw"}:
        raise ValueError("invalid diagnostic runtime options")
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"time-profile output already exists: {output_path}")
    started = wall_time.monotonic()
    time_grid = tuple(float(value) for value in times)
    run_dir = run_dir.resolve()
    run = load_joint_v2_run_directory(run_dir, expected_binding={})
    require_joint_v2_contract(run, source="time-profile run")
    config, data_config = run.get("resolved_config"), run.get("resolved_data_config")
    if not isinstance(config, dict) or not isinstance(data_config, dict):
        raise ValueError("run lacks resolved configs")
    validate_joint_v2_config(config)
    validate_joint_v2_data_config(data_config)
    config_sha256 = canonical_sha256(config)
    data_config_sha256 = canonical_sha256(data_config)
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
    train_dataset = JointV2Dataset(pocket_root, target_root, data_config["train_split"])
    validation_dataset = JointV2Dataset(pocket_root, target_root, data_config["validation_split"])
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
            raise ValueError("current dataset or split view differs from the run binding")
        training = config["training"]
        effective_batch_size = training["batch_size"] if batch_size is None else batch_size
        device = _device(device_name)
        loader = DataLoader(
            validation_dataset,
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
        effective_seed = training["seed"] + 700_000 if seed is None else seed
        profile = validate_joint_v2_time_profile(
            model,
            loader,
            device=device,
            translation_sigma_angstrom=config["flow"]["base_translation_sigma_angstrom"],
            seed=effective_seed,
            times=time_grid,
            bases_per_sample=bases_per_sample,
            network_precision=config["precision"]["network"],
            weights=_weights(config),
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
    if sha256_file(checkpoint) != checkpoint_sha256:
        raise RuntimeError("checkpoint changed during diagnostic evaluation")
    payload = {
        "schema_version": TIME_PROFILE_SCHEMA_VERSION,
        "joint_v2_contract_sha256": JOINT_V2_CONTRACT_SHA256,
        "formal": False,
        "run_id": run["run_id"],
        "run_dir": str(run_dir),
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": checkpoint_sha256,
        "optimizer_step": int(checkpoint_payload["optimizer_step"]),
        "model_state_view": model_state,
        "config_sha256": config_sha256,
        "data_config_sha256": data_config_sha256,
        "dataset_identity_sha256": dataset_identity["identity_sha256"],
        "data_view_identity_sha256": views["identity_sha256"],
        "validation_split": data_config["validation_split"],
        "seed": effective_seed,
        "time_grid": list(time_grid),
        "effective_batch_size": effective_batch_size,
        "num_workers": num_workers,
        "created_at_utc": _utc_now(),
        "duration_seconds": wall_time.monotonic() - started,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "network_precision": config["precision"]["network"],
        },
        **profile,
    }
    write_diagnostic_json(output_path, payload)
    return payload
