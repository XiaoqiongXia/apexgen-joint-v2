"""Deterministic validation-loss checkpoint selection for Joint-v2."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from apexgen.joint_v2.contracts.contract import JOINT_V2_CONTRACT_SHA256, require_joint_v2_contract
from apexgen.joint_v2.runtime.lineage import sha256_file


def select_validation_checkpoint(
    metrics_path: str | Path,
    checkpoint_directory: str | Path,
    *,
    relative_tolerance: float = 0.01,
    expected_train_split: str | None = None,
    expected_validation_split: str | None = None,
    expected_run_id: str | None = None,
    expected_config_sha256: str | None = None,
    expected_dataset_identity_sha256: str | None = None,
    expected_data_view_identity_sha256: str | None = None,
) -> dict[str, Any]:
    """Choose the earliest checkpoint within tolerance of minimum validation loss."""

    if not 0.0 <= relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must lie in [0, 1)")
    metrics_path = Path(metrics_path)
    checkpoint_directory = Path(checkpoint_directory)
    candidates: list[dict[str, Any]] = []
    observed_steps: set[int] = set()
    with metrics_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") != "validation":
                continue
            step, loss = row.get("optimizer_step"), row.get("loss_total")
            if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
                raise ValueError(f"invalid validation step at metrics line {line_number}")
            if isinstance(loss, bool) or not isinstance(loss, (int, float)):
                raise ValueError(f"invalid validation loss at metrics line {line_number}")
            loss = float(loss)
            if not math.isfinite(loss) or loss < 0.0:
                raise ValueError(f"non-finite or negative validation loss at step {step}")
            if step in observed_steps:
                raise ValueError(f"duplicate validation metrics at step {step}")
            observed_steps.add(step)
            checkpoint = checkpoint_directory / f"checkpoint_{step:08d}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"validation checkpoint is absent at step {step}")
            candidates.append({"optimizer_step": step, "loss_total": loss, "path": checkpoint})
    if not candidates:
        raise ValueError("no validation row has a matching periodic checkpoint")
    minimum = min(candidate["loss_total"] for candidate in candidates)
    threshold = minimum * (1.0 + relative_tolerance)
    selected = min(
        (candidate for candidate in candidates if candidate["loss_total"] <= threshold),
        key=lambda candidate: candidate["optimizer_step"],
    )
    checkpoint = selected["path"].resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        "apexgen.joint_v2.sequence_structure_endpoint.checkpoint.v1"
    ):
        raise ValueError("selected checkpoint schema mismatch")
    require_joint_v2_contract(payload, source="selected checkpoint")
    if payload.get("optimizer_step") != selected["optimizer_step"]:
        raise ValueError("selected checkpoint step differs from validation metrics")
    run_id = payload.get("run_id")
    config_sha256 = payload.get("config_sha256")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("selected checkpoint has no run identity")
    if not isinstance(config_sha256, str) or len(config_sha256) != 64:
        raise ValueError("selected checkpoint has no config identity")
    if expected_run_id is not None and run_id != expected_run_id:
        raise ValueError("selected checkpoint run identity differs from expectation")
    if expected_config_sha256 is not None and config_sha256 != expected_config_sha256:
        raise ValueError("selected checkpoint config identity differs from expectation")
    extra = payload.get("extra")
    data_config = extra.get("resolved_data_config") if isinstance(extra, dict) else None
    if expected_train_split is not None and (
        not isinstance(data_config, dict) or data_config.get("train_split") != expected_train_split
    ):
        raise ValueError("selected checkpoint training split differs from expectation")
    if expected_validation_split is not None and (
        not isinstance(data_config, dict)
        or data_config.get("validation_split") != expected_validation_split
    ):
        raise ValueError("selected checkpoint validation split differs from expectation")
    dataset_identity = payload.get("dataset_identity_sha256")
    data_view_identity = payload.get("data_view_identity_sha256")
    if not isinstance(dataset_identity, str) or len(dataset_identity) != 64:
        raise ValueError("selected checkpoint has no dataset identity")
    if not isinstance(data_view_identity, str) or len(data_view_identity) != 64:
        raise ValueError("selected checkpoint has no data-view identity")
    if (
        expected_dataset_identity_sha256 is not None
        and dataset_identity != expected_dataset_identity_sha256
    ):
        raise ValueError("selected checkpoint dataset identity differs from expectation")
    if (
        expected_data_view_identity_sha256 is not None
        and data_view_identity != expected_data_view_identity_sha256
    ):
        raise ValueError("selected checkpoint data-view identity differs from expectation")
    return {
        "schema_version": "apexgen.diagnostics.joint_v2.checkpoint_selection.v1",
        "selection_metric": "deterministic_full_validation_loss_total_ema",
        "relative_tolerance": relative_tolerance,
        "minimum_loss_total": minimum,
        "eligibility_threshold": threshold,
        "selected_optimizer_step": selected["optimizer_step"],
        "selected_loss_total": selected["loss_total"],
        "selected_checkpoint": str(checkpoint),
        "selected_checkpoint_sha256": sha256_file(checkpoint),
        "source_contract_sha256": JOINT_V2_CONTRACT_SHA256,
        "source_run_id": run_id,
        "source_config_sha256": config_sha256,
        "source_dataset_identity_sha256": dataset_identity,
        "source_data_view_identity_sha256": data_view_identity,
        "source_train_split": None if data_config is None else data_config.get("train_split"),
        "source_validation_split": (
            None if data_config is None else data_config.get("validation_split")
        ),
        "candidate_count": len(candidates),
    }
