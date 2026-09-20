"""Fail-closed exploratory run-directory ownership for Joint-v2."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


RUN_SCHEMA_VERSION = "apexgen.joint_v2.sequence_structure_endpoint.exploratory_run.v1"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def create_joint_v2_run_directory(output: str | Path, manifest: dict[str, Any]) -> None:
    """Create a new owned directory; never mix with pre-existing artifacts."""

    root = Path(output)
    if root.exists():
        if not root.is_dir():
            raise ValueError(f"Joint-v2 output is not a directory: {root}")
        if any(root.iterdir()):
            raise ValueError(f"Joint-v2 output directory is not empty: {root}")
    else:
        root.mkdir(parents=True)
    if manifest.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError("Joint-v2 run manifest schema mismatch")
    _atomic_json(root / "run.json", manifest)


def load_joint_v2_run_directory(
    output: str | Path, *, expected_binding: dict[str, Any]
) -> dict[str, Any]:
    """Load an existing owned run and verify immutable runtime bindings."""

    path = Path(output) / "run.json"
    if not path.is_file():
        raise ValueError(f"resume output has no Joint-v2 run.json: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError("Joint-v2 run manifest schema mismatch")
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Joint-v2 run manifest has no run_id")
    for name, expected in expected_binding.items():
        if payload.get(name) != expected:
            raise ValueError(f"Joint-v2 resume run binding mismatch: {name}")
    return payload


def last_logged_train_step(path: str | Path) -> int:
    """Return the maximum complete train step in metrics.jsonl, or zero."""

    metrics_path = Path(path)
    if not metrics_path.exists():
        return 0
    maximum = 0
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid metrics JSON at line {line_number}: {metrics_path}"
                ) from error
            if row.get("kind") == "train":
                step = row.get("optimizer_step")
                if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
                    raise ValueError("metrics train row has an invalid optimizer_step")
                maximum = max(maximum, step)
    return maximum
