"""Fail-closed dataset and file identities for Joint-v2 runtimes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def joint_v2_dataset_view_identity(rows: list[dict[str, Any]], *, split: str) -> dict[str, Any]:
    """Bind an ordered runtime split to the exact records consumed by a loader."""

    if not isinstance(split, str) or not split:
        raise ValueError("dataset view split must be a non-empty string")
    ordered_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("split") != split:
            raise ValueError(f"dataset view row {index} does not belong to split {split!r}")
        identity: dict[str, Any] = {}
        for name in (
            "sample_id",
            "shard_id",
            "tensor_key",
            "pocket_size",
            "peptide_length",
        ):
            value = row.get(name)
            if name in {"pocket_size", "peptide_length"}:
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    raise ValueError(f"dataset view row {index} has invalid {name}")
            elif not isinstance(value, str) or not value:
                raise ValueError(f"dataset view row {index} has invalid {name}")
            identity[name] = value
        ordered_rows.append(identity)
    if not ordered_rows:
        raise ValueError(f"dataset view split {split!r} is empty")
    view = {
        "schema_version": "apexgen.joint_v2.dataset_view.v1",
        "split": split,
        "sample_count": len(ordered_rows),
        "ordered_rows_sha256": canonical_sha256(ordered_rows),
    }
    return {**view, "identity_sha256": canonical_sha256(view)}


def joint_v2_data_views_identity(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    *,
    train_split: str,
    validation_split: str,
) -> dict[str, Any]:
    """Bind both training and validation selections, including their ordering."""

    if train_split == validation_split:
        raise ValueError("training and validation splits must differ")
    views = {
        "schema_version": "apexgen.joint_v2.data_views.v1",
        "train": joint_v2_dataset_view_identity(train_rows, split=train_split),
        "validation": joint_v2_dataset_view_identity(validation_rows, split=validation_split),
    }
    return {**views, "identity_sha256": canonical_sha256(views)}


def _metadata(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"dataset metadata is not an object: {path}")
    return payload


def joint_v2_dataset_identity(
    pocket_root: str | Path, target_root: str | Path | None = None
) -> dict[str, Any]:
    """Verify and hash every manifest/tensor store consumed by Joint-v2."""

    pocket_root = Path(pocket_root)
    pocket_manifest = pocket_root / "manifest.parquet"
    pocket_metadata_path = pocket_root / "metadata.json"
    pocket_metadata = _metadata(pocket_metadata_path)
    pocket_manifest_sha = sha256_file(pocket_manifest)
    if pocket_metadata.get("manifest_sha256") != pocket_manifest_sha:
        raise ValueError("pocket manifest digest differs from pocket metadata")
    declared_shards = pocket_metadata.get("shards")
    if not isinstance(declared_shards, list) or not declared_shards:
        raise ValueError("pocket metadata has no shard identity list")
    shard_digests: dict[str, str] = {}
    for row in declared_shards:
        if (not isinstance(row, dict) or not isinstance(row.get("shard_id"), str)
                or not row["shard_id"]):
            raise ValueError("pocket metadata contains an invalid shard identity")
        shard_id = row["shard_id"]
        if shard_id in shard_digests:
            raise ValueError(f"pocket metadata contains a duplicate shard identity: {shard_id}")
        digest = sha256_file(pocket_root / "shards" / shard_id / "data.mdb")
        if row.get("data_mdb_sha256") != digest:
            raise ValueError(f"pocket tensor shard digest mismatch: {shard_id}")
        shard_digests[shard_id] = digest

    # The loader follows the manifest, not the metadata's shard list. Bind
    # every included split so omitted declarations cannot bypass resume checks.
    referenced_shards: set[str] = set()
    manifest = pq.ParquetFile(pocket_manifest)
    for batch in manifest.iter_batches(columns=["status", "shard_id"]):
        for row in batch.to_pylist():
            if row["status"] != "included":
                continue
            shard_id = row["shard_id"]
            if not isinstance(shard_id, str) or not shard_id:
                raise ValueError("included manifest row has an invalid shard identity")
            referenced_shards.add(shard_id)
    undeclared = referenced_shards - shard_digests.keys()
    if undeclared:
        raise ValueError(f"manifest references undeclared tensor shards: {sorted(undeclared)}")

    from apexgen.joint_v2.data.dataset import COMPLEX_DATASET_SCHEMA, COMPLEX_RECORD_SCHEMA

    if target_root is None:
        if (
            pocket_metadata.get("schema_version") != COMPLEX_DATASET_SCHEMA
            or pocket_metadata.get("record_schema_version") != COMPLEX_RECORD_SCHEMA
        ):
            raise ValueError("unified complex dataset metadata required without target_root")
        identity = {
            "schema_version": "apexgen.joint_v2.dataset_identity.v2",
            "storage": "unified_complex",
            "manifest_sha256": pocket_manifest_sha,
            "metadata_sha256": sha256_file(pocket_metadata_path),
            "shard_data_sha256": shard_digests,
        }
        return {**identity, "identity_sha256": canonical_sha256(identity)}
    if pocket_metadata.get("schema_version") == COMPLEX_DATASET_SCHEMA:
        raise ValueError("unified complex dataset must not use a legacy target_root")
    target_root = Path(target_root)
    target_manifest = target_root / "manifest.parquet"
    target_metadata_path = target_root / "metadata.json"
    target_metadata = _metadata(target_metadata_path)
    target_manifest_sha = sha256_file(target_manifest)
    if target_metadata.get("manifest_sha256") != target_manifest_sha:
        raise ValueError("target manifest digest differs from target metadata")
    if target_metadata.get("legacy_manifest_sha256") != pocket_manifest_sha:
        raise ValueError("target sidecar was not built from the active pocket manifest")
    target_data_sha = sha256_file(target_root / "targets.lmdb" / "data.mdb")
    if target_metadata.get("targets_lmdb_data_sha256") != target_data_sha:
        raise ValueError("target tensor-store digest differs from target metadata")
    identity = {
        "schema_version": "apexgen.joint_v2.dataset_identity.v1",
        "pocket_manifest_sha256": pocket_manifest_sha,
        "pocket_metadata_sha256": sha256_file(pocket_metadata_path),
        "pocket_shard_data_sha256": shard_digests,
        "target_manifest_sha256": target_manifest_sha,
        "target_metadata_sha256": sha256_file(target_metadata_path),
        "target_data_mdb_sha256": target_data_sha,
    }
    return {**identity, "identity_sha256": canonical_sha256(identity)}
