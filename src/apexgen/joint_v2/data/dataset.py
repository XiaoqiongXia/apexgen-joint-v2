"""Read complete complex records; support explicitly selected legacy sidecars."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from apexgen.shared.storage.dataset import ApexGenDataset
from apexgen.shared.storage.joint_store import open_joint_targets, unpack_joint_target


COMPLEX_RECORD_SCHEMA = "apexgen.complex_record.joint_v2.v1"
COMPLEX_DATASET_SCHEMA = "apexgen.complex_dataset.joint_v2.v1"
NATIVE_TARGET_SCHEMA = "apexgen.peptide_native_target.v1"


class JointV2Dataset(Dataset[dict[str, Any]]):
    """One LMDB read per complex; target_root opts into the legacy split format.

    New usage: JointV2Dataset(dataset_root, split="train").
    Both formats expose the same record to collate_joint_v2_records, which
    derives native frames from observed coordinates.
    """

    def __init__(
        self,
        pocket_root: str | Path,
        target_root: str | Path | None = None,
        split: str = "train",
    ) -> None:
        self.pockets = ApexGenDataset(pocket_root, split)
        self.target_root = Path(target_root) if target_root is not None else None
        self._target_environment = None

    @classmethod
    def from_codesign_config(cls, config, *, role="train", base_dir="."):
        from apexgen.joint_v2.runtime.config import (
            joint_v2_data_roots, resolve_codesign_data_config,
        )

        if role not in {"train", "validation"}:
            raise ValueError("dataset role must be train or validation")
        data = resolve_codesign_data_config(config, base_dir=base_dir)
        root, targets = joint_v2_data_roots(data)
        return cls(root, targets, data[f"{role}_split"])

    @property
    def rows(self) -> list[dict[str, Any]]:
        return self.pockets.rows

    def __len__(self) -> int:
        return len(self.pockets)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_target_environment"] = None
        return state

    def close(self) -> None:
        self.pockets.close()
        if self._target_environment is not None:
            self._target_environment.close()
            self._target_environment = None

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.pockets[index]
        if self.target_root is None:
            if record.get("complex_schema_version") != COMPLEX_RECORD_SCHEMA:
                raise ValueError(
                    "expected a unified complex record; legacy data requires target_root"
                )
            target = record.get("joint_v2_target")
            if not isinstance(target, dict) or target.get("schema_version") != NATIVE_TARGET_SCHEMA:
                raise ValueError("unified complex record has no valid native peptide target")
        else:
            if "complex_schema_version" in record:
                raise ValueError("unified complex records must not be combined with legacy targets")
            if self._target_environment is None:
                self._target_environment = open_joint_targets(self.target_root / "targets.lmdb")
            with self._target_environment.begin(write=False, buffers=True) as transaction:
                payload = transaction.get(record["sample_id"].encode())
            if payload is None:
                raise KeyError(f"missing geometry target for {record['sample_id']}")
            target = unpack_joint_target(bytes(payload))
        if target.get("sample_id") != record.get("sample_id") or int(
            target.get("peptide_length", -1)
        ) != int(record.get("peptide_length", -2)):
            raise RuntimeError("complex record and peptide target identity differ")
        result = {**record, "joint_v2_target": target}
        if "boltz_adapter" in result:
            from apexgen.joint_v2.data.boltz_npz import validate_boltz_record_contract
            validate_boltz_record_contract(result)
            # Packaged datasets carry content-addressed sources. Prefer those
            # after relocation; the recorded absolute path is provenance only.
            digest = result.get("raw_file_sha256", "")
            if len(digest) == 64 and all(c in "0123456789abcdef" for c in digest):
                local_source = self.pockets.root / "sources" / f"{digest}.npz"
                if local_source.is_file():
                    result["source_recorded_raw_path"] = result["raw_path"]
                    result["raw_path"] = str(local_source.resolve())
        return result


def deterministic_subset_indices(
    rows: list[dict[str, Any]], *, count: int, seed: int
) -> tuple[int, ...]:
    """Select a reproducible random-looking subset without RNG-version coupling."""

    if isinstance(count, bool) or not isinstance(count, int) or not 0 < count <= len(rows):
        raise ValueError("subset count must lie in [1, len(rows)]")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("subset seed must be a non-negative integer")
    ranked: list[tuple[str, int]] = []
    for index, row in enumerate(rows):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"dataset row {index} has no valid sample_id")
        digest = hashlib.sha256(f"{seed}\0{sample_id}\0{index}".encode()).hexdigest()
        ranked.append((digest, index))
    return tuple(index for _, index in sorted(ranked)[:count])


def deterministic_categorical_matched_subset_indices(
    rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    field: str,
) -> tuple[int, ...]:
    """Select a deterministic subset with Hamilton-apportioned reference categories."""

    if isinstance(count, bool) or not isinstance(count, int) or not 0 < count <= len(rows):
        raise ValueError("subset count must lie in [1, len(rows)]")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("subset seed must be a non-negative integer")
    if not isinstance(field, str) or not field:
        raise ValueError("matched subset field must be a non-empty string")
    if not reference_rows:
        raise ValueError("matched subset reference rows must be non-empty")

    source_groups: dict[int, list[tuple[str, int]]] = {}
    for index, row in enumerate(rows):
        sample_id = row.get("sample_id")
        value = row.get(field)
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"dataset row {index} has no valid sample_id")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"dataset row {index} has no integer {field}")
        digest = hashlib.sha256(
            f"{seed}\0{field}\0{value}\0{sample_id}\0{index}".encode()
        ).hexdigest()
        source_groups.setdefault(value, []).append((digest, index))

    reference_counts: dict[int, int] = {}
    for index, row in enumerate(reference_rows):
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"reference row {index} has no integer {field}")
        reference_counts[value] = reference_counts.get(value, 0) + 1

    reference_total = len(reference_rows)
    exact_quotas = {
        value: count * category_count / reference_total
        for value, category_count in reference_counts.items()
    }
    quotas = {value: math.floor(exact) for value, exact in exact_quotas.items()}
    remaining = count - sum(quotas.values())
    remainder_order = sorted(
        reference_counts,
        key=lambda value: (
            -(exact_quotas[value] - quotas[value]),
            -reference_counts[value],
            value,
        ),
    )
    for value in remainder_order[:remaining]:
        quotas[value] += 1

    insufficient = {
        value: {"required": quota, "available": len(source_groups.get(value, ()))}
        for value, quota in quotas.items()
        if len(source_groups.get(value, ())) < quota
    }
    if insufficient:
        raise ValueError(f"source rows cannot satisfy matched {field} quotas: {insufficient}")

    selected: list[tuple[str, int]] = []
    for value, quota in quotas.items():
        selected.extend(sorted(source_groups[value])[:quota])
    if len(selected) != count:
        raise RuntimeError("matched subset apportionment produced the wrong count")
    return tuple(index for _, index in sorted(selected))


class JointV2DatasetView(Dataset[dict[str, Any]]):
    """Ordered, close-owning view over an explicitly selected Joint-v2 dataset."""

    def __init__(self, source: JointV2Dataset, indices: tuple[int, ...]) -> None:
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("dataset view indices must be non-empty and unique")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(source)
            for index in indices
        ):
            raise ValueError("dataset view index lies outside the source dataset")
        self.source = source
        self.indices = indices
        self._rows = [source.rows[index] for index in indices]

    @property
    def rows(self) -> list[dict[str, Any]]:
        return self._rows

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.source[self.indices[index]]

    def close(self) -> None:
        self.source.close()
