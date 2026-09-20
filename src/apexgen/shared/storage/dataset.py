"""Readonly manifest-indexed access to ApexGen sharded tensor records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from torch.utils.data import Dataset

from apexgen.shared.storage.store import open_readonly_lmdb, unpack_record


def identity_record(record: dict[str, Any]) -> dict[str, Any]:
    """Keep variable-length NumPy records unchanged until the batch collator pads them."""

    return record


class ApexGenDataset(Dataset[dict[str, Any]]):
    """Lazy per-process LMDB reader over included rows of one split."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        environments: dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        table = pq.read_table(self.root / "manifest.parquet")
        rows = table.to_pylist()
        self.rows = [row for row in rows if row["status"] == "included" and row["split"] == split]
        self._environments = {} if environments is None else environments

    def __len__(self) -> int:
        return len(self.rows)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_environments"] = {}
        return state

    def close(self) -> None:
        """Close environments before changing worker topology or discarding the dataset."""

        for environment in self._environments.values():
            environment.close()
        self._environments.clear()

    def _environment(self, shard_id: str):
        environment = self._environments.get(shard_id)
        if environment is None:
            environment = open_readonly_lmdb(self.root / "shards" / shard_id)
            self._environments[shard_id] = environment
        return environment

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        environment = self._environment(row["shard_id"])
        with environment.begin(write=False, buffers=True) as transaction:
            payload = transaction.get(row["tensor_key"].encode("utf-8"))
            if payload is None:
                raise KeyError(f"missing tensor key {row['tensor_key']}")
            record = unpack_record(bytes(payload))
        if record["sample_id"] != row["sample_id"]:
            raise RuntimeError("manifest/tensor sample identity mismatch")
        return record
