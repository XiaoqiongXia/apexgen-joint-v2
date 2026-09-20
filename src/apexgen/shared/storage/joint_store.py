"""Pickle-free compact storage for joint-v1 peptide target sidecars."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lmdb
import msgpack
import numpy as np

# Kept local so storage readers do not import the legacy Joint-v1 state runtime.
JOINT_RECORD_SCHEMA_VERSION = "apexgen.tensor_record.joint_v1.v2"


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data": array.tobytes(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _encode(child) for key, child in value.items()}
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    raise TypeError(f"unsupported joint target value: {type(value).__name__}")


def _decode(value: Any) -> Any:
    if isinstance(value, dict) and value.get("__ndarray__") is True:
        if set(value) != {"__ndarray__", "dtype", "shape", "data"}:
            raise ValueError("invalid joint ndarray envelope")
        return (
            np.frombuffer(value["data"], dtype=np.dtype(value["dtype"]))
            .reshape(tuple(value["shape"]))
            .copy()
        )
    if isinstance(value, dict):
        return {key: _decode(child) for key, child in value.items()}
    return value


def pack_joint_target(record: dict[str, Any]) -> bytes:
    if record.get("schema_version") != JOINT_RECORD_SCHEMA_VERSION:
        raise ValueError("joint target record schema mismatch")
    return msgpack.packb(_encode(record), use_bin_type=True, strict_types=True)


def unpack_joint_target(payload: bytes) -> dict[str, Any]:
    record = _decode(msgpack.unpackb(payload, raw=False, strict_map_key=True))
    if not isinstance(record, dict) or record.get("schema_version") != JOINT_RECORD_SCHEMA_VERSION:
        raise ValueError("joint target record schema mismatch")
    return record


def open_joint_targets(path: str | Path) -> lmdb.Environment:
    return lmdb.open(
        str(Path(path)),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=2048,
        subdir=True,
    )
