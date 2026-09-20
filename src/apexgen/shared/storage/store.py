"""Pickle-free ndarray serialization and immutable sharded LMDB access."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import lmdb
import msgpack
import numpy as np


RECORD_SCHEMA_VERSION = "apexgen.tensor_record.v0"


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
    if isinstance(value, (list, tuple)):
        return [_encode(child) for child in value]
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    raise TypeError(f"unsupported tensor-record value: {type(value).__name__}")


def _decode(value: Any) -> Any:
    if isinstance(value, dict) and value.get("__ndarray__") is True:
        expected = {"__ndarray__", "dtype", "shape", "data"}
        if set(value) != expected:
            raise ValueError("invalid ndarray envelope")
        array = np.frombuffer(value["data"], dtype=np.dtype(value["dtype"]))
        return array.reshape(tuple(value["shape"])).copy()
    if isinstance(value, dict):
        return {key: _decode(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_decode(child) for child in value]
    return value


def pack_record(record: dict[str, Any]) -> bytes:
    """Serialize one schema-tagged record without executable Python objects."""

    if record.get("schema_version") != RECORD_SCHEMA_VERSION:
        raise ValueError(f"record must declare {RECORD_SCHEMA_VERSION}")
    return msgpack.packb(_encode(record), use_bin_type=True, strict_types=True)


def unpack_record(payload: bytes) -> dict[str, Any]:
    """Deserialize and validate one tensor record."""

    decoded = _decode(msgpack.unpackb(payload, raw=False, strict_map_key=True))
    if not isinstance(decoded, dict) or decoded.get("schema_version") != RECORD_SCHEMA_VERSION:
        raise ValueError("tensor record schema mismatch")
    return decoded


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def open_readonly_lmdb(path: str | Path) -> lmdb.Environment:
    """Open an immutable shard with settings safe for DataLoader workers."""

    return lmdb.open(
        str(Path(path)),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=2048,
        subdir=True,
    )
