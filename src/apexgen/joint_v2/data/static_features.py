"""Versioned, coordinate-bound static features for Joint-v2 records.

Only deterministic observed geometry is cached. No noisy state, padding, or
learned encoder outputs are persisted here.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from apexgen.joint_v2.contracts.contract import JOINT_V2_CONTRACT_SHA256
from apexgen.shared.geometry.joint_residue_constants import ATOM14_CONSTANTS_SHA256

STATIC_FEATURES_KEY = "joint_v2_static_features"
STATIC_FEATURES_SCHEMA = "apexgen.joint_v2.static_features.v1"
UPGRADE_MARKER = "static_features_upgrade.inprogress.json"

POCKET_SPECS = {
    "sequence_index": ("int64", ()),
    "chain_index": ("int64", ()),
    "backbone_angles_sin_cos": ("float32", (3, 2)),
    "backbone_angle_mask": ("bool", (3,)),
    "sidechain_angles_sin_cos": ("float32", (4, 2)),
    "sidechain_angle_mask": ("bool", (4,)),
}
PEPTIDE_SPECS = {
    "translation": ("float32", (3,)),
    "rotation": ("float32", (3, 3)),
    "backbone_angles_sin_cos": ("float32", (3, 2)),
    "backbone_angle_mask": ("bool", (3,)),
    "sidechain_angles_sin_cos": ("float32", (4, 2)),
    "sidechain_angle_mask": ("bool", (4,)),
}


def _hash_value(digest, value):
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(json.dumps([array.dtype.str, array.shape]).encode())
        digest.update(array.tobytes())
    elif isinstance(value, dict):
        for key in sorted(value):
            digest.update(json.dumps(key).encode())
            _hash_value(digest, value[key])
    else:
        digest.update(json.dumps(value, sort_keys=True, separators=(",", ":"),
            default=lambda item: item.item() if isinstance(item, np.generic) else str(item)).encode())
    digest.update(b"\0")


def _digest(value):
    digest = hashlib.sha256()
    _hash_value(digest, value)
    return digest.hexdigest()


def input_fingerprint(record):
    """Invalidate cached labels after any coordinate, chemistry or topology edit."""
    fields = ("sample_id", "peptide_length", "pocket_aatype", "pocket_atom_xyz",
              "pocket_atom_mask", "pocket_residue_rotation", "pocket_residue_translation",
              "pocket_core_mask", "pocket_residue_keys", "pocket_backbone_link_mask")
    target = record["joint_v2_target"]
    return _digest({
        "record": {key: record.get(key) for key in fields},
        "target": {key: target[key] for key in
                   ("aatype", "experimental_atom14", "experimental_atom14_mask")},
        "atom14_constants_sha256": ATOM14_CONSTANTS_SHA256,
    })


def read_static_features(record):
    """Return a checked cache, or None for an explicitly legacy record."""
    if STATIC_FEATURES_KEY not in record:
        return None
    cache = record[STATIC_FEATURES_KEY]
    if not isinstance(cache, dict) or cache.get("schema_version") != STATIC_FEATURES_SCHEMA:
        raise ValueError("unsupported Joint-v2 static feature schema")
    if cache.get("joint_v2_contract_sha256") != JOINT_V2_CONTRACT_SHA256:
        raise ValueError("static feature model contract mismatch")
    if cache.get("input_sha256") != input_fingerprint(record):
        raise ValueError("stale static features: observed coordinates or topology changed; rebuild cache")
    for role, length, specs in (
        ("pocket", len(record["pocket_aatype"]), POCKET_SPECS),
        ("peptide", int(record["peptide_length"]), PEPTIDE_SPECS),
    ):
        values = cache.get(role)
        if not isinstance(values, dict) or set(values) != set(specs):
            raise ValueError(f"incomplete {role} static features")
        for name, (dtype, tail) in specs.items():
            array = values[name]
            if (not isinstance(array, np.ndarray) or array.dtype != np.dtype(dtype)
                    or array.shape != (length, *tail)):
                raise ValueError(f"invalid cached {role}.{name} shape/dtype")
            if array.dtype.kind == "f" and not np.isfinite(array).all():
                raise ValueError(f"nonfinite cached {role}.{name}")
    if cache.get("features_sha256") != _digest({role: cache[role] for role in ("pocket", "peptide")}):
        raise ValueError("static feature checksum mismatch")
    return cache


def precompute_record(record: dict[str, Any], *, verify: bool = False) -> dict[str, Any]:
    """Attach static features using exactly the legacy observed-atom collator."""
    from apexgen.joint_v2.data.batch import collate_joint_v2_records

    # Always rebuild from observed atoms, never trust legacy frame sidecars.
    legacy = {key: value for key, value in record.items() if key != STATIC_FEATURES_KEY}
    batch = collate_joint_v2_records([legacy])
    length = len(record["pocket_aatype"])
    p, q = slice(0, length), slice(length, length + int(record["peptide_length"]))
    def array(value, part):
        return value[0, part].detach().cpu().numpy().copy()
    pocket = {}
    for name in POCKET_SPECS:
        field = name if name in ("sequence_index", "chain_index") else "pocket_" + name
        pocket[name] = array(getattr(batch.condition, field), p)
    peptide = {}
    for name in PEPTIDE_SPECS:
        field = "endpoint_" + name if name in ("translation", "rotation") else name
        peptide[name] = array(getattr(batch.targets, field), q)
    record[STATIC_FEATURES_KEY] = dict(
        schema_version=STATIC_FEATURES_SCHEMA,
        joint_v2_contract_sha256=JOINT_V2_CONTRACT_SHA256,
        input_sha256=input_fingerprint(record),
        features_sha256=_digest(dict(pocket=pocket, peptide=peptide)),
        pocket=pocket, peptide=peptide,
    )
    if verify:
        import torch
        actual = collate_joint_v2_records([record])
        for role in ("condition", "targets"):
            for name, value in vars(getattr(batch, role)).items():
                if not torch.equal(value, getattr(getattr(actual, role), name)):
                    raise ValueError(f"static cache changed model tensors: {role}.{name}")
    return record
