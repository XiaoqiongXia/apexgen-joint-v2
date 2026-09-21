"""Lightweight batch assembly after startup verification of immutable LMDB data.

The ordinary collator remains the strict preprocessing/debug boundary. This
module never computes geometry or hashes per-record arrays on the training path.
Do not use its light mode for records edited or augmented after dataset loading.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from apexgen.joint_v2.contracts.contract import (
    JOINT_V2_CONTRACT_SHA256, PeptideNativeTargets, UnifiedComplexCondition,
)
from apexgen.joint_v2.contracts.state import UNKNOWN_AATYPE, debug_invariants_enabled
from apexgen.joint_v2.data.batch import JointV2Batch, collate_joint_v2_records
from apexgen.joint_v2.data.static_features import (
    STATIC_FEATURES_KEY, STATIC_FEATURES_SCHEMA, POCKET_SPECS, PEPTIDE_SPECS, UPGRADE_MARKER,
)
from apexgen.joint_v2.runtime.lineage import joint_v2_dataset_identity, sha256_file


CONDITION_SPECS = {
    "residue_mask": (torch.bool, ()), "pocket_mask": (torch.bool, ()),
    "peptide_mask": (torch.bool, ()), "pocket_core_mask": (torch.bool, ()),
    "pocket_translation": (torch.float32, (3,)), "pocket_rotation": (torch.float32, (3, 3)),
    "aatype": (torch.long, ()), "sequence_index": (torch.long, ()), "chain_index": (torch.long, ()),
    "pocket_atom_xyz": (torch.float32, (38, 3)), "pocket_atom_mask": (torch.bool, (38,)),
    "pocket_backbone_angles_sin_cos": (torch.float32, (3, 2)),
    "pocket_backbone_angle_mask": (torch.bool, (3,)),
    "pocket_sidechain_angles_sin_cos": (torch.float32, (4, 2)),
    "pocket_sidechain_angle_mask": (torch.bool, (4,)),
}
TARGET_SPECS = {
    "endpoint_aatype": (torch.long, ()),
    "endpoint_translation": (torch.float32, (3,)), "endpoint_rotation": (torch.float32, (3, 3)),
    "atom14_xyz": (torch.float32, (14, 3)), "atom14_mask": (torch.bool, (14,)),
    "backbone_xyz": (torch.float32, (3, 3)), "backbone_atom_mask": (torch.bool, (3,)),
    "backbone_angles_sin_cos": (torch.float32, (3, 2)), "backbone_angle_mask": (torch.bool, (3,)),
    "sidechain_angles_sin_cos": (torch.float32, (4, 2)), "sidechain_angle_mask": (torch.bool, (4,)),
}


def _array(values, key, length, dtype, tail):
    value = values.get(key)
    if not isinstance(value, np.ndarray) or value.shape != (length, *tail) or value.dtype != np.dtype(dtype):
        raise ValueError(f"invalid training array {key}: shape/dtype mismatch")
    return torch.from_numpy(value)


def _cache(record, p_length, q_length):
    cache = record.get(STATIC_FEATURES_KEY)
    if not isinstance(cache, dict) or cache.get("schema_version") != STATIC_FEATURES_SCHEMA:
        raise ValueError("light training requires a supported static feature cache in every record")
    if cache.get("joint_v2_contract_sha256") != JOINT_V2_CONTRACT_SHA256:
        raise ValueError("training static feature contract mismatch")
    result = []
    for role, length, specs in (("pocket", p_length, POCKET_SPECS), ("peptide", q_length, PEPTIDE_SPECS)):
        values = cache.get(role)
        if not isinstance(values, dict) or set(values) != set(specs):
            raise ValueError(f"incomplete training {role} static features")
        result.append({key: _array(values, key, length, dtype, tail)
                       for key, (dtype, tail) in specs.items()})
    return result


def _collate_verified_records(records):
    if not records:
        raise ValueError("cannot collate an empty training batch")
    p_lengths = [len(record["pocket_aatype"]) for record in records]
    q_lengths = [int(record["peptide_length"]) for record in records]
    if any(p < 1 for p in p_lengths) or any(q < 3 for q in q_lengths):
        raise ValueError("training requires a nonempty pocket and peptide length >= 3")
    layout = (len(records), max(p + q for p, q in zip(p_lengths, q_lengths, strict=True)))
    condition = {name: torch.zeros(*layout, *tail, dtype=dtype)
                 for name, (dtype, tail) in CONDITION_SPECS.items()}
    targets = {name: torch.zeros(*layout, *tail, dtype=dtype)
               for name, (dtype, tail) in TARGET_SPECS.items()}
    condition["aatype"].fill_(UNKNOWN_AATYPE)
    targets["endpoint_aatype"].fill_(UNKNOWN_AATYPE)
    condition["pocket_rotation"][:] = torch.eye(3)
    targets["endpoint_rotation"][:] = torch.eye(3)
    for row, (record, p_length, q_length) in enumerate(zip(records, p_lengths, q_lengths, strict=True)):
        target = record["joint_v2_target"]
        pc, qc = _cache(record, p_length, q_length)
        p, q = slice(0, p_length), slice(p_length, p_length + q_length)
        paa = _array(record, "pocket_aatype", p_length, "int64", ())
        qaa = _array(target, "aatype", q_length, "int64", ())
        # Cheap index checks remain before embedding and categorical losses.
        if any(bool(((value < 0) | (value >= UNKNOWN_AATYPE)).any()) for value in (paa, qaa)):
            raise ValueError("training amino-acid index must lie in [0, 20)")
        xyz = _array(record, "pocket_atom_xyz", p_length, "float32", (38, 3))
        atom_mask = _array(record, "pocket_atom_mask", p_length, "bool", (38,))
        qxyz = _array(target, "experimental_atom14", q_length, "float32", (14, 3))
        qmask = _array(target, "experimental_atom14_mask", q_length, "bool", (14,))
        condition["residue_mask"][row, :p_length + q_length] = True
        condition["pocket_mask"][row, p] = True
        condition["peptide_mask"][row, q] = True
        condition["aatype"][row, p] = paa
        for field, source, dtype, tail in (
            ("pocket_translation", "pocket_residue_translation", "float32", (3,)),
            ("pocket_rotation", "pocket_residue_rotation", "float32", (3, 3)),
            ("pocket_core_mask", "pocket_core_mask", "bool", ()),
        ):
            condition[field][row, p] = _array(record, source, p_length, dtype, tail)
        condition["pocket_atom_xyz"][row, p] = torch.where(atom_mask[..., None], xyz, 0.0)
        condition["pocket_atom_mask"][row, p] = atom_mask
        for key, value in pc.items():
            field = key if key in ("sequence_index", "chain_index") else "pocket_" + key
            condition[field][row, p] = value
        condition["sequence_index"][row, q] = torch.arange(q_length)
        condition["chain_index"][row, q] = int(pc["chain_index"].max()) + 1
        targets["endpoint_aatype"][row, q] = qaa
        targets["atom14_xyz"][row, q] = torch.where(qmask[..., None], qxyz, 0.0)
        targets["atom14_mask"][row, q] = qmask
        targets["backbone_xyz"][row, q] = torch.where(qmask[:, :3, None], qxyz[:, :3], 0.0)
        targets["backbone_atom_mask"][row, q] = qmask[:, :3]
        for key, value in qc.items():
            field = "endpoint_" + key if key in ("translation", "rotation") else key
            targets[field][row, q] = value
    return JointV2Batch(
        sample_ids=tuple(str(record["sample_id"]) for record in records),
        source_pdb_ids=tuple(str(record["source_pdb_id"]) for record in records),
        splits=tuple(str(record["split"]) for record in records),
        peptide_lengths=tuple(q_lengths),
        condition=UnifiedComplexCondition(**condition), targets=PeptideNativeTargets(**targets),
    )


@dataclass(frozen=True)
class TrainingCollator:
    """Picklable collator returned only after the training startup preflight."""

    report: dict

    def __call__(self, records):
        if self.report["mode"] == "full" or debug_invariants_enabled():
            return collate_joint_v2_records(records)
        return _collate_verified_records(records)


def prepare_training_collator(dataset, *, geometry_checks="auto", verified_identity=None, sample_count=8):
    """Verify publication/identity once and fully validate sampled native records.

    ``verified_identity`` may be supplied by a caller that has already run
    joint_v2_dataset_identity (including a rank-zero verified DDP broadcast).
    Otherwise this function verifies all manifest and shard hashes itself.
    ``auto`` enables light mode only for a versioned static-feature dataset.
    """
    if geometry_checks not in ("auto", "full"):
        raise ValueError("geometry_checks must be auto or full")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError("sample_count must be a positive integer")
    if not len(dataset):
        raise ValueError("training split is empty")
    source = dataset
    while hasattr(source, "source"):
        source = source.source
    root = source.pockets.root
    if (root / UPGRADE_MARKER).exists():
        raise RuntimeError("dataset static feature upgrade is incomplete")
    identity = (joint_v2_dataset_identity(root, source.target_root)
                if verified_identity is None else verified_identity)
    metadata_key = "metadata_sha256" if source.target_root is None else "pocket_metadata_sha256"
    manifest_key = "manifest_sha256" if source.target_root is None else "pocket_manifest_sha256"
    if (identity.get(metadata_key) != sha256_file(root / "metadata.json")
            or identity.get(manifest_key) != sha256_file(root / "manifest.parquet")):
        raise ValueError("training dataset identity changed after startup verification")
    mode = "full"
    if (geometry_checks == "auto" and not debug_invariants_enabled()
            and source.target_root is None and source._static_features_schema == STATIC_FEATURES_SCHEMA):
        mode = "light"
    count = min(sample_count, len(dataset))
    indices = sorted({i * (len(dataset) - 1) // max(count - 1, 1) for i in range(count)})
    try:
        for index in indices:
            record = dataset[index]
            strict = collate_joint_v2_records([record])
            strict.condition.validate_model_input()
            if mode == "light":
                light = _collate_verified_records([record])
                for role in ("condition", "targets"):
                    for name, value in vars(getattr(strict, role)).items():
                        if not torch.equal(value, getattr(getattr(light, role), name)):
                            raise ValueError(f"training collation differs from strict path: {role}.{name}")
    finally:
        # Avoid inherited LMDB handles when DataLoader workers fork or spawn.
        dataset.close()
    return TrainingCollator(dict(
        mode=mode, requested=geometry_checks, dataset_identity_sha256=identity["identity_sha256"],
        static_features_schema=source._static_features_schema,
        startup_checked_indices=indices, startup_checked_samples=len(indices),
        per_batch_geometry_checks=(mode == "full"),
        debug_override="APEXGEN_JOINT_V2_DEBUG_INVARIANTS=1",
    ))
