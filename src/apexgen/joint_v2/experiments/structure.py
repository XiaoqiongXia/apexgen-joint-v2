"""PDB/mmCIF artifacts for Joint-v2 backbone rollout candidates."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from apexgen.shared.geometry.joint_residue_constants import (
    AA1_ORDER,
    AA3_ORDER,
    ATOM14_MASK,
    CHI_EXISTS,
    as_torch,
)
from apexgen.shared.io.joint_structure import write_joint_structure
from apexgen.joint_v2.contracts.contract import JOINT_V2_CONTRACT_SHA256, require_joint_v2_contract
from apexgen.joint_v2.runtime.lineage import sha256_file
from apexgen.joint_v2.contracts.state import AMINO_ACID_TYPES, SEQUENCE_EPSILON
from apexgen.joint_v2.evaluation.validation import JointV2RolloutCandidate
from apexgen.joint_v2.evaluation.full_structure import full_structure_metrics


ROLLOUT_CANDIDATE_SCHEMA_VERSION = (
    "apexgen.joint_v2.sequence_structure_endpoint.rollout_candidate.v3"
)
_ROLLOUT_CANDIDATE_KEYS = {
    "schema_version",
    "joint_v2_contract_sha256",
    "sample_id",
    "source_pdb_id",
    "optimizer_step",
    "base_index",
    "base_seed",
    "peptide_chain_id",
    "receptor_chain_ids",
    "peptide_length",
    "predicted_sequence",
    "sequence_decoding",
    "sequence_epsilon",
    "sequence_status",
    "predicted_aatype",
    "predicted_residue_names",
    "sidechain_status",
    "predicted_chi_radians",
    "predicted_chi_mask",
    "chi_source",
    "angle_query_time",
    "oxygen_source",
    "terminal_oxygen_policy",
    "coordinate_frame",
    "structure_file",
    "raw_file_sha256",
    "metrics",
    "provenance",
}


def safe_sample_id(sample_id: str) -> str:
    """Return a path-safe, non-empty representation of a sample identifier."""

    safe = sample_id.replace("/", "_").replace("\\", "_").strip(". ")
    if not safe or safe in {".", ".."}:
        raise ValueError("sample_id cannot be represented as a safe artifact path")
    return safe


def verify_joint_v2_rollout_source(
    record: dict[str, Any], *, raw_structure_root: str | Path | None = None
) -> Path:
    """Resolve and verify the immutable raw complex referenced by a record."""

    raw_path = Path(str(record.get("raw_path", "")))
    expected_raw_sha256 = record.get("raw_file_sha256")
    if not isinstance(expected_raw_sha256, str) or len(expected_raw_sha256) != 64:
        raise ValueError("rollout source record has no valid raw_file_sha256")
    candidates = [raw_path]
    if raw_structure_root is not None:
        candidates.append(Path(raw_structure_root) / raw_path.name)
    resolved = next((candidate for candidate in candidates if candidate.is_file()), None)
    if resolved is None:
        attempted = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"rollout source structure is absent: {attempted}")
    if sha256_file(resolved) != expected_raw_sha256:
        raise ValueError("rollout source structure digest differs from the dataset record")
    return resolved


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
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


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"rollout candidate {name} must be a non-negative integer")
    return value


def validate_joint_v2_rollout_candidate_payload(payload: dict[str, Any]) -> None:
    """Reject incomplete, legacy, or internally inconsistent candidate sidecars."""

    if not isinstance(payload, dict) or set(payload) != _ROLLOUT_CANDIDATE_KEYS:
        observed = set(payload) if isinstance(payload, dict) else set()
        raise ValueError(
            "rollout candidate keys differ: "
            f"missing={sorted(_ROLLOUT_CANDIDATE_KEYS - observed)}, "
            f"extra={sorted(observed - _ROLLOUT_CANDIDATE_KEYS)}"
        )
    if payload["schema_version"] != ROLLOUT_CANDIDATE_SCHEMA_VERSION:
        raise ValueError("rollout candidate schema_version is not full-atom sequence-structure v3")
    require_joint_v2_contract(payload, source="rollout candidate")
    for name in ("sample_id", "peptide_chain_id", "structure_file"):
        if not isinstance(payload[name], str) or not payload[name].strip():
            raise ValueError(f"rollout candidate {name} must be a non-empty string")
    for name in ("source_pdb_id", "predicted_sequence"):
        if not isinstance(payload[name], str):
            raise ValueError(f"rollout candidate {name} must be a string")
    for name in ("optimizer_step", "base_index", "base_seed"):
        _nonnegative_int(payload[name], name)
    length = _nonnegative_int(payload["peptide_length"], "peptide_length")
    if length == 0:
        raise ValueError("rollout candidate peptide_length must be positive")
    receptor_chains = payload["receptor_chain_ids"]
    if (
        not isinstance(receptor_chains, list)
        or not receptor_chains
        or any(not isinstance(chain, str) or not chain for chain in receptor_chains)
    ):
        raise ValueError("rollout candidate receptor_chain_ids must be non-empty strings")
    aatype = payload["predicted_aatype"]
    if (
        not isinstance(aatype, list)
        or len(aatype) != length
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < AMINO_ACID_TYPES
            for index in aatype
        )
    ):
        raise ValueError("rollout candidate predicted_aatype must be int64 values in [0, 20)")
    expected_sequence = "".join(AA1_ORDER[index] for index in aatype)
    expected_residue_names = [AA3_ORDER[index] for index in aatype]
    if payload["predicted_sequence"] != expected_sequence:
        raise ValueError("rollout candidate predicted_sequence differs from predicted_aatype")
    if payload["predicted_residue_names"] != expected_residue_names:
        raise ValueError("rollout candidate residue names differ from predicted_aatype")
    expected_literals = {
        "sequence_decoding": "argmax",
        "sequence_epsilon": SEQUENCE_EPSILON,
        "sequence_status": "predicted_full_atom_sidechain_reconstructed",
        "sidechain_status": "predicted_chi_atom14_reconstructed",
        "chi_source": "decoder_angle_head",
        "oxygen_source": "final_backbone_psi_head_nonterminal",
        "terminal_oxygen_policy": "fixed_zero_psi_ideal_geometry",
        "coordinate_frame": "global_source_structure_frame",
    }
    for name, expected in expected_literals.items():
        if payload[name] != expected:
            raise ValueError(f"rollout candidate {name} must be {expected!r}")
    angle_query_time = payload["angle_query_time"]
    if (
        isinstance(angle_query_time, bool)
        or not isinstance(angle_query_time, (int, float))
        or not math.isfinite(angle_query_time)
        or not 0.0 <= angle_query_time <= 1.0
    ):
        raise ValueError("rollout candidate angle_query_time must be finite in [0, 1]")
    chi = payload["predicted_chi_radians"]
    chi_mask = payload["predicted_chi_mask"]
    if (
        not isinstance(chi, list)
        or len(chi) != length
        or any(
            not isinstance(row, list)
            or len(row) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in row
            )
            for row in chi
        )
    ):
        raise ValueError("rollout candidate predicted_chi_radians must be finite [L,4]")
    if (
        not isinstance(chi_mask, list)
        or len(chi_mask) != length
        or any(
            not isinstance(row, list)
            or len(row) != 4
            or any(not isinstance(value, bool) for value in row)
            for row in chi_mask
        )
    ):
        raise ValueError("rollout candidate predicted_chi_mask must be bool [L,4]")
    expected_chi_mask = as_torch(
        CHI_EXISTS,
        device=torch.device("cpu"),
        dtype=torch.bool,
    )[torch.tensor(aatype, dtype=torch.long)].tolist()
    if chi_mask != expected_chi_mask:
        raise ValueError("rollout candidate predicted_chi_mask differs from residue identities")
    raw_sha256 = payload["raw_file_sha256"]
    if (
        not isinstance(raw_sha256, str)
        or len(raw_sha256) != 64
        or any(character not in "0123456789abcdef" for character in raw_sha256)
    ):
        raise ValueError("rollout candidate raw_file_sha256 must be lowercase SHA-256")
    metrics = payload["metrics"]
    if not isinstance(metrics, dict) or any(
        not isinstance(name, str)
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for name, value in metrics.items()
    ):
        raise ValueError("rollout candidate metrics must be finite numeric values")
    if not isinstance(payload["provenance"], dict):
        raise ValueError("rollout candidate provenance must be a mapping")
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("rollout candidate payload must be finite JSON data") from error


def load_joint_v2_rollout_candidate_sidecar(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate a Joint-v2 sequence--structure sidecar."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("rollout candidate sidecar must contain a JSON object")
    validate_joint_v2_rollout_candidate_payload(payload)
    return payload


def write_joint_v2_rollout_candidate(
    record: dict[str, Any],
    candidate: JointV2RolloutCandidate,
    output_path: str | Path,
    *,
    optimizer_step: int,
    provenance: dict[str, Any] | None = None,
    raw_structure_root: str | Path | None = None,
) -> None:
    """Replace the native peptide with a predicted full atom14 peptide."""

    output = Path(output_path)
    if output.suffix.lower() not in {".pdb", ".cif", ".mmcif"}:
        raise ValueError("Joint-v2 rollout artifacts must use PDB or mmCIF")
    if str(record.get("sample_id")) != candidate.sample_id:
        raise ValueError("rollout candidate and source record sample IDs differ")
    sidecar = output.with_suffix(output.suffix + ".json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"rollout artifact already exists: {output}")
    if any(not math.isfinite(float(value)) for value in candidate.metrics.values()):
        raise ValueError("rollout candidate metrics must be finite")
    if (
        isinstance(candidate.angle_query_time, bool)
        or not isinstance(candidate.angle_query_time, (int, float))
        or not math.isfinite(candidate.angle_query_time)
        or not 0.0 <= candidate.angle_query_time <= 1.0
    ):
        raise ValueError("rollout candidate angle_query_time must be finite in [0, 1]")
    raw_path = verify_joint_v2_rollout_source(record, raw_structure_root=raw_structure_root)
    observed_raw_sha256 = str(record["raw_file_sha256"])
    backbone = candidate.backbone
    if (
        backbone.ndim != 3
        or backbone.shape[-2:] != (3, 3)
        or not torch.is_floating_point(backbone)
        or not bool(torch.isfinite(backbone).all())
    ):
        raise ValueError("rollout backbone must be finite floating-point [L,3,3]")
    length = backbone.shape[0]
    if not isinstance(candidate.aatype, torch.Tensor) or candidate.aatype.dtype != torch.long:
        raise TypeError("rollout candidate aatype must be int64")
    aatype = candidate.aatype.detach().cpu()
    if aatype.shape != (length,) or bool(((aatype < 0) | (aatype >= 20)).any()):
        raise ValueError("rollout candidate aatype must be int64 [L] in [0, 20)")
    if not isinstance(candidate.atom14, torch.Tensor) or not torch.is_floating_point(
        candidate.atom14
    ):
        raise TypeError("rollout candidate atom14 must be floating point")
    atom14 = candidate.atom14.detach().cpu()
    if atom14.shape != (length, 14, 3) or not bool(torch.isfinite(atom14).all()):
        raise ValueError("rollout candidate atom14 must be finite [L,14,3]")
    if not isinstance(candidate.atom14_mask, torch.Tensor) or candidate.atom14_mask.dtype != torch.bool:
        raise TypeError("rollout candidate atom14_mask must be bool")
    atom14_mask = candidate.atom14_mask.detach().cpu()
    expected_atom14_mask = as_torch(
        ATOM14_MASK,
        device=torch.device("cpu"),
        dtype=torch.bool,
    )[aatype]
    if atom14_mask.shape != (length, 14) or not torch.equal(
        atom14_mask, expected_atom14_mask
    ):
        raise ValueError("rollout candidate atom14_mask differs from residue identities")
    if not torch.equal(atom14[:, :3], backbone.detach().cpu()):
        raise ValueError("rollout candidate atom14 N/CA/C differ from frame backbone")
    if not isinstance(candidate.chi_radians, torch.Tensor) or not torch.is_floating_point(
        candidate.chi_radians
    ):
        raise TypeError("rollout candidate chi_radians must be floating point")
    chi_radians = candidate.chi_radians.detach().cpu()
    if chi_radians.shape != (length, 4) or not bool(torch.isfinite(chi_radians).all()):
        raise ValueError("rollout candidate chi_radians must be finite [L,4]")
    if not isinstance(candidate.chi_mask, torch.Tensor) or candidate.chi_mask.dtype != torch.bool:
        raise TypeError("rollout candidate chi_mask must be bool")
    chi_mask = candidate.chi_mask.detach().cpu()
    expected_chi_mask = as_torch(
        CHI_EXISTS,
        device=torch.device("cpu"),
        dtype=torch.bool,
    )[aatype]
    if chi_mask.shape != (length, 4) or not torch.equal(chi_mask, expected_chi_mask):
        raise ValueError("rollout candidate chi_mask differs from residue identities")
    payload = {
        "schema_version": ROLLOUT_CANDIDATE_SCHEMA_VERSION,
        "joint_v2_contract_sha256": JOINT_V2_CONTRACT_SHA256,
        "sample_id": candidate.sample_id,
        "source_pdb_id": str(record.get("source_pdb_id", "")),
        "optimizer_step": optimizer_step,
        "base_index": candidate.base_index,
        "base_seed": candidate.base_seed,
        "peptide_chain_id": str(record["peptide_chain_id"]),
        "receptor_chain_ids": [str(record["receptor_chain_id"])],
        "peptide_length": length,
        "predicted_sequence": "".join(AA1_ORDER[index] for index in aatype.tolist()),
        "sequence_decoding": "argmax",
        "sequence_epsilon": SEQUENCE_EPSILON,
        "sequence_status": "predicted_full_atom_sidechain_reconstructed",
        "predicted_aatype": aatype.tolist(),
        "predicted_residue_names": [AA3_ORDER[index] for index in aatype.tolist()],
        "sidechain_status": "predicted_chi_atom14_reconstructed",
        "predicted_chi_radians": chi_radians.tolist(),
        "predicted_chi_mask": chi_mask.tolist(),
        "chi_source": "decoder_angle_head",
        "angle_query_time": float(candidate.angle_query_time),
        "oxygen_source": "final_backbone_psi_head_nonterminal",
        "terminal_oxygen_policy": "fixed_zero_psi_ideal_geometry",
        "coordinate_frame": "global_source_structure_frame",
        "structure_file": output.name,
        "raw_file_sha256": observed_raw_sha256,
        "metrics": {**candidate.metrics, **full_structure_metrics(
            record, atom14, atom14_mask, raw_structure_root=raw_structure_root)},
        "provenance": {**({} if provenance is None else provenance),
                       "selected_model_index": record.get("selected_model_index", 0),
                       "environment": "full_selected_source_model_except_replaced_peptide_chain"},
    }
    validate_joint_v2_rollout_candidate_payload(payload)
    write_joint_structure(
        raw_path,
        atom14,
        atom14_mask,
        aatype,
        output,
        site_origin=torch.as_tensor(record["site_origin"], dtype=torch.float32),
        replace_chain_id=str(record["peptide_chain_id"]),
        receptor_chain_ids=(str(record["receptor_chain_id"]),),
        model_index=int(record.get("selected_model_index", 0)),
        preserve_environment=True,
    )
    _atomic_json(sidecar, payload)
