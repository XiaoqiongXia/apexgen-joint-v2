"""Padding and target-safe unified-complex collation for Joint-v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from apexgen.shared.geometry.joint_residue_constants import ATOM14_NAMES
from apexgen.shared.geometry.sidechain import extract_chi
from apexgen.shared.geometry.torsion import extract_backbone_torsions
from apexgen.joint_v2.contracts.contract import PeptideNativeTargets, UnifiedComplexCondition
from apexgen.joint_v2.geometry.rotations import canonicalize_masked_angles
from apexgen.joint_v2.data.native_targets import observed_backbone_frames
from apexgen.joint_v2.data.static_features import read_static_features
from apexgen.joint_v2.contracts.state import (
    BACKBONE_ANGLE_SLOTS,
    POCKET_ATOM_SLOTS,
    SIDECHAIN_ANGLE_SLOTS,
    UNKNOWN_AATYPE,
)


POCKET_ATOM_NAMES = (
    "N",
    "CA",
    "C",
    "O",
    "CB",
    "CG",
    "CG1",
    "CG2",
    "CD",
    "CD1",
    "CD2",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "CZ",
    "CZ2",
    "CZ3",
    "CH2",
    "ND1",
    "ND2",
    "NE",
    "NE1",
    "NE2",
    "NH1",
    "NH2",
    "NZ",
    "OD1",
    "OD2",
    "OE1",
    "OE2",
    "OG",
    "OG1",
    "OH",
    "SD",
    "SG",
    "OXT",
    "SE",
)
POCKET_ATOM_TO_INDEX = {name: index for index, name in enumerate(POCKET_ATOM_NAMES)}
if len(POCKET_ATOM_NAMES) != POCKET_ATOM_SLOTS:
    raise RuntimeError("Joint-v2 pocket atom-slot contract is inconsistent")

_ATOM14_TO_ATOM38 = torch.tensor(
    [[POCKET_ATOM_TO_INDEX.get(name, 0) for name in names] for names in ATOM14_NAMES],
    dtype=torch.long,
)
_ATOM14_PRESENT = torch.tensor(
    [[bool(name) for name in names] for names in ATOM14_NAMES], dtype=torch.bool
)


def _angle_sin_cos(angle: Tensor, mask: Tensor) -> Tensor:
    angle = canonicalize_masked_angles(angle.float(), mask)
    value = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
    return torch.where(mask[..., None], value, 0.0)


def _peptide_angle_mask(atom_mask: Tensor) -> Tensor:
    """Return phi/psi/omega validity from observed peptide N/CA/C atoms."""

    length = atom_mask.shape[0]
    n, ca, c = atom_mask[:, 0], atom_mask[:, 1], atom_mask[:, 2]
    mask = torch.zeros(length, 3, dtype=torch.bool)
    if length > 1:
        mask[1:, 0] = c[:-1] & n[1:] & ca[1:] & c[1:]
        mask[:-1, 1] = n[:-1] & ca[:-1] & c[:-1] & n[1:]
        mask[:-1, 2] = ca[:-1] & c[:-1] & n[1:] & ca[1:]
    return mask


def _insertion_rank(value: Any) -> int | None:
    code = str(value or "").strip().upper()
    if not code:
        return 0
    return ord(code) - ord("A") + 1 if len(code) == 1 and "A" <= code <= "Z" else None


def _residue_topology(record: dict[str, Any], pocket_length: int) -> tuple[Tensor, Tensor, Tensor]:
    sequence = torch.arange(pocket_length, dtype=torch.long)
    chain = torch.zeros(pocket_length, dtype=torch.long)
    keys = record.get("pocket_residue_keys")
    if keys is None:
        return sequence, chain, torch.ones(max(pocket_length - 1, 0), dtype=torch.bool)
    if not isinstance(keys, (list, tuple)) or len(keys) != pocket_length:
        raise ValueError("pocket_residue_keys must match pocket length")
    chain_ids: dict[tuple[str, str, int], int] = {}
    for index, key in enumerate(keys):
        if not isinstance(key, dict):
            raise TypeError("each pocket residue key must be a mapping")
        value = next(
            (
                key.get(name)
                for name in ("polymer_index", "label_seq_id", "auth_seq_id", "sample_index")
                if key.get(name) is not None
            ),
            None,
        )
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("pocket residue has no integer sequence index")
        identity = (
            str(key.get("auth_chain_id", "")),
            str(key.get("label_asym_id", "")),
            int(key.get("model_segment_index", key.get("segment_index", 0))),
        )
        chain_ids.setdefault(identity, len(chain_ids))
        sequence[index] = int(value)
        chain[index] = chain_ids[identity]
    adjacent = torch.zeros(max(pocket_length - 1, 0), dtype=torch.bool)
    for index, (left, right) in enumerate(zip(keys[:-1], keys[1:], strict=True)):
        if chain[index] != chain[index + 1]:
            continue
        left_polymer, right_polymer = left.get("polymer_index"), right.get("polymer_index")
        if left_polymer is not None and right_polymer is not None:
            adjacent[index] = right_polymer - left_polymer == 1
            continue
        left_label, right_label = left.get("label_seq_id"), right.get("label_seq_id")
        if isinstance(left_label, (int, np.integer)) and isinstance(right_label, (int, np.integer)):
            adjacent[index] = right_label - left_label == 1
            continue
        left_auth, right_auth = left.get("auth_seq_id"), right.get("auth_seq_id")
        left_insert = _insertion_rank(left.get("insertion_code"))
        right_insert = _insertion_rank(right.get("insertion_code"))
        if all(
            isinstance(value, (int, np.integer))
            for value in (left_auth, right_auth, left_insert, right_insert)
        ):
            adjacent[index] = (right_auth == left_auth and right_insert == left_insert + 1) or (
                right_auth == left_auth + 1 and right_insert == 0
            )
        else:
            adjacent[index] = sequence[index + 1] - sequence[index] == 1
    # Newly preprocessed records also screen observed C--N bond geometry. This
    # prevents torsions across physical breaks even with consecutive numbering.
    if "pocket_backbone_link_mask" in record:
        observed_links = torch.as_tensor(record["pocket_backbone_link_mask"])
        if observed_links.dtype != torch.bool or observed_links.shape != adjacent.shape:
            raise ValueError("pocket_backbone_link_mask must be boolean [pocket_length - 1]")
        adjacent &= observed_links
    return sequence, chain, adjacent


def _pocket_angles(
    aatype: Tensor,
    atom_xyz: Tensor,
    atom_mask: Tensor,
    adjacent: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    length = aatype.shape[0]
    n, ca, c = atom_xyz[:, 0], atom_xyz[:, 1], atom_xyz[:, 2]
    backbone, raw_mask = extract_backbone_torsions(n, ca, c)
    previous = torch.zeros(length, dtype=torch.bool)
    following = torch.zeros(length, dtype=torch.bool)
    previous[1:] = adjacent
    following[:-1] = adjacent
    observed_n, observed_ca, observed_c = atom_mask[:, 0], atom_mask[:, 1], atom_mask[:, 2]
    previous_c = torch.zeros_like(observed_c)
    next_n = torch.zeros_like(observed_n)
    next_ca = torch.zeros_like(observed_ca)
    previous_c[1:] = observed_c[:-1]
    next_n[:-1] = observed_n[1:]
    next_ca[:-1] = observed_ca[1:]
    observed = torch.stack(
        (
            previous & previous_c & observed_n & observed_ca & observed_c,
            following & observed_n & observed_ca & observed_c & next_n,
            following & observed_ca & observed_c & next_n & next_ca,
        ),
        dim=-1,
    )
    backbone_mask = raw_mask & observed

    atom14_index = _ATOM14_TO_ATOM38[aatype]
    atom14 = torch.gather(atom_xyz, 1, atom14_index[..., None].expand(-1, -1, 3))
    atom14_mask = torch.gather(atom_mask, 1, atom14_index) & _ATOM14_PRESENT[aatype]
    sidechain, sidechain_mask = extract_chi(atom14, atom14_mask, aatype)
    return backbone, backbone_mask, sidechain, sidechain_mask


@dataclass(frozen=True)
class JointV2Batch:
    sample_ids: tuple[str, ...]
    source_pdb_ids: tuple[str, ...]
    splits: tuple[str, ...]
    peptide_lengths: tuple[int, ...]
    condition: UnifiedComplexCondition
    targets: PeptideNativeTargets

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "JointV2Batch":
        return JointV2Batch(
            sample_ids=self.sample_ids,
            source_pdb_ids=self.source_pdb_ids,
            splits=self.splits,
            peptide_lengths=self.peptide_lengths,
            condition=self.condition.to(device, non_blocking=non_blocking),
            targets=self.targets.to(device, non_blocking=non_blocking),
        )


def collate_joint_v2_records(records: list[dict[str, Any]]) -> JointV2Batch:
    """Build static conditions and supervision targets in separate containers."""

    if not records:
        raise ValueError("cannot collate an empty Joint-v2 batch")
    if any("boltz_adapter" in record for record in records):
        from apexgen.joint_v2.data.boltz_npz import validate_boltz_record_contract
        for record in records:
            validate_boltz_record_contract(record)
    batch = len(records)
    pocket_lengths = [len(record["pocket_aatype"]) for record in records]
    peptide_lengths = [int(record["peptide_length"]) for record in records]
    if any(length <= 0 for length in pocket_lengths):
        raise ValueError("every Joint-v2 record must contain at least one pocket residue")
    if any(length <= 0 for length in peptide_lengths):
        raise ValueError("every Joint-v2 record must contain at least one peptide residue")
    graph_length = max(p + q for p, q in zip(pocket_lengths, peptide_lengths, strict=True))
    layout = (batch, graph_length)

    residue_mask = torch.zeros(layout, dtype=torch.bool)
    pocket_mask = torch.zeros(layout, dtype=torch.bool)
    peptide_mask = torch.zeros(layout, dtype=torch.bool)
    pocket_translation = torch.zeros(*layout, 3)
    pocket_rotation = torch.eye(3).expand(*layout, 3, 3).clone()
    aatype = torch.full(layout, UNKNOWN_AATYPE, dtype=torch.long)
    atom_xyz = torch.zeros(*layout, POCKET_ATOM_SLOTS, 3)
    atom_mask = torch.zeros(*layout, POCKET_ATOM_SLOTS, dtype=torch.bool)
    core_mask = torch.zeros(layout, dtype=torch.bool)
    sequence_index = torch.zeros(layout, dtype=torch.long)
    chain_index = torch.zeros(layout, dtype=torch.long)
    pocket_backbone = torch.zeros(*layout, 3, 2)
    pocket_backbone_mask = torch.zeros(*layout, 3, dtype=torch.bool)
    pocket_sidechain = torch.zeros(*layout, 4, 2)
    pocket_sidechain_mask = torch.zeros(*layout, 4, dtype=torch.bool)

    endpoint_translation = torch.zeros(*layout, 3)
    endpoint_rotation = torch.eye(3).expand(*layout, 3, 3).clone()
    endpoint_aatype = torch.full(layout, UNKNOWN_AATYPE, dtype=torch.long)
    target_atom14 = torch.zeros(*layout, 14, 3)
    target_atom14_mask = torch.zeros(*layout, 14, dtype=torch.bool)
    backbone_xyz = torch.zeros(*layout, 3, 3)
    backbone_atom_mask = torch.zeros(*layout, 3, dtype=torch.bool)
    backbone_angles = torch.zeros(*layout, BACKBONE_ANGLE_SLOTS, 2)
    backbone_angle_mask = torch.zeros(*layout, BACKBONE_ANGLE_SLOTS, dtype=torch.bool)
    sidechain_angles = torch.zeros(*layout, SIDECHAIN_ANGLE_SLOTS, 2)
    sidechain_angle_mask = torch.zeros(*layout, SIDECHAIN_ANGLE_SLOTS, dtype=torch.bool)

    for row, (record, pocket_length, peptide_length) in enumerate(
        zip(records, pocket_lengths, peptide_lengths, strict=True)
    ):
        target = record.get("joint_v2_target")
        if not isinstance(target, dict):
            raise ValueError("record is missing its Joint-v2 geometry target")
        cached = read_static_features(record)
        p = slice(0, pocket_length)
        q = slice(pocket_length, pocket_length + peptide_length)

        p_aatype_array = np.asarray(record["pocket_aatype"])
        if not np.issubdtype(p_aatype_array.dtype, np.integer):
            raise TypeError("pocket aatype must have an integer dtype")
        p_aatype = torch.as_tensor(p_aatype_array, dtype=torch.long)
        p_xyz = torch.as_tensor(np.asarray(record["pocket_atom_xyz"]), dtype=torch.float32)
        p_atom_mask = torch.as_tensor(np.asarray(record["pocket_atom_mask"]), dtype=torch.bool)
        p_rotation = torch.as_tensor(
            np.asarray(record["pocket_residue_rotation"]), dtype=torch.float32
        )
        p_translation = torch.as_tensor(
            np.asarray(record["pocket_residue_translation"]), dtype=torch.float32
        )
        if p_xyz.shape != (pocket_length, POCKET_ATOM_SLOTS, 3):
            raise ValueError("Joint-v2 requires 38-slot pocket atom records")
        p_xyz = torch.where(p_atom_mask[..., None], p_xyz, 0.0)
        if cached is None:
            p_sequence, p_chain, p_adjacent = _residue_topology(record, pocket_length)
            p_bb, p_bb_mask, p_sc, p_sc_mask = _pocket_angles(p_aatype, p_xyz, p_atom_mask, p_adjacent)
            p_bb_sincos, p_sc_sincos = _angle_sin_cos(p_bb, p_bb_mask), _angle_sin_cos(p_sc, p_sc_mask)
        else:
            pc = {name: torch.as_tensor(value) for name, value in cached["pocket"].items()}
            p_sequence, p_chain = pc["sequence_index"], pc["chain_index"]
            p_bb_sincos, p_bb_mask = pc["backbone_angles_sin_cos"], pc["backbone_angle_mask"]
            p_sc_sincos, p_sc_mask = pc["sidechain_angles_sin_cos"], pc["sidechain_angle_mask"]

        q_aatype_array = np.asarray(target["aatype"])
        if not np.issubdtype(q_aatype_array.dtype, np.integer):
            raise TypeError("peptide target aatype must have an integer dtype")
        q_aatype = torch.as_tensor(q_aatype_array, dtype=torch.long)
        q_atom14 = torch.as_tensor(np.asarray(target["experimental_atom14"]), dtype=torch.float32)
        q_atom14_mask = torch.as_tensor(
            np.asarray(target["experimental_atom14_mask"]), dtype=torch.bool
        )
        if (
            q_aatype.shape != (peptide_length,)
            or q_atom14.shape != (peptide_length, 14, 3)
            or q_atom14_mask.shape != (peptide_length, 14)
        ):
            raise ValueError("peptide sequence/geometry target shape differs from record length")
        if bool(((q_aatype < 0) | (q_aatype >= UNKNOWN_AATYPE)).any()):
            raise ValueError("peptide target aatype must lie in [0, 20)")
        # Frozen v1 sidecars also contain whole-chain idealized frames. They are
        # migration artifacts, not v2 labels. Both native and generated teachers
        # derive their frames and torsions from their supplied observed atoms.
        q_backbone_mask = q_atom14_mask[:, :3]
        if cached is None:
            q_frames = observed_backbone_frames(q_atom14[:, :3], q_backbone_mask)
            q_translation, q_rotation = q_frames.translation, q_frames.rotation
            q_angles, _ = extract_backbone_torsions(q_atom14[:, 0], q_atom14[:, 1], q_atom14[:, 2])
            q_angle_mask = _peptide_angle_mask(q_backbone_mask)
            q_sidechain, q_sidechain_mask = extract_chi(q_atom14, q_atom14_mask, q_aatype)
            q_bb_sincos = _angle_sin_cos(q_angles, q_angle_mask)
            q_sc_sincos = _angle_sin_cos(q_sidechain, q_sidechain_mask)
        else:
            qc = {name: torch.as_tensor(value) for name, value in cached["peptide"].items()}
            q_translation, q_rotation = qc["translation"], qc["rotation"]
            q_bb_sincos, q_angle_mask = qc["backbone_angles_sin_cos"], qc["backbone_angle_mask"]
            q_sc_sincos, q_sidechain_mask = qc["sidechain_angles_sin_cos"], qc["sidechain_angle_mask"]
        residue_mask[row, : pocket_length + peptide_length] = True
        pocket_mask[row, p] = True
        peptide_mask[row, q] = True
        pocket_translation[row, p] = p_translation
        pocket_rotation[row, p] = p_rotation
        aatype[row, p] = p_aatype
        atom_xyz[row, p] = p_xyz
        atom_mask[row, p] = p_atom_mask
        core_mask[row, p] = torch.as_tensor(record["pocket_core_mask"], dtype=torch.bool)
        sequence_index[row, p] = p_sequence
        sequence_index[row, q] = torch.arange(peptide_length)
        chain_index[row, p] = p_chain
        chain_index[row, q] = int(p_chain.max()) + 1
        pocket_backbone[row, p] = p_bb_sincos
        pocket_backbone_mask[row, p] = p_bb_mask
        pocket_sidechain[row, p] = p_sc_sincos
        pocket_sidechain_mask[row, p] = p_sc_mask

        endpoint_translation[row, q] = q_translation
        endpoint_rotation[row, q] = q_rotation
        endpoint_aatype[row, q] = q_aatype
        target_atom14[row, q] = torch.where(q_atom14_mask[..., None], q_atom14, 0.0)
        target_atom14_mask[row, q] = q_atom14_mask
        backbone_xyz[row, q] = torch.where(q_backbone_mask[..., None], q_atom14[:, :3], 0.0)
        backbone_atom_mask[row, q] = q_backbone_mask
        backbone_angles[row, q] = q_bb_sincos
        backbone_angle_mask[row, q] = q_angle_mask
        sidechain_angles[row, q] = q_sc_sincos
        sidechain_angle_mask[row, q] = q_sidechain_mask

    condition = UnifiedComplexCondition(
        residue_mask=residue_mask,
        pocket_mask=pocket_mask,
        peptide_mask=peptide_mask,
        pocket_translation=pocket_translation,
        pocket_rotation=pocket_rotation,
        aatype=aatype,
        pocket_atom_xyz=atom_xyz,
        pocket_atom_mask=atom_mask,
        pocket_core_mask=core_mask,
        sequence_index=sequence_index,
        chain_index=chain_index,
        pocket_backbone_angles_sin_cos=pocket_backbone,
        pocket_backbone_angle_mask=pocket_backbone_mask,
        pocket_sidechain_angles_sin_cos=pocket_sidechain,
        pocket_sidechain_angle_mask=pocket_sidechain_mask,
    )
    targets = PeptideNativeTargets(
        endpoint_aatype=endpoint_aatype,
        endpoint_translation=endpoint_translation,
        endpoint_rotation=endpoint_rotation,
        atom14_xyz=target_atom14,
        atom14_mask=target_atom14_mask,
        backbone_xyz=backbone_xyz,
        backbone_atom_mask=backbone_atom_mask,
        backbone_angles_sin_cos=backbone_angles,
        backbone_angle_mask=backbone_angle_mask,
        sidechain_angles_sin_cos=sidechain_angles,
        sidechain_angle_mask=sidechain_angle_mask,
    )
    condition.validate_invariants()
    targets.validate_for(condition)
    return JointV2Batch(
        sample_ids=tuple(str(record["sample_id"]) for record in records),
        source_pdb_ids=tuple(str(record["source_pdb_id"]) for record in records),
        splits=tuple(str(record["split"]) for record in records),
        peptide_lengths=tuple(peptide_lengths),
        condition=condition,
        targets=targets,
    )
