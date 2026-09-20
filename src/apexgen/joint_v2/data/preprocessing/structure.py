"""Deterministic Gemmi parsing with explicit chain roles and reversible identities."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import gemmi


@dataclass(frozen=True, order=True)
class ResidueKey:
    auth_chain_id: str
    auth_seq_id: int
    insertion_code: str
    label_asym_id: str
    label_seq_id: int | None
    sample_index: int
    segment_index: int = 0
    polymer_index: int | None = None
    index_source: str = "unassigned"
    model_segment_index: int = 0


@dataclass(frozen=True)
class AtomRecord:
    name: str
    element: str
    altloc: str
    occupancy: float
    b_iso: float
    serial: int
    xyz: tuple[float, float, float]


@dataclass(frozen=True)
class ResidueRecord:
    key: ResidueKey
    name: str
    entity_type: str
    atoms: tuple[AtomRecord, ...]


@dataclass(frozen=True)
class ChainRecord:
    chain_id: str
    role: str
    residues: tuple[ResidueRecord, ...]


@dataclass(frozen=True)
class ParsedStructure:
    source_path: str
    source_sha256: str
    source_format: str
    assembly_id: str | None
    selected_model_index: int
    selected_model_name: str
    total_model_count: int
    chains: tuple[ChainRecord, ...]
    sequence_metadata: dict
    context: dict


@dataclass(frozen=True)
class PepBenchIndexRecord:
    sample_id: str
    receptor_chain_id: str
    peptide_chain_id: str
    flag: str


def parse_pepbench_index_line(line: str) -> PepBenchIndexRecord:
    """Parse one four-column PepBench index row without interpreting its flag."""

    fields = line.split()
    if len(fields) != 4 or any(not field for field in fields):
        raise ValueError("PepBench index row must contain exactly four non-empty fields")
    sample_id, receptor_chain_id, peptide_chain_id, flag = fields
    if receptor_chain_id == peptide_chain_id:
        raise ValueError("receptor and peptide chain IDs must differ")
    return PepBenchIndexRecord(sample_id, receptor_chain_id, peptide_chain_id, flag)


def _normalized_character(value: str) -> str:
    return "" if value in {"\x00", " ", ".", "?"} else value


def _source_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        return "mmcif"
    if suffix in {".pdb", ".ent"}:
        return "pdb"
    raise ValueError(f"unsupported structure suffix: {path.suffix}")


def parse_structure(
    path: str | Path,
    *,
    receptor_chain_ids: tuple[str, ...],
    peptide_chain_id: str,
    model_index: int = 0,
    assembly_id: str | None = None,
    environment_radius: float = 5.0,
) -> ParsedStructure:
    """Parse one selected model; chain roles must come from an external index or user input."""

    path = Path(path)
    if not receptor_chain_ids or len(set(receptor_chain_ids)) != len(receptor_chain_ids):
        raise ValueError("receptor_chain_ids must be a non-empty unique tuple")
    if peptide_chain_id in receptor_chain_ids:
        raise ValueError("peptide chain must not also be a receptor chain")
    raw_bytes = path.read_bytes()
    structure = gemmi.read_structure(str(path), merge_chain_parts=False)
    if assembly_id is not None:
        raise ValueError(
            "assembly expansion is not implemented; supply explicitly prepared assembly coordinates"
        )
    if not 0 <= model_index < len(structure):
        raise ValueError(f"model_index {model_index} is outside [0, {len(structure)})")
    model = structure[model_index]
    available = {chain.name for chain in model}
    requested = set(receptor_chain_ids) | {peptide_chain_id}
    missing = requested - available
    if missing:
        raise ValueError(f"requested chains are missing from selected model: {sorted(missing)}")
    if any(sum(chain.name == name for chain in model) != 1 for name in requested):
        raise ValueError("requested chain IDs must identify unique chains in the selected model")

    from apexgen.joint_v2.data.preprocessing.quality import sequence_metadata

    metadata = sequence_metadata(path, structure, model_index, sorted(requested))
    from apexgen.joint_v2.data.preprocessing.environment import inspect_structure_context

    context = inspect_structure_context(
        path, structure, model_index, receptor_chain_ids, peptide_chain_id, environment_radius
    )

    chains: list[ChainRecord] = []
    for chain in model:
        if chain.name not in requested:
            continue
        role = "peptide" if chain.name == peptide_chain_id else "receptor"
        residues: list[ResidueRecord] = []
        seen_keys = set()
        segment_index = 0
        boundaries = {
            (r["auth_seq_id"], r["insertion_code"])
            for r in metadata[chain.name]["ter_after_residues"]
        }
        previous_identity = None
        for sample_index, residue in enumerate(chain):
            if previous_identity in boundaries:
                segment_index += 1
            identity = (int(residue.seqid.num), _normalized_character(residue.seqid.icode))
            if identity in seen_keys:
                raise ValueError(
                    f"duplicate or ambiguous residue identity: {chain.name} {identity}"
                )
            seen_keys.add(identity)
            previous_identity = identity
            atoms: list[AtomRecord] = []
            for atom in residue:
                xyz = (float(atom.pos.x), float(atom.pos.y), float(atom.pos.z))
                if not all(math.isfinite(value) for value in xyz):
                    raise ValueError("atom coordinates must be finite")
                atoms.append(
                    AtomRecord(
                        name=atom.name,
                        element=atom.element.name,
                        altloc=_normalized_character(atom.altloc),
                        occupancy=float(atom.occ),
                        b_iso=float(atom.b_iso),
                        serial=int(atom.serial),
                        xyz=xyz,
                    )
                )
            label_seq = None if residue.label_seq is None else int(residue.label_seq)
            key = ResidueKey(
                auth_chain_id=chain.name,
                auth_seq_id=int(residue.seqid.num),
                insertion_code=_normalized_character(residue.seqid.icode),
                label_asym_id=residue.subchain or chain.name,
                label_seq_id=label_seq,
                sample_index=sample_index,
                segment_index=segment_index,
            )
            residues.append(
                ResidueRecord(
                    key=key,
                    name=residue.name,
                    entity_type=residue.entity_type.name,
                    atoms=tuple(atoms),
                )
            )
        chains.append(ChainRecord(chain.name, role, tuple(residues)))

    role_order = {
        chain_id: index for index, chain_id in enumerate((*receptor_chain_ids, peptide_chain_id))
    }
    chains.sort(key=lambda chain: role_order[chain.chain_id])
    return ParsedStructure(
        source_path=str(path),
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        source_format=_source_format(path),
        assembly_id=assembly_id,
        selected_model_index=model_index,
        selected_model_name=model.name,
        total_model_count=len(structure),
        chains=tuple(chains),
        sequence_metadata=metadata,
        context=context,
    )
