"""Build a native record from the full raw complex at a configurable radius."""

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path

import gemmi
import numpy as np
import torch

from apexgen.shared.storage.features import virtual_cb
from apexgen.joint_v2.data.dataset import NATIVE_TARGET_SCHEMA
from apexgen.shared.storage.store import RECORD_SCHEMA_VERSION
from apexgen.shared.geometry.joint_residue_constants import AA1_ORDER, AA3_TO_INDEX, ATOM14_NAMES
from apexgen.shared.geometry.torsion import extract_backbone_torsions
from apexgen.joint_v2.data.batch import POCKET_ATOM_NAMES, collate_joint_v2_records
from apexgen.joint_v2.contracts.contract import JOINT_V2_CONTRACT_SHA256
from apexgen.joint_v2.data.native_targets import observed_backbone_frames
from apexgen.joint_v2.data.preprocessing.chemistry import (
    RECEPTOR_RESIDUE_MAPPINGS,
    STANDARD_AMINO_ACIDS,
    normalize_residue_chemistry,
)
from apexgen.joint_v2.data.preprocessing.structure import parse_structure
from apexgen.joint_v2.data.preprocessing.quality import (
    backbone_links,
    observed_conformer,
    validate_peptide_sequence,
)
from apexgen.joint_v2.data.preprocessing.stereochemistry import (
    GeometryPolicy,
    validate_elements,
    validate_residue_geometry,
    validate_link_angles,
)
from apexgen.joint_v2.data.preprocessing.topology import assign_polymer_indices


@dataclass(frozen=True)
class PocketParameters:
    core_cutoff_angstrom: float = 5.0
    context_radius_angstrom: float = 11.0
    environment_radius_angstrom: float = 5.0
    geometry: GeometryPolicy = field(default_factory=GeometryPolicy)

    def __post_init__(self):
        if not isinstance(self.geometry, GeometryPolicy):
            raise TypeError("geometry must be GeometryPolicy")
        for name in (
            "core_cutoff_angstrom",
            "context_radius_angstrom",
            "environment_radius_angstrom",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


def _atom(residue, name):
    atoms = [a.xyz for a in residue.atoms if a.name == name]
    if len(atoms) != 1:
        raise ValueError(f"{residue.key}: requires one observed {name}")
    return torch.tensor(atoms[0], dtype=torch.float64)


def _cb(residue):
    if residue.name == "GLY":
        return _atom(residue, "CA")
    if any(a.name == "CB" for a in residue.atoms):
        return _atom(residue, "CB")
    return virtual_cb(_atom(residue, "N"), _atom(residue, "CA"), _atom(residue, "C"))


def _heavy(residue):
    xyz = [a.xyz for a in residue.atoms if a.element.upper() not in {"H", "D"}]
    if not xyz:
        raise ValueError(f"{residue.key}: no observed heavy atoms")
    return torch.tensor(xyz, dtype=torch.float64)


def _backbone(residues):
    return torch.stack([torch.stack([_atom(r, a) for a in ("N", "CA", "C")]) for r in residues])


def preprocess_complex(
    path: str | Path,
    *,
    sample_id: str,
    receptor_chain_id: str,
    peptide_chain_id: str,
    split: str,
    source_pdb_id: str,
    parameters: PocketParameters | None = None,
    model_index: int = 0,
    expected_peptide_sequence: str | None = None,
):
    """Extract observed frames; never idealize/refit the native peptide chain.

    Core: any receptor/native-peptide heavy-atom distance <= core cutoff.
    Context: CB distance to any core CB <= radius, always retaining all core.
    Changing context radius leaves the core centroid and peptide labels fixed.
    """
    parameters = parameters or PocketParameters()
    if not all(isinstance(s, str) and s for s in (sample_id, split, source_pdb_id)):
        raise ValueError("sample_id, split and source_pdb_id must be nonempty strings")
    parsed = parse_structure(
        Path(path).resolve(),
        receptor_chain_ids=(receptor_chain_id,),
        peptide_chain_id=peptide_chain_id,
        model_index=model_index,
        environment_radius=parameters.environment_radius_angstrom,
    )
    conformers, ignored_peptide = [], []
    peptide = []
    allowed = STANDARD_AMINO_ACIDS | frozenset(RECEPTOR_RESIDUE_MAPPINGS)
    for r in parsed.chains[1].residues:
        if (
            r.name not in allowed
            and not gemmi.find_tabulated_residue(r.name).is_amino_acid()
            and r.entity_type != "Polymer"
        ):
            ignored_peptide.append({"key": asdict(r.key), "name": r.name})
            continue
        observed, decision = observed_conformer(r)
        validate_elements(observed)
        conformers.append(decision)
        peptide.append(normalize_residue_chemistry(observed, role="peptide")[0])
    if not peptide:
        raise ValueError("empty peptide chain")
    peptide_bb = _backbone(peptide)
    observed_backbone_frames(peptide_bb, torch.ones_like(peptide_bb[..., 0], dtype=torch.bool))
    peptide_metadata = parsed.sequence_metadata[peptide_chain_id]
    # Check declared/expected completeness before alignment to produce a direct
    # missing-label error rather than inventing a shorter peptide.
    if peptide_metadata["unobserved_residues"] or (
        peptide_metadata["declared_sequence"]
        and len(peptide_metadata["declared_sequence"]) != len(peptide)
    ):
        validate_peptide_sequence(peptide, peptide_metadata, parameters.geometry)
    if expected_peptide_sequence is not None:
        if (
            not isinstance(expected_peptide_sequence, str)
            or not expected_peptide_sequence
            or any(a not in AA1_ORDER for a in expected_peptide_sequence)
        ):
            raise ValueError(
                "expected_peptide_sequence must be a nonempty uppercase canonical sequence"
            )
        observed_sequence = "".join(AA1_ORDER[AA3_TO_INDEX[r.name]] for r in peptide)
        if expected_peptide_sequence != observed_sequence:
            raise ValueError("peptide observed sequence differs from expected_peptide_sequence")
    peptide = assign_polymer_indices(peptide, peptide_metadata, parameters.geometry)
    sequence_status = validate_peptide_sequence(peptide, peptide_metadata, parameters.geometry)
    if expected_peptide_sequence is not None:
        sequence_status = "expected_sequence_matched"
    chirality = [
        dict(
            role="peptide",
            key=asdict(r.key),
            status=validate_residue_geometry(r, parameters.geometry),
        )
        for r in peptide
    ]
    validate_link_angles(
        peptide, backbone_links(peptide, parameters.geometry)[0], parameters.geometry
    )
    peptide_atoms = torch.cat([_heavy(r) for r in peptide])
    receptor_raw, ignored = [], []
    for r in parsed.chains[0].residues:
        # Explicitly omit water/ions/nonprotein ligands from the receptor condition.
        if (
            r.name not in allowed
            and not gemmi.find_tabulated_residue(r.name).is_amino_acid()
            and r.entity_type != "Polymer"
        ):
            ignored.append({"key": asdict(r.key), "name": r.name})
            continue
        observed, decision = observed_conformer(r)
        conformers.append(decision)
        # A residue with no observed heavy atoms has no usable location.
        if not any(a.element.upper() not in {"H", "D"} for a in observed.atoms):
            raise ValueError(f"{r.key}: receptor residue has no observed heavy atoms")
        receptor_raw.append(observed)
    if not receptor_raw:
        raise ValueError("empty receptor protein chain")
    receptor_raw = assign_polymer_indices(
        receptor_raw, parsed.sequence_metadata[receptor_chain_id], parameters.geometry
    )
    core = [
        bool((torch.cdist(_heavy(r), peptide_atoms) <= parameters.core_cutoff_angstrom).any())
        for r in receptor_raw
    ]
    if not any(core):
        raise ValueError("no receptor contact core at the requested cutoff")
    core_residues = [
        normalize_residue_chemistry(r, role="receptor")[0]
        for r, keep in zip(receptor_raw, core)
        if keep
    ]
    core_cb = torch.stack([_cb(r) for r in core_residues])
    origin = torch.stack([_atom(r, "CA") for r in core_residues]).mean(0)
    receptor, original_names, distances, is_core = [], [], [], []
    chemistry, missing_atoms = [], []
    for r, keep_core in zip(receptor_raw, core):
        try:
            distance = float(torch.cdist(_cb(r)[None], core_cb).min())
        except ValueError as exc:
            # Its location relative to the crop cannot be established reliably.
            raise ValueError(f"{r.key}: cannot locate receptor context residue: {exc}") from exc
        if not keep_core and distance > parameters.context_radius_angstrom:
            continue
        normalized, decision = normalize_residue_chemistry(r, role="receptor")
        validate_elements(r)
        if decision.mapping:
            chemistry.append(dict(key=asdict(r.key), **asdict(decision)))
        receptor.append(normalized)
        original_names.append(r.name)
        distances.append(distance)
        is_core.append(keep_core)
    backbone = _backbone(receptor) - origin
    frames = observed_backbone_frames(backbone, torch.ones_like(backbone[..., 0], dtype=torch.bool))
    chirality += [
        dict(
            role="receptor",
            key=asdict(r.key),
            status=validate_residue_geometry(r, parameters.geometry),
        )
        for r in receptor
    ]
    links, pocket_breaks = backbone_links(receptor, parameters.geometry)
    _, receptor_breaks = backbone_links(receptor_raw, parameters.geometry)
    validate_link_angles(receptor, links, parameters.geometry)
    atoms = np.zeros((len(receptor), len(POCKET_ATOM_NAMES), 3), dtype=np.float32)
    masks = np.zeros(atoms.shape[:-1], dtype=bool)
    atom_lookup = {name: i for i, name in enumerate(POCKET_ATOM_NAMES)}
    for i, r in enumerate(receptor):
        for atom in r.atoms:
            if atom.name not in atom_lookup:
                raise ValueError(f"unsupported receptor atom {atom.name}")
            j = atom_lookup[atom.name]
            atoms[i, j] = np.asarray(atom.xyz) - origin.numpy()
            masks[i, j] = True
    aa = np.array([AA3_TO_INDEX[r.name] for r in peptide], dtype=np.int64)
    atom14 = np.zeros((len(peptide), 14, 3), dtype=np.float32)
    atom14_mask = np.zeros(atom14.shape[:-1], dtype=bool)
    for i, r in enumerate(peptide):
        observed = {a.name: a.xyz for a in r.atoms}
        for j, name in enumerate(ATOM14_NAMES[aa[i]]):
            if name and name in observed:
                atom14[i, j] = np.asarray(observed[name]) - origin.numpy()
                atom14_mask[i, j] = True
    for role, residues in [("receptor", receptor), ("peptide", peptide)]:
        for r in residues:
            present = {a.name for a in r.atoms}
            absent = [a for a in ATOM14_NAMES[AA3_TO_INDEX[r.name]] if a and a not in present]
            if absent:
                missing_atoms.append(dict(role=role, key=asdict(r.key), atoms=absent))
    warnings = []
    if sequence_status == "unknown_no_declared_sequence":
        warnings.append("peptide_terminal_completeness_unknown")
    receptor_metadata = parsed.sequence_metadata[receptor_chain_id]
    if receptor_metadata["unobserved_residues"] or (
        receptor_metadata["declared_sequence"]
        and len(receptor_metadata["declared_sequence"]) != len(receptor_raw)
    ):
        warnings.append("receptor_has_unobserved_or_sequence_mismatched_residues")
    if receptor_breaks:
        warnings.append("receptor_chain_breaks_or_numbering_gaps")
    if chemistry:
        warnings.append("receptor_chemistry_mapped_to_canonical_approximation")
    if missing_atoms:
        warnings.append("missing_observed_atoms_masked")
    if ignored or ignored_peptide:
        warnings.append("nonprotein_residues_omitted")
    if any(d["zero_occupancy_atoms"] for d in conformers):
        warnings.append("zero_occupancy_atoms_omitted")
    if any(r.key.index_source == "observed_connected_fragment" for r in [*receptor, *peptide]):
        warnings.append("polymer_positions_only_known_within_observed_fragments")
    if any(c["status"] == "unknown_missing_CB" for c in chirality):
        warnings.append("chirality_unknown_missing_CB")
    if parsed.context.get("nearby_excluded_residues"):
        warnings.append("nearby_full_structure_environment_not_in_condition")
    quality = dict(
        warnings=warnings,
        peptide_sequence_status=sequence_status,
        expected_peptide_sequence=expected_peptide_sequence,
        sequence_metadata=parsed.sequence_metadata,
        receptor_observed_breaks=receptor_breaks,
        pocket_backbone_breaks=pocket_breaks,
        receptor_chemistry_mappings=chemistry,
        missing_atoms=missing_atoms,
        conformer_decisions=[
            d for d in conformers if d["zero_occupancy_atoms"] or d["altloc"]["candidates"]
        ],
        ignored_peptide_nonprotein_residues=ignored_peptide,
        chirality=chirality,
        full_structure_context=parsed.context,
    )
    native_bb = torch.from_numpy(atom14[:, :3])
    native_frames = observed_backbone_frames(native_bb, torch.from_numpy(atom14_mask[:, :3]))
    torsions, _ = extract_backbone_torsions(native_bb[:, 0], native_bb[:, 1], native_bb[:, 2])
    # These observed peptide labels are embedded in the complete complex record.
    target = dict(
        schema_version=NATIVE_TARGET_SCHEMA,
        sample_id=sample_id,
        peptide_length=len(peptide),
        aatype=aa,
        translation=native_frames.translation.numpy(),
        rotation=native_frames.rotation.numpy(),
        backbone_torsion=torsions.numpy(),
        experimental_atom14=atom14,
        experimental_atom14_mask=atom14_mask,
        supervision_source="observed_pdb",
        joint_v2_contract_sha256=JOINT_V2_CONTRACT_SHA256,
    )
    record = dict(
        schema_version=RECORD_SCHEMA_VERSION,
        sample_id=sample_id,
        source_pdb_id=source_pdb_id,
        split=split,
        raw_path=parsed.source_path,
        raw_file_sha256=parsed.source_sha256,
        receptor_chain_id=receptor_chain_id,
        peptide_chain_id=peptide_chain_id,
        peptide_length=len(peptide),
        site_origin=origin.float().numpy(),
        pocket_residue_keys=[asdict(r.key) for r in receptor],
        peptide_residue_keys=[asdict(r.key) for r in peptide],
        pocket_original_residue_names=original_names,
        pocket_normalized_residue_names=[r.name for r in receptor],
        pocket_aatype=np.array([AA3_TO_INDEX[r.name] for r in receptor], dtype=np.int16),
        pocket_atom_xyz=atoms,
        pocket_atom_mask=masks,
        pocket_residue_translation=frames.translation.float().numpy(),
        pocket_residue_rotation=frames.rotation.float().numpy(),
        pocket_core_mask=np.array(is_core, dtype=bool),
        pocket_backbone_link_mask=np.array(links, dtype=bool),
        pocket_cb_to_core_distance=np.array(distances, dtype=np.float32),
        preprocessing_parameters=asdict(parameters),
        selected_model_index=model_index,
        ignored_receptor_nonprotein_residues=ignored,
        structure_quality=quality,
    )
    collate_joint_v2_records([{**record, "joint_v2_target": target}])
    return record, target
