"""Explicit receptor chemistry normalization for native Joint-v2 records."""

from __future__ import annotations

from dataclasses import dataclass

from apexgen.joint_v2.data.preprocessing.structure import AtomRecord, ResidueRecord


STANDARD_AMINO_ACIDS = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }
)
RECEPTOR_RESIDUE_MAPPINGS = {
    "ABA": "ALA",
    "ALY": "LYS",
    "CME": "CYS",
    "CSO": "CYS",
    "CSD": "CYS",
    "CSU": "CYS",
    "HS8": "HIS",
    "LLP": "LYS",
    "MSE": "MET",
    "PCA": "GLU",
    "PTR": "TYR",
    "SEP": "SER",
    "SNN": "ASN",
    "TPO": "THR",
    "YCM": "CYS",
}
MAPPED_RESIDUE_ATOMS = {
    "ALA": frozenset({"N", "CA", "C", "O", "OXT", "CB"}),
    "ASN": frozenset({"N", "CA", "C", "O", "OXT", "CB", "CG", "OD1", "ND2"}),
    "LYS": frozenset({"N", "CA", "C", "O", "OXT", "CB", "CG", "CD", "CE", "NZ"}),
    "CYS": frozenset({"N", "CA", "C", "O", "OXT", "CB", "SG"}),
    "GLU": frozenset({"N", "CA", "C", "O", "OXT", "CB", "CG", "CD", "OE1", "OE2"}),
    "HIS": frozenset({"N", "CA", "C", "O", "OXT", "CB", "CG", "ND1", "CD2", "CE1", "NE2"}),
    "MET": frozenset({"N", "CA", "C", "O", "OXT", "CB", "CG", "SD", "CE"}),
    "SER": frozenset({"N", "CA", "C", "O", "OXT", "CB", "OG"}),
    "THR": frozenset({"N", "CA", "C", "O", "OXT", "CB", "OG1", "CG2"}),
    "TYR": frozenset(
        {"N", "CA", "C", "O", "OXT", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"}
    ),
}


class ChemistryExclusion(ValueError):
    """A structure is outside the supported Joint-v2 chemistry scope."""


@dataclass(frozen=True)
class ChemistryDecision:
    original_residue_name: str
    normalized_residue_name: str
    mapping: str | None
    removed_hydrogen_count: int


def normalize_residue_chemistry(
    residue: ResidueRecord, *, role: str
) -> tuple[ResidueRecord, ChemistryDecision]:
    """Normalize an allowed residue while preserving an explicit audit decision."""

    if role not in {"receptor", "peptide"}:
        raise ValueError("role must be receptor or peptide")
    original = residue.name.upper()
    mapping = None
    if original in STANDARD_AMINO_ACIDS:
        normalized_name = original
    elif role == "receptor" and original in RECEPTOR_RESIDUE_MAPPINGS:
        normalized_name = RECEPTOR_RESIDUE_MAPPINGS[original]
        mapping = f"{original}->{normalized_name}"
    else:
        raise ChemistryExclusion(f"unsupported_{role}_residue:{original}")

    atoms = []
    removed_hydrogens = 0
    for atom in residue.atoms:
        if atom.element.upper() in {"H", "D"}:
            removed_hydrogens += 1
            continue
        if (
            mapping is not None
            and atom.name not in MAPPED_RESIDUE_ATOMS[normalized_name]
            and not (original == "MSE" and atom.name == "SE")
        ):
            continue
        if original == "MSE" and atom.name == "SE":
            atom = AtomRecord(
                "SD", "S", atom.altloc, atom.occupancy, atom.b_iso, atom.serial, atom.xyz
            )
        atoms.append(atom)
    normalized = ResidueRecord(residue.key, normalized_name, residue.entity_type, tuple(atoms))
    return normalized, ChemistryDecision(original, normalized_name, mapping, removed_hydrogens)
