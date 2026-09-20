"""Deterministic residue-level alternate-conformer selection."""

from __future__ import annotations

from dataclasses import dataclass

from apexgen.joint_v2.data.preprocessing.structure import AtomRecord, ResidueRecord


BACKBONE_NAMES = frozenset({"N", "CA", "C"})


@dataclass(frozen=True)
class AltlocCandidate:
    altloc: str
    backbone_atoms_present: int
    occupancy_sum: float


@dataclass(frozen=True)
class AltlocDecision:
    selected: str
    candidates: tuple[AltlocCandidate, ...]
    tied: bool


def select_residue_altloc(residue: ResidueRecord) -> tuple[ResidueRecord, AltlocDecision]:
    """Keep shared atoms plus one internally consistent non-blank conformer."""

    shared = tuple(atom for atom in residue.atoms if not atom.altloc)
    altlocs = sorted({atom.altloc for atom in residue.atoms if atom.altloc})
    if not altlocs:
        _reject_duplicate_names(shared)
        return residue, AltlocDecision("", (), False)

    candidates = []
    for altloc in altlocs:
        conformer = tuple(atom for atom in residue.atoms if atom.altloc == altloc)
        names = {atom.name for atom in (*shared, *conformer)}
        candidates.append(
            AltlocCandidate(
                altloc=altloc,
                backbone_atoms_present=len(names & BACKBONE_NAMES),
                occupancy_sum=sum(atom.occupancy for atom in conformer),
            )
        )
    best_score = max((item.backbone_atoms_present, item.occupancy_sum) for item in candidates)
    tied_candidates = [
        item
        for item in candidates
        if (item.backbone_atoms_present, item.occupancy_sum) == best_score
    ]
    selected = min(tied_candidates, key=lambda item: (item.altloc != "A", item.altloc)).altloc
    selected_atoms = tuple(
        atom for atom in residue.atoms if not atom.altloc or atom.altloc == selected
    )
    _reject_duplicate_names(selected_atoms)
    normalized_atoms = tuple(
        AtomRecord(
            name=atom.name,
            element=atom.element,
            altloc="",
            occupancy=atom.occupancy,
            b_iso=atom.b_iso,
            serial=atom.serial,
            xyz=atom.xyz,
        )
        for atom in selected_atoms
    )
    return (
        ResidueRecord(residue.key, residue.name, residue.entity_type, normalized_atoms),
        AltlocDecision(selected, tuple(candidates), len(tied_candidates) > 1),
    )


def _reject_duplicate_names(atoms: tuple[AtomRecord, ...]) -> None:
    names = [atom.name for atom in atoms]
    if len(names) != len(set(names)):
        raise ValueError("selected residue conformer contains duplicate atom names")
