"""Observed-atom chemical QC using the repository's pinned OpenFold tables."""

from dataclasses import asdict, dataclass
from functools import lru_cache
import math

import numpy as np

from apexgen.shared.geometry import _openfold_joint_constants as constants
from apexgen.shared.geometry.joint_residue_constants import AA3_TO_INDEX, ATOM14_NAMES


@dataclass(frozen=True)
class GeometryPolicy:
    # Deliberately broad input screening, not refinement targets.
    bond_stddevs: float = 12.0
    angle_stddevs: float = 12.0
    bond_tolerance_floor_angstrom: float = 0.15
    angle_tolerance_floor_degrees: float = 15.0
    min_normalized_chiral_volume: float = 0.1
    min_cn_angstrom: float = 1.0
    max_cn_angstrom: float = 2.0
    link_angle_tolerance_degrees: float = 35.0

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if self.min_cn_angstrom >= self.max_cn_angstrom:
            raise ValueError("C-N lower bound must be below upper bound")
        if self.min_normalized_chiral_volume >= 1:
            raise ValueError("chiral volume threshold must be below 1")


def validate_elements(residue):
    """Check protein atom names before chemistry mapping or H removal."""
    for atom in residue.atoms:
        name = atom.name.lstrip("0123456789").upper()
        expected = "SE" if name == "SE" and residue.name == "MSE" else name[:1]
        actual = atom.element.upper()
        if expected in {"H", "D"}:
            valid = actual in {"H", "D"}
        else:
            valid = expected in {"C", "N", "O", "S", "P", "SE"} and actual == expected
        if not valid:
            raise ValueError(
                f"atom_element_mismatch: {residue.key} {atom.name} element={atom.element}"
            )


@lru_cache(maxsize=1)
def _tables():
    return constants.load_stereo_chemical_props()


def _angle(a, b, c):
    left, right = a - b, c - b
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator < 1e-8:
        raise ValueError("degenerate bond angle")
    return math.acos(float(np.clip(np.dot(left, right) / denominator, -1, 1)))


def validate_residue_geometry(residue, policy):
    xyz = {a.name: np.asarray(a.xyz) for a in residue.atoms}
    allowed = set(ATOM14_NAMES[AA3_TO_INDEX[residue.name]]) | {"OXT"}
    if not xyz.keys() <= allowed:
        raise ValueError(
            f"unsupported_atom_for_residue: {residue.key} {sorted(xyz.keys() - allowed)}"
        )
    bonds, _, angles = _tables()
    for bond in bonds[residue.name]:
        if bond.atom1_name not in xyz or bond.atom2_name not in xyz:
            continue
        distance = float(np.linalg.norm(xyz[bond.atom1_name] - xyz[bond.atom2_name]))
        tolerance = max(policy.bond_stddevs * bond.stddev, policy.bond_tolerance_floor_angstrom)
        if abs(distance - bond.length) > tolerance:
            raise ValueError(
                f"invalid_bond_length: {residue.key} {bond.atom1_name}-{bond.atom2_name}={distance:.4f} A"
            )
    if "OXT" in xyz and "C" in xyz:
        distance = float(np.linalg.norm(xyz["OXT"] - xyz["C"]))
        if not 1.0 <= distance <= 1.6:
            raise ValueError(f"invalid_bond_length: {residue.key} C-OXT={distance:.4f} A")
    for angle in angles[residue.name]:
        names = (angle.atom1_name, angle.atom2_name, angle.atom3name)
        if not all(n in xyz for n in names):
            continue
        actual = _angle(*(xyz[n] for n in names))
        tolerance = max(
            policy.angle_stddevs * angle.stddev, math.radians(policy.angle_tolerance_floor_degrees)
        )
        if abs(actual - angle.angle_rad) > tolerance:
            raise ValueError(
                f"invalid_bond_angle: {residue.key} {'-'.join(names)}={math.degrees(actual):.3f} deg"
            )
    if residue.name == "GLY":
        return "achiral_glycine"
    if "CB" not in xyz:
        return "unknown_missing_CB"
    vectors = np.stack([xyz[n] - xyz["CA"] for n in ("N", "C", "CB")])
    volume = float(np.linalg.det(vectors) / np.prod(np.linalg.norm(vectors, axis=-1)))
    # Pinned L-residue N/C/CB ordered volume is positive (including CYS).
    if volume < policy.min_normalized_chiral_volume:
        raise ValueError(f"invalid_CA_chirality: {residue.key} normalized_volume={volume:.4f}")
    # ILE and THR have a second stereocentre; compare its ordered volume to
    # the same atom ordering in the pinned residue template.
    neighbors = {"ILE": ("CA", "CG1", "CG2"), "THR": ("CA", "OG1", "CG2")}.get(residue.name)
    if neighbors and all(n in xyz for n in neighbors):
        reference = {}
        for name, group, pos in constants.rigid_group_atom_positions[residue.name]:
            if group == 0:
                reference[name] = np.array(pos)
            elif group == 4:
                frame = constants.restype_rigid_group_default_frame[AA3_TO_INDEX[residue.name], 4]
                reference[name] = frame[:3, :3] @ np.array(pos) + frame[:3, 3]
        observed = np.stack([xyz[n] - xyz["CB"] for n in neighbors])
        expected = np.stack([reference[n] - reference["CB"] for n in neighbors])
        signed = float(np.linalg.det(observed) / np.prod(np.linalg.norm(observed, axis=-1)))
        if signed * np.sign(np.linalg.det(expected)) < policy.min_normalized_chiral_volume:
            raise ValueError(f"invalid_CB_chirality: {residue.key}")
    return "observed_L_CA"


def validate_link_angles(residues, links, policy):
    for left, right, linked in zip(residues[:-1], residues[1:], links):
        if not linked:
            continue
        a, b = ({x.name: np.asarray(x.xyz) for x in r.atoms} for r in (left, right))
        for points, mean_cos, label in [
            ((a["CA"], a["C"], b["N"]), -0.4473, "CA-C-N"),
            ((a["C"], b["N"], b["CA"]), -0.5203, "C-N-CA"),
        ]:
            # Broad 35 degree inter-residue screen includes cis/trans; it does
            # not enforce omega or turn cis peptides into trans peptides.
            if abs(_angle(*points) - math.acos(mean_cos)) > math.radians(
                policy.link_angle_tolerance_degrees
            ):
                raise ValueError(f"invalid_link_angle: {left.key} -> {right.key} {label}")
