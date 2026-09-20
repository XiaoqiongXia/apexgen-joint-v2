"""Pinned 20-amino-acid atom14/chi tables for joint-v1.

The numerical tables are adapted from OpenFold commit
``be2ec1841f16c966c65ae0e7599ebbadc725757d`` (Apache-2.0).  Keeping this
small public façade prevents joint-v1 from depending on OpenFold at runtime.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
from torch import Tensor

from apexgen.shared.geometry import _openfold_joint_constants as _of


AA3_ORDER = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
)
AA1_ORDER = tuple(_of.restypes)
AA3_TO_INDEX = {name: index for index, name in enumerate(AA3_ORDER)}
ATOM14_NAMES = tuple(tuple(_of.restype_name_to_atom14_names[name]) for name in AA3_ORDER)
ATOM14_MASK = np.asarray(_of.restype_atom14_mask[:20], dtype=np.float32)
ATOM14_GROUP = np.asarray(_of.restype_atom14_to_rigid_group[:20], dtype=np.int64)
ATOM14_LOCAL_POSITIONS = np.asarray(_of.restype_atom14_rigid_group_positions[:20], dtype=np.float32)
RIGID_GROUP_DEFAULT_FRAMES = np.asarray(_of.restype_rigid_group_default_frame[:20], dtype=np.float32)
CHI_EXISTS = np.asarray(_of.chi_angles_mask[:20], dtype=np.float32)
CHI_PI_PERIODIC = np.asarray(_of.chi_pi_periodic[:20], dtype=np.float32)
CHI_ATOM_NAMES = tuple(tuple(tuple(atom for atom in chi) for chi in _of.chi_angles_atoms[name]) for name in AA3_ORDER)
ATOM14_AMBIGUOUS = np.asarray(_of.restype_atom14_ambiguous_atoms[:20], dtype=np.float32)
ATOM14_AMBIGUITY_SWAP_INDEX = np.asarray(_of.restype_atom14_ambiguous_atoms_swap_idx[:20], dtype=np.int64)
BETWEEN_RES_BOND_LENGTH_C_N = np.asarray(_of.between_res_bond_length_c_n, dtype=np.float32)
BETWEEN_RES_BOND_STDDEV_C_N = np.asarray(_of.between_res_bond_length_stddev_c_n, dtype=np.float32)
BETWEEN_RES_COS_ANGLES_C_N_CA = np.asarray(_of.between_res_cos_angles_c_n_ca, dtype=np.float32)
BETWEEN_RES_COS_ANGLES_CA_C_N = np.asarray(_of.between_res_cos_angles_ca_c_n, dtype=np.float32)


def _backbone_stereochemical_statistics() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return residue-conditioned N-CA/CA-C and N-CA-C statistics."""

    residue_bonds, _, residue_angles = _of.load_stereo_chemical_props()
    bond_mean = np.empty((20, 2), dtype=np.float32)
    bond_stddev = np.empty((20, 2), dtype=np.float32)
    angle_mean = np.empty(20, dtype=np.float32)
    angle_stddev = np.empty(20, dtype=np.float32)
    for residue_index, residue_name in enumerate(AA3_ORDER):
        bond_lookup = {
            frozenset((bond.atom1_name, bond.atom2_name)): bond
            for bond in residue_bonds[residue_name]
        }
        for column, atom_pair in enumerate((("N", "CA"), ("CA", "C"))):
            bond = bond_lookup[frozenset(atom_pair)]
            bond_mean[residue_index, column] = bond.length
            bond_stddev[residue_index, column] = bond.stddev
        angle = next(
            angle
            for angle in residue_angles[residue_name]
            if angle.atom2_name == "CA"
            and {angle.atom1_name, angle.atom3name} == {"N", "C"}
        )
        angle_mean[residue_index] = angle.angle_rad
        angle_stddev[residue_index] = angle.stddev
    return bond_mean, bond_stddev, angle_mean, angle_stddev


(
    BACKBONE_BOND_LENGTH_MEAN,
    BACKBONE_BOND_LENGTH_STDDEV,
    BACKBONE_N_CA_C_ANGLE_MEAN_RAD,
    BACKBONE_N_CA_C_ANGLE_STDDEV_RAD,
) = _backbone_stereochemical_statistics()


def _backbone_stereochemistry_sha256() -> str:
    digest = hashlib.sha256()
    for value in (
        BACKBONE_BOND_LENGTH_MEAN,
        BACKBONE_BOND_LENGTH_STDDEV,
        BACKBONE_N_CA_C_ANGLE_MEAN_RAD,
        BACKBONE_N_CA_C_ANGLE_STDDEV_RAD,
        BETWEEN_RES_BOND_LENGTH_C_N,
        BETWEEN_RES_BOND_STDDEV_C_N,
        BETWEEN_RES_COS_ANGLES_C_N_CA,
        BETWEEN_RES_COS_ANGLES_CA_C_N,
    ):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


BACKBONE_STEREOCHEMISTRY_SHA256 = _backbone_stereochemistry_sha256()


def _make_atom14_distance_bounds(
    overlap_tolerance: float = 1.5,
    bond_length_tolerance_factor: float = 12.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.zeros((20, 14, 14), dtype=np.float32)
    upper = np.zeros((20, 14, 14), dtype=np.float32)
    stddev = np.zeros((20, 14, 14), dtype=np.float32)
    residue_bonds, residue_virtual_bonds, _ = _of.load_stereo_chemical_props()
    for residue_index, residue_name in enumerate(AA3_ORDER):
        names = ATOM14_NAMES[residue_index]
        for first_index, first_name in enumerate(names):
            if not first_name:
                continue
            for second_index, second_name in enumerate(names):
                if not second_name or first_index == second_index:
                    continue
                lower[residue_index, first_index, second_index] = (
                    _of.van_der_waals_radius[first_name[0]]
                    + _of.van_der_waals_radius[second_name[0]]
                    - overlap_tolerance
                )
                upper[residue_index, first_index, second_index] = 1e10
        for bond in residue_bonds[residue_name] + residue_virtual_bonds[residue_name]:
            first_index = names.index(bond.atom1_name)
            second_index = names.index(bond.atom2_name)
            lo = bond.length - bond_length_tolerance_factor * bond.stddev
            hi = bond.length + bond_length_tolerance_factor * bond.stddev
            lower[residue_index, first_index, second_index] = lo
            lower[residue_index, second_index, first_index] = lo
            upper[residue_index, first_index, second_index] = hi
            upper[residue_index, second_index, first_index] = hi
            stddev[residue_index, first_index, second_index] = bond.stddev
            stddev[residue_index, second_index, first_index] = bond.stddev
    return lower, upper, stddev


ATOM14_DISTANCE_LOWER_BOUND, ATOM14_DISTANCE_UPPER_BOUND, ATOM14_DISTANCE_STDDEV = _make_atom14_distance_bounds()


def _constants_sha256() -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps({"aa1": AA1_ORDER, "aa3": AA3_ORDER, "atom14": ATOM14_NAMES}, separators=(",", ":")).encode())
    for value in (ATOM14_MASK, ATOM14_GROUP, ATOM14_LOCAL_POSITIONS, RIGID_GROUP_DEFAULT_FRAMES, CHI_EXISTS, CHI_PI_PERIODIC, ATOM14_AMBIGUITY_SWAP_INDEX, ATOM14_DISTANCE_LOWER_BOUND, ATOM14_DISTANCE_UPPER_BOUND):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


ATOM14_CONSTANTS_SHA256 = _constants_sha256()


def as_torch(value: np.ndarray, *, device: torch.device, dtype: torch.dtype | None = None) -> Tensor:
    """Materialize a pinned table on the requested device without global caches."""

    return torch.as_tensor(value, device=device, dtype=dtype)


def atom14_index(aatype: Tensor, atom_name: str) -> Tensor:
    """Return atom14 index or ``-1`` for each residue type."""

    if aatype.dtype != torch.long or aatype.numel() and (aatype.min() < 0 or aatype.max() >= 20):
        raise ValueError("aatype must contain OpenFold-order integers in [0, 20)")
    lookup = torch.full((20,), -1, dtype=torch.long, device=aatype.device)
    for residue, names in enumerate(ATOM14_NAMES):
        if atom_name in names:
            lookup[residue] = names.index(atom_name)
    return lookup[aatype]
