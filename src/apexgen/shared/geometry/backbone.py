"""Differentiable fixed-covalent-geometry peptide backbone reconstruction."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from apexgen.shared.geometry.residue_constants import BACKBONE_GEOMETRY, BackboneGeometry
from apexgen.shared.geometry.torsion import wrap_angle


BACKBONE_ATOMS = ("N", "CA", "C", "O")


def place_atom(
    first: Tensor,
    second: Tensor,
    third: Tensor,
    bond_length: float,
    bond_angle: float,
    dihedral_angle: Tensor,
) -> Tensor:
    """Place a fourth atom from three atoms and standard internal coordinates."""

    eps = torch.finfo(third.dtype).eps
    bc = third - second
    bc = bc / torch.linalg.vector_norm(bc, dim=-1, keepdim=True).clamp_min(eps)
    normal = torch.linalg.cross(second - first, bc, dim=-1)
    normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(eps)
    in_plane = torch.linalg.cross(normal, bc, dim=-1)
    direction = (
        -math.cos(bond_angle) * bc
        + math.sin(bond_angle)
        * (
            torch.cos(dihedral_angle)[..., None] * in_plane
            + torch.sin(dihedral_angle)[..., None] * normal
        )
    )
    return third + bond_length * direction


def build_backbone(
    torsions: Tensor,
    *,
    center_on_ca: bool = True,
    geometry: BackboneGeometry = BACKBONE_GEOMETRY,
) -> Tensor:
    """Build ``(..., L, 4, 3)`` N/CA/C/O coordinates from ``(φ, ψ, ω)``."""

    if torsions.ndim < 2 or torsions.shape[-1] != 3:
        raise ValueError("torsions must have shape (..., L, 3)")
    length = torsions.shape[-2]
    if not 4 <= length <= 25:
        raise ValueError("ApexGen v0 peptide length must be in [4, 25]")
    if not torch.is_floating_point(torsions):
        raise TypeError("torsions must be floating point")
    batch_shape = torsions.shape[:-2]
    zero = torch.zeros(*batch_shape, 3, dtype=torsions.dtype, device=torsions.device)
    n_atoms = [zero]
    ca_atoms = [zero + torch.tensor([geometry.n_ca, 0.0, 0.0], dtype=torsions.dtype, device=torsions.device)]
    initial_direction = torch.tensor(
        [
            -math.cos(geometry.n_ca_c),
            math.sin(geometry.n_ca_c),
            0.0,
        ],
        dtype=torsions.dtype,
        device=torsions.device,
    )
    c_atoms = [ca_atoms[0] + geometry.ca_c * initial_direction]

    for index in range(length - 1):
        next_n = place_atom(
            n_atoms[index],
            ca_atoms[index],
            c_atoms[index],
            geometry.c_n,
            geometry.ca_c_n,
            torsions[..., index, 1],
        )
        next_ca = place_atom(
            ca_atoms[index],
            c_atoms[index],
            next_n,
            geometry.n_ca,
            geometry.c_n_ca,
            torsions[..., index, 2],
        )
        next_c = place_atom(
            c_atoms[index],
            next_n,
            next_ca,
            geometry.ca_c,
            geometry.n_ca_c,
            torsions[..., index + 1, 0],
        )
        n_atoms.append(next_n)
        ca_atoms.append(next_ca)
        c_atoms.append(next_c)

    oxygen_atoms = []
    for index in range(length):
        psi = torsions[..., index, 1] if index < length - 1 else torch.zeros_like(torsions[..., 0, 1])
        oxygen_atoms.append(
            place_atom(
                n_atoms[index],
                ca_atoms[index],
                c_atoms[index],
                geometry.c_o,
                geometry.ca_c_o,
                wrap_angle(psi + math.pi),
            )
        )

    coordinates = torch.stack(
        (
            torch.stack(n_atoms, dim=-2),
            torch.stack(ca_atoms, dim=-2),
            torch.stack(c_atoms, dim=-2),
            torch.stack(oxygen_atoms, dim=-2),
        ),
        dim=-2,
    )
    if center_on_ca:
        coordinates = coordinates - coordinates[..., :, 1, :].mean(dim=-2, keepdim=True)[..., None, :]
    return coordinates
