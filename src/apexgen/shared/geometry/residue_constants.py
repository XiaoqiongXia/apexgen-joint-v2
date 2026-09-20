"""Versioned ideal peptide-backbone constants and OpenFold reference identity."""

from __future__ import annotations

import math
from dataclasses import dataclass


OPENFOLD_REPOSITORY = "https://github.com/aqlaboratory/openfold"
OPENFOLD_COMMIT = "be2ec1841f16c966c65ae0e7599ebbadc725757d"
OPENFOLD_LICENSE = "Apache-2.0"
OPENFOLD_RESIDUE_CONSTANTS_PATH = "openfold/np/residue_constants.py"
OPENFOLD_RIGID_UTILS_PATH = "openfold/utils/rigid_utils.py"


@dataclass(frozen=True)
class BackboneGeometry:
    """Ideal trans-peptide covalent geometry in Å and radians."""

    n_ca: float = 1.458
    ca_c: float = 1.525
    c_n: float = 1.329
    c_o: float = 1.229
    c_n_ca: float = math.radians(121.7)
    n_ca_c: float = math.radians(111.2)
    ca_c_n: float = math.radians(116.2)
    ca_c_o: float = math.radians(120.8)


BACKBONE_GEOMETRY = BackboneGeometry()
