"""Atom identity helpers for published Boltz and BoltzGen structure arrays."""

import numpy as np


def decode_atom_name(encoded):
    """Decode a canonical protein atom name without assuming source slot order."""
    values = np.asarray(encoded)
    if values.shape == () and values.dtype.kind == "U":
        name = str(values.item())
        if not 1 <= len(name) <= 4:
            raise ValueError("invalid string atom name length")
    else:
        if values.shape != (4,) or values.dtype.kind not in "iu":
            raise ValueError("atom name must be a string or four encoded integers")
        nonzero = values[values != 0]
        if (not len(nonzero) or np.any(nonzero < 1) or np.any(nonzero > 94)
                or np.any(values[:len(nonzero)] == 0)):
            raise ValueError("invalid zero-padded atom name")
        name = "".join(chr(int(v) + 32) for v in nonzero)
    if not name.isascii() or not name.isalnum() or name != name.upper():
        raise ValueError(f"unsupported atom name: {name!r}")
    return name


def protein_heavy_atom_mask(atoms):
    """Use explicit elements when available, otherwise protein atom-name conventions.

    Only used for protein chains. This is not a general ligand element inference
    routine; actual training chemistry is checked against canonical AA templates.
    """
    if "element" in (atoms.dtype.names or ()):
        if atoms["element"].dtype.kind not in "iu":
            raise ValueError("invalid atoms.element dtype")
        return atoms["element"] > 1
    if "name" not in (atoms.dtype.names or ()) or atoms["name"].dtype.kind != "U":
        raise ValueError("protein atoms require element IDs or string names")
    names = np.char.lstrip(atoms["name"], "0123456789")
    if names.shape != (len(atoms),) or np.any(names == ""):
        raise ValueError("invalid protein atom names")
    return ~(np.char.startswith(names, "H") | np.char.startswith(names, "D"))
