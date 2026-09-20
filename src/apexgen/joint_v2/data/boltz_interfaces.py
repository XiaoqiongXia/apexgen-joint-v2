"""Metadata-only inventory of Boltz protein chain-pair interfaces.

Source `interfaces` indexes chain TABLE ROWS (see Boltz compute_interfaces),
not entity IDs or assumed author chain names. Retain masked and noncanonical
chains: this is an inventory, not the stricter training adapter.
"""

from __future__ import annotations

import hashlib
import io
import math

import numpy as np
import pyarrow as pa
from scipy.spatial import cKDTree
from apexgen.joint_v2.data.boltz_schema import protein_heavy_atom_mask


SCHEMA_VERSION = "apexgen.boltz_interface_inventory.v1"
SOURCE_FIELDS = {
    "source_archive": pa.string(), "source_member": pa.string(),
    "source_npz_filename": pa.string(), "source_structure_id": pa.string(),
    "source_offset": pa.int64(), "source_size": pa.int64(), "source_npz_sha256": pa.string(),
}
CHAIN_FIELDS = {
    "row": pa.int32(), "id": pa.string(), "asym_id": pa.int32(),
    "entity_id": pa.int32(), "sym_id": pa.int32(), "source_mask": pa.bool_(),
    "length": pa.int32(), "residue_table_start": pa.int32(),
    "source_present_residue_count": pa.int32(), "observed_residue_count": pa.int32(),
    "observed_heavy_atom_count": pa.int32(), "nonstandard_residue_count": pa.int32(),
    "interface_residue_count": pa.int32(),
    "interface_residue_indices": pa.list_(pa.int32()),
    "interface_residue_rows": pa.list_(pa.int32()),
}
PAIR_SCHEMA = pa.schema([
    pa.field("pair_id", pa.string()),
    *[pa.field(k, t) for k, t in SOURCE_FIELDS.items()],
    pa.field("coordinate_source", pa.string()), pa.field("ensemble_model_count", pa.int32()),
    pa.field("contact_cutoff_angstrom", pa.float64()),
    pa.field("both_source_masks_true", pa.bool_()), pa.field("contact_verified", pa.bool_()),
    pa.field("minimum_contact_distance_angstrom", pa.float64()),
    *[pa.field(f"chain_{side}_{key}", kind) for side in ("a", "b")
      for key, kind in CHAIN_FIELDS.items()],
])
CSV_FIELDS = [f.name for f in PAIR_SCHEMA if not pa.types.is_list(f.type)]


def _table(data, key, fields):
    array = data[key]
    if array.ndim != 1 or not set(fields) <= set(array.dtype.names or ()):
        raise ValueError(f"unsupported {key} table")
    for field, kinds in fields.items():
        dtype = array.dtype.fields[field][0]
        if dtype.subdtype:
            dtype = dtype.subdtype[0]
        if dtype.kind not in kinds:
            raise ValueError(f"invalid {key}.{field} dtype")
    return array


def _span(start, count, total):
    start, count = int(start), int(count)
    if start < 0 or count < 0 or start + count > total:
        raise ValueError(f"invalid source span {start}:{start + count} / {total}")
    return start, start + count


def _chain_geometry(index, chains, residues, atoms, masks):
    chain = chains[index]
    r0, r1 = _span(chain["res_idx"], chain["res_num"], len(residues))
    a0, a1 = _span(chain["atom_idx"], chain["atom_num"], len(atoms))
    rr = residues[r0:r1]
    counts = rr["atom_num"].astype(np.int64)
    starts = a0 + np.concatenate(([0], np.cumsum(counts)))
    if (np.any(counts < 0) or starts[-1] != a1
            or not np.array_equal(rr["atom_idx"], starts[:-1])):
        raise ValueError(f"residue atom spans do not partition chain row {index}")
    atom_residues = np.repeat(np.arange(r0, r1, dtype=np.int32), counts)
    aa = atoms[a0:a1]
    keep = aa["is_present"] & protein_heavy_atom_mask(aa)
    coords = aa["coords"][keep].astype(np.float64)
    if coords.shape != (int(keep.sum()), 3) or not np.isfinite(coords).all():
        raise ValueError(f"invalid observed heavy-atom coordinates in chain row {index}")
    source_rows = atom_residues[keep]
    metadata = dict(
        row=index, id=str(chain["name"]), asym_id=int(chain["asym_id"]),
        entity_id=int(chain["entity_id"]), sym_id=int(chain["sym_id"]),
        source_mask=bool(masks[index]), length=r1 - r0, residue_table_start=r0,
        source_present_residue_count=int(rr["is_present"].sum()),
        observed_residue_count=len(np.unique(source_rows)), observed_heavy_atom_count=len(coords),
        nonstandard_residue_count=int((~rr["is_standard"]).sum()),
    )
    return metadata, coords, source_rows, cKDTree(coords) if len(coords) else None


def describe_interfaces(payload: bytes, source: dict, *, cutoff: float = 5.0):
    """Return all stored protein-pair rows and one source-file audit record.

    Each undirected pair occurs once (a.row < b.row). A residue participates iff
    any present heavy atom is within cutoff, inclusive, of a heavy atom on the
    other chain. Nearest-neighbor queries count residues without materializing
    the much larger all-atom contact graph. Multiple models are not averaged:
    counts always describe the coordinates in atoms.coords.
    """
    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be finite and positive")
    if len(payload) != int(source["source_size"]):
        raise ValueError("source byte length differs from inventory")
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        chains = _table(data, "chains", dict(name="U", mol_type="iu", entity_id="iu",
            sym_id="iu", asym_id="iu", atom_idx="iu", atom_num="iu", res_idx="iu", res_num="iu"))
        interfaces = _table(data, "interfaces", dict(chain_1="iu", chain_2="iu"))
        masks = data["mask"]
        if masks.dtype.kind != "b" or masks.shape != (len(chains),):
            raise ValueError("source mask must be boolean per-chain")
        protein_pairs = set()
        for pair in interfaces:
            a, b = int(pair["chain_1"]), int(pair["chain_2"])
            if not (0 <= a < len(chains) and 0 <= b < len(chains)) or a == b:
                raise ValueError("invalid interface chain-row index")
            if int(chains[a]["mol_type"]) == int(chains[b]["mol_type"]) == 0:
                protein_pairs.add(tuple(sorted((a, b))))
        audit = dict(**source, source_npz_sha256=hashlib.sha256(payload).hexdigest(),
            status="ok", chain_count=len(chains), stored_interface_rows=len(interfaces),
            protein_pair_count=len(protein_pairs))
        if not protein_pairs:
            return [], {**audit, "verified_pair_count": 0, "valid_mask_pair_count": 0}
        atoms = _table(data, "atoms", dict(coords="f", is_present="b"))
        residues = _table(data, "residues", dict(res_idx="iu", atom_idx="iu", atom_num="iu",
            is_present="b", is_standard="b"))
        if atoms["coords"].shape != (len(atoms), 3):
            raise ValueError("atom coordinates must have shape [atoms, 3]")
        ensemble_count = len(data["ensemble"]) if "ensemble" in data else 1
    required = {i for pair in protein_pairs for i in pair}
    cache = {i: _chain_geometry(i, chains, residues, atoms, masks) for i in sorted(required)}
    # nextafter ensures exact-cutoff contacts survive cKDTree's strict upper bound.
    bound = np.nextafter(float(cutoff), math.inf)
    rows = []
    for a, b in sorted(protein_pairs):
        ma, xa, ra, ta = cache[a]
        mb, xb, rb, tb = cache[b]
        da = tb.query(xa, distance_upper_bound=bound, workers=1)[0] if tb is not None else np.full(len(xa), np.inf)
        db = ta.query(xb, distance_upper_bound=bound, workers=1)[0] if ta is not None else np.full(len(xb), np.inf)
        contact_a = np.unique(ra[da <= cutoff])
        contact_b = np.unique(rb[db <= cutoff])
        if bool(len(contact_a)) != bool(len(contact_b)):
            raise RuntimeError("asymmetric nearest-neighbor contact result")
        verified = bool(len(contact_a))
        row = dict(pair_id=f"{source['source_structure_id']}:chain{a}-chain{b}",
            **source, source_npz_sha256=audit["source_npz_sha256"],
            coordinate_source="atoms.coords", ensemble_model_count=ensemble_count,
            contact_cutoff_angstrom=cutoff, both_source_masks_true=ma["source_mask"] and mb["source_mask"],
            contact_verified=verified,
            minimum_contact_distance_angstrom=float(da.min()) if verified else None)
        for side, metadata, contacts in (("a", ma, contact_a), ("b", mb, contact_b)):
            detail = dict(**metadata, interface_residue_count=len(contacts),
                interface_residue_indices=residues["res_idx"][contacts].astype(int).tolist(),
                interface_residue_rows=contacts.astype(int).tolist())
            row.update({f"chain_{side}_{key}": value for key, value in detail.items()})
        rows.append(row)
    audit.update(verified_pair_count=sum(r["contact_verified"] for r in rows),
                 valid_mask_pair_count=sum(r["both_source_masks_true"] for r in rows))
    return rows, audit
