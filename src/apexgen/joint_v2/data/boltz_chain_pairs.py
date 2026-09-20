"""Directed interface conditions with continuous target-chain fragments.

Each maximal consecutive run of interface positions is a separate sample,
never the envelope of several runs and never concatenated pieces. The
existing strict NPZ adapter owns chemistry, observed backbone and mapping QC.
"""

import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from apexgen.joint_v2.data.boltz_npz import adapt_boltz_npz, audit_boltz_record
from apexgen.joint_v2.data.preprocessing.records import PocketParameters


SCHEMA = "apexgen.boltz_all_interface_fragments.v1"
DIRECTIONS = ("a_to_b", "b_to_a")


def fragment_selections(pair, direction):
    """All maximal contact runs in source order; length/QC filtering is separate."""
    if direction not in DIRECTIONS:
        raise ValueError(f"unsupported direction: {direction}")
    condition_side, target_side = ("a", "b") if direction == "a_to_b" else ("b", "a")
    positions = pair[f"chain_{target_side}_interface_residue_indices"]
    length = pair[f"chain_{target_side}_length"]
    if (not positions or positions != sorted(set(positions))
            or any(not 0 <= p < length for p in positions)):
        raise ValueError("invalid interface polymer positions")
    runs = []
    start = previous = positions[0]
    for position in positions[1:]:
        if position != previous + 1:
            runs.append([start, previous + 1])
            start = position
        previous = position
    runs.append([start, previous + 1])
    return [dict(condition_side=condition_side, target_side=target_side,
        condition_chain_id=pair[f"chain_{condition_side}_id"],
        target_chain_id=pair[f"chain_{target_side}_id"],
        target_source_length=length, target_start=start, target_stop=stop,
        fragment_length=stop - start, interface_residue_count=len(positions),
        selected_interface_residue_count=stop - start,
        omitted_interface_residue_count=len(positions) - (stop - start),
        contact_runs=runs, contact_run_count=len(runs),
        noninterface_residues_in_fragment=0,
        interface_coverage=(stop - start) / len(positions), interface_density=1.0,
        largest_internal_noncontact_gap=max((b - a - 1 for a, b in zip(positions, positions[1:])), default=0),
        target_is_complete_source_chain=(start == 0 and stop == length), fragment_index=index)
        for index, (start, stop) in enumerate(runs)]


def adapt_interface_direction(path, pair, direction, *, fragment_index=0, split="smoke"):
    """Return a record and independent atom-identity audit; reject rather than trim.

The source interface selects the binding site. Conditions use the adapter's
    5-A core plus 11-A CB context. The generated side retains EVERY residue of
    the explicitly selected consecutive interface run. Other runs become
    separate samples when selected by the caller.
Both directions must use the same externally assigned source-structure split.
"""
    if direction not in DIRECTIONS:
        raise ValueError(f"unsupported direction: {direction}")
    payload = Path(path).read_bytes()
    if (len(payload) != pair["source_size"]
            or hashlib.sha256(payload).hexdigest() != pair["source_npz_sha256"]):
        raise ValueError("interface inventory/source NPZ identity mismatch")
    if not pair["contact_verified"] or pair["contact_cutoff_angstrom"] != 5.0:
        raise ValueError("requires a verified 5-A interface inventory")
    if not pair["chain_a_row"] < pair["chain_b_row"]:
        raise ValueError("expected one ordered, distinct source chain pair")
    expected_id = (f"{pair['source_structure_id']}:chain{pair['chain_a_row']}"
                   f"-chain{pair['chain_b_row']}")
    if pair["pair_id"] != expected_id:
        raise ValueError("pair ID disagrees with source chain rows")
    with np.load(path, allow_pickle=False) as data:
        for side in ("a", "b"):
            prefix = f"chain_{side}_"
            index = pair[prefix + "row"]
            if not 0 <= index < len(data["chains"]):
                raise ValueError("inventory chain row out of bounds")
            chain = data["chains"][index]
            for dest, source in (("id", "name"), ("length", "res_num"),
                                 ("residue_table_start", "res_idx"), ("asym_id", "asym_id"),
                                 ("entity_id", "entity_id"), ("sym_id", "sym_id")):
                if pair[prefix + dest] != chain[source]:
                    raise ValueError(f"inventory chain identity mismatch: {prefix + dest}")
            rows = pair[prefix + "interface_residue_rows"]
            start, count = int(chain["res_idx"]), int(chain["res_num"])
            if (not rows or rows != sorted(set(rows))
                    or any(not start <= r < start + count for r in rows)
                    or len(rows) != pair[prefix + "interface_residue_count"]
                    or data["residues"]["res_idx"][rows].tolist()
                    != pair[prefix + "interface_residue_indices"]):
                raise ValueError("inventory interface residue identity mismatch")
    selections = fragment_selections(pair, direction)
    if (isinstance(fragment_index, bool) or not isinstance(fragment_index, int)
            or not 0 <= fragment_index < len(selections)):
        raise ValueError("fragment_index outside source contact runs")
    selection = selections[fragment_index]
    condition_side, target_side = selection["condition_side"], selection["target_side"]
    condition_id, target_id = selection["condition_chain_id"], selection["target_chain_id"]
    lo, hi = selection["target_start"], selection["target_stop"]
    sample_id = f"{pair['pair_id']}:{direction}:fragment{lo}-{hi}"
    record = adapt_boltz_npz(path, sample_id=sample_id,
        source_pdb_id=pair["source_structure_id"], receptor_chain_ids=(condition_id,),
        peptide_chain_id=target_id, split=split, parameters=PocketParameters(),
        target_residue_range=(lo, hi))
    length = hi - lo
    start = pair[f"chain_{target_side}_residue_table_start"] + lo
    keys = record["peptide_residue_keys"]
    if (record["peptide_length"] != length
            or [k["boltz_residue_row"] for k in keys] != list(range(start, start + length))
            or [k["polymer_index"] for k in keys] != list(range(lo, hi))
            or any(k["boltz_chain_name"] != target_id for k in keys)):
        raise ValueError("generated fragment was truncated, reordered or concatenated")
    # Recheck the actual model atoms after missing-backbone filtering, context
    # cropping and atom-slot packing. Original source contacts are insufficient.
    context_xyz = record["pocket_atom_xyz"][record["pocket_atom_mask"].astype(bool)]
    target = record["joint_v2_target"]
    target_mask = target["experimental_atom14_mask"].astype(bool)
    nearest = cKDTree(context_xyz).query(target["experimental_atom14"][target_mask])[0]
    contact = np.zeros_like(target_mask)
    # Centered coordinates are float32; tolerate only rounding at the 5-A edge.
    contact[target_mask] = nearest <= pair["contact_cutoff_angstrom"] + 1e-5
    lost = np.flatnonzero(~contact.any(axis=1))
    if len(lost):
        positions = [keys[int(i)]["polymer_index"] for i in lost]
        raise ValueError(f"target residues lost contact with retained context: {positions}")
    # Provenance only: collate does not expose native interface membership of
    # the generated side as an input to the model.
    record["interface_pair"] = dict(schema=SCHEMA, pair_id=pair["pair_id"], direction=direction,
        split_group=pair["source_structure_id"], **selection,
        condition_interface_residue_rows=pair[f"chain_{condition_side}_interface_residue_rows"],
        target_interface_residue_rows=list(range(start, start + length)),
        all_target_interface_residue_rows=pair[f"chain_{target_side}_interface_residue_rows"],
        source_archive=pair["source_archive"], source_member=pair["source_member"],
        source_offset=pair["source_offset"], source_size=pair["source_size"])
    audit = audit_boltz_record(record)
    audit.update(record["interface_pair"])
    audit["nearby_excluded_atom_count"] = len(record["boltz_adapter"]["nearby_excluded_atom_rows"])
    return record, audit
