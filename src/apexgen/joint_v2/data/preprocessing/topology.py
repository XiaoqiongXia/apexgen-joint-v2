"""Polymer positions are distinct from author identities and cropped tensor rows."""

from dataclasses import replace
import math


def assign_polymer_indices(residues, metadata, policy):
    """Use labels or unambiguous sequence alignment; otherwise local fragments."""
    if not residues:
        return residues
    labels = [r.key.label_seq_id for r in residues]
    declared = metadata["declared_sequence"]
    if all(x is not None for x in labels):
        if any(x <= 0 for x in labels) or any(b <= a for a, b in zip(labels, labels[1:])):
            raise ValueError("invalid_or_nonmonotonic_label_seq_id")
        if declared and any(
            x > len(declared) or r.name not in declared[x - 1].split(",")
            for r, x in zip(residues, labels)
        ):
            raise ValueError("label_seq_id disagrees with declared polymer sequence")
        source = "deposited_label_seq_id"
    elif declared:
        # Refuse ambiguous missing runs rather than arbitrarily shifting indices.
        def match(backwards=False):
            result = []
            cursor = len(declared) - 1 if backwards else 0
            for r in reversed(residues) if backwards else residues:
                while 0 <= cursor < len(declared) and r.name not in declared[cursor].split(","):
                    cursor += -1 if backwards else 1
                if not 0 <= cursor < len(declared):
                    raise ValueError("observed sequence differs from declared polymer sequence")
                result.append(cursor + 1)
                cursor += -1 if backwards else 1
            return list(reversed(result)) if backwards else result

        labels = match()
        if labels != match(True):
            raise ValueError(
                "ambiguous_declared_sequence_mapping: supply authoritative mmCIF label_seq_id"
            )
        source = "declared_sequence_alignment"
        for r, label in zip(residues, labels):
            if r.key.label_seq_id is not None and r.key.label_seq_id != label:
                raise ValueError("partial label_seq_id disagrees with sequence alignment")
    elif any(x is not None for x in labels):
        raise ValueError("partial_label_seq_id_without_declared_sequence")
    else:
        source = "observed_connected_fragment"
    result, segment, local_index = [], 0, 0
    previous = None
    for i, r in enumerate(residues):
        if previous is not None:
            c = next((a.xyz for a in previous.atoms if a.name == "C"), None)
            n = next((a.xyz for a in r.atoms if a.name == "N"), None)
            same_segment = (
                previous.key.segment_index == r.key.segment_index
                and previous.key.label_asym_id == r.key.label_asym_id
            )
            if source == "observed_connected_fragment":
                connected = (
                    same_segment
                    and c is not None
                    and n is not None
                    and policy.min_cn_angstrom <= math.dist(c, n) <= policy.max_cn_angstrom
                )
                if not connected:
                    segment += 1
                    local_index = 0
            elif not same_segment:
                segment += 1
        index = local_index if source == "observed_connected_fragment" else labels[i] - 1
        key = replace(r.key, polymer_index=index, index_source=source, model_segment_index=segment)
        result.append(replace(r, key=key))
        local_index += 1
        previous = r
    return result
