"""Conservative observation and polymer-continuity checks; never fill coordinates."""

from dataclasses import asdict, replace
import math

import gemmi

from apexgen.joint_v2.data.preprocessing.altloc import select_residue_altloc
from apexgen.joint_v2.data.preprocessing.stereochemistry import GeometryPolicy


# Broad screening bounds, not a restraint or an idealization target.
MIN_PEPTIDE_CN_ANGSTROM = 1.0
MAX_PEPTIDE_CN_ANGSTROM = 2.0


def observed_conformer(residue):
    """Zero-occupancy placeholders are not experimental atom observations."""
    if any(not math.isfinite(a.occupancy) or a.occupancy < 0 for a in residue.atoms):
        raise ValueError(f"{residue.key}: invalid atom occupancy")
    omitted = [a.name for a in residue.atoms if a.occupancy == 0]
    observed = replace(residue, atoms=tuple(a for a in residue.atoms if a.occupancy > 0))
    selected, decision = select_residue_altloc(observed)
    return selected, dict(
        key=asdict(residue.key), zero_occupancy_atoms=omitted, altloc=asdict(decision)
    )


def _successive_keys(left, right):
    if left.polymer_index is not None and right.polymer_index is not None:
        return (
            left.auth_chain_id == right.auth_chain_id
            and left.label_asym_id == right.label_asym_id
            and left.model_segment_index == right.model_segment_index
            and right.polymer_index == left.polymer_index + 1
        )
    if left.segment_index != right.segment_index:
        return False
    if (left.auth_chain_id, left.label_asym_id) != (right.auth_chain_id, right.label_asym_id):
        return False
    if left.label_seq_id is not None and right.label_seq_id is not None:
        return right.label_seq_id == left.label_seq_id + 1
    li, ri = left.insertion_code, right.insertion_code
    if not ri and right.auth_seq_id == left.auth_seq_id + 1:
        return True
    return (
        right.auth_seq_id == left.auth_seq_id
        and len(ri) == 1
        and "A" <= ri <= "Z"
        and ((not li and ri == "A") or (len(li) == 1 and ord(ri) == ord(li) + 1))
    )


def backbone_links(residues, policy=None):
    """Require both sequence adjacency and an observed plausible C--N bond."""
    policy = policy or GeometryPolicy()
    links, breaks = [], []
    for left, right in zip(residues[:-1], residues[1:]):
        reasons = []
        if not _successive_keys(left.key, right.key):
            reasons.append("sequence_or_segment_gap")
        c = next((a.xyz for a in left.atoms if a.name == "C"), None)
        n = next((a.xyz for a in right.atoms if a.name == "N"), None)
        distance = math.dist(c, n) if c is not None and n is not None else None
        if distance is None:
            reasons.append("missing_link_atom")
        elif not policy.min_cn_angstrom <= distance <= policy.max_cn_angstrom:
            reasons.append("implausible_C_N_distance")
        links.append(not reasons)
        if reasons:
            breaks.append(
                dict(
                    left=asdict(left.key),
                    right=asdict(right.key),
                    c_n_distance_angstrom=distance,
                    reasons=reasons,
                )
            )
    return links, breaks


def sequence_metadata(path, structure, model_index, chain_ids):
    """Read declared sequences and explicit unobserved-residue annotations.

    Metadata-free ATOM-only files cannot prove terminal completeness. Never
    synthesize SEQRES from observed atoms or invent coordinates for omissions.
    """
    annotated = structure.clone()
    annotated.setup_entities()
    result = {}
    for chain_id in chain_ids:
        polymer = annotated[model_index][chain_id].get_polymer()
        entity = annotated.get_entity_of(polymer)
        sequence = list(entity.full_sequence) if entity is not None else []
        result[chain_id] = dict(
            declared_sequence=sequence, unobserved_residues=[], ter_after_residues=[]
        )
    model_name = str(structure[model_index].name)
    if path.suffix.lower() in {".pdb", ".ent"}:
        current_model, last_atom = "1", None
        for line in path.read_text().splitlines():
            if line.startswith("MODEL "):
                current_model, last_atom = line[10:14].strip(), None
            elif line.startswith("ENDMDL"):
                last_atom = None
            elif line.startswith(("ATOM  ", "HETATM")) and current_model == model_name:
                last_atom = (line[21:22], line[22:26].strip(), line[26:27].strip())
            elif line.startswith("TER") and last_atom is not None:
                chain, seq, insertion = last_atom
                if chain in result:
                    result[chain]["ter_after_residues"].append(
                        dict(auth_seq_id=int(seq), insertion_code=insertion)
                    )
                last_atom = None
            if not line.startswith("REMARK 465") or len(line) < 26:
                continue
            chain = line[19:20]
            seq = line[21:26].strip()
            model = line[11:14].strip()
            if chain not in result or not seq.lstrip("-").isdigit():
                continue
            if model and model != model_name:
                continue
            result[chain]["unobserved_residues"].append(
                dict(
                    residue_name=line[15:18].strip(),
                    auth_seq_id=int(seq),
                    insertion_code=line[26:27].strip(),
                    source="REMARK 465",
                )
            )
    else:
        block = gemmi.cif.read_file(str(path)).sole_block()
        category = block.get_mmcif_category("_pdbx_unobs_or_zero_occ_residues.")
        for i, chain in enumerate(category.get("auth_asym_id", [])):

            def value(key):
                values = category.get(key)
                return values[i] if values is not None else None

            model = value("PDB_model_num")
            if chain not in result or (model not in {None, False, "", model_name}):
                continue
            if value("polymer_flag") == "N":
                continue
            result[chain]["unobserved_residues"].append(
                dict(
                    residue_name=value("auth_comp_id"),
                    auth_seq_id=value("auth_seq_id"),
                    insertion_code=value("PDB_ins_code"),
                    source="_pdbx_unobs_or_zero_occ_residues",
                )
            )
    return result


def validate_peptide_sequence(peptide, metadata, policy=None):
    if metadata["unobserved_residues"]:
        raise ValueError(
            "peptide has annotated unobserved residues; cannot use incomplete native labels"
        )
    declared = metadata["declared_sequence"]
    if (
        peptide[0].key.index_source in {"deposited_label_seq_id", "declared_sequence_alignment"}
        and peptide[0].key.polymer_index > 0
    ):
        raise ValueError(
            "peptide has unresolved N-terminal polymer positions before first label_seq_id"
        )
    if declared and (
        len(declared) != len(peptide)
        or any(r.name not in names.split(",") for r, names in zip(peptide, declared))
    ):
        raise ValueError("peptide observed sequence differs from declared SEQRES/entity_poly_seq")
    links, breaks = backbone_links(peptide, policy)
    if not all(links):
        raise ValueError(f"peptide chain break or unresolved sequence gap: {breaks[0]}")
    return "declared_sequence_matched" if declared else "unknown_no_declared_sequence"
