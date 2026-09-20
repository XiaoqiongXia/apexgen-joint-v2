"""Inspect explicit covalent topology and omitted surroundings in the full model."""

import gemmi
import numpy as np

from apexgen.joint_v2.data.preprocessing.chemistry import STANDARD_AMINO_ACIDS, RECEPTOR_RESIDUE_MAPPINGS
from apexgen.joint_v2.data.preprocessing.stereochemistry import _tables


def _identity(chain, residue):
    return (chain, int(residue.seqid.num), residue.seqid.icode.strip())


def _address(address):
    return (
        address.chain_name,
        int(address.res_id.seqid.num),
        address.res_id.seqid.icode.strip(),
        address.atom_name,
    )


def inspect_structure_context(path, structure, model_index, receptor_ids, peptide_id, radius):
    model = structure[model_index]
    peptide = [
        r
        for r in model[peptide_id]
        if r.name in STANDARD_AMINO_ACIDS
        or r.entity_type == gemmi.EntityType.Polymer
        or gemmi.find_tabulated_residue(r.name).is_amino_acid()
    ]
    identities = {_identity(peptide_id, r): i for i, r in enumerate(peptide)}
    bonds, _, _ = _tables()
    checked = []

    def check_connection(a, b, kind, source):
        if a[0] != peptide_id and b[0] != peptide_id:
            return
        if kind in {"Hydrog", "MetalC"}:
            return
        allowed = False
        if a[:3] in identities and b[:3] in identities:
            i, j = identities[a[:3]], identities[b[:3]]
            if i == j and peptide[i].name in bonds:
                allowed = frozenset((a[3], b[3])) in {
                    frozenset((x.atom1_name, x.atom2_name)) for x in bonds[peptide[i].name]
                }
                allowed |= set((a[3], b[3])) == {"C", "OXT"}
            else:
                allowed = (j == i + 1 and a[3] == "C" and b[3] == "N") or (
                    i == j + 1 and b[3] == "C" and a[3] == "N"
                )
        if not allowed:
            raise ValueError(f"unsupported_peptide_covalent_connection: {source} {kind} {a} -> {b}")
        checked.append(dict(partner1=list(a), partner2=list(b), type=kind, source=source))

    for connection in structure.connections:
        check_connection(
            _address(connection.partner1),
            _address(connection.partner2),
            connection.type.name,
            "LINK/SSBOND/struct_conn",
        )
    if path.suffix.lower() in {".pdb", ".ent"}:
        serials = {}
        for chain in model:
            for residue in chain:
                for atom in residue:
                    serials.setdefault(atom.serial, []).append(
                        (*_identity(chain.name, residue), atom.name)
                    )
        selected_model, current_model = str(model.name), None
        for line in path.read_text().splitlines():
            if line.startswith("MODEL "):
                current_model = line[10:14].strip()
            elif line.startswith("ENDMDL"):
                current_model = None
            elif line.startswith("CONECT") and current_model in {None, selected_model}:
                try:
                    values = [
                        int(line[i : i + 5])
                        for i in range(6, min(len(line), 31), 5)
                        if line[i : i + 5].strip()
                    ]
                except ValueError as exc:
                    raise ValueError("unreadable_CONECT_serial") from exc
                for other in values[1:]:
                    left, right = serials.get(values[0], []), serials.get(other, [])
                    if len(left) != 1 or len(right) != 1:
                        raise ValueError("unresolved_or_ambiguous_CONECT_partner")
                    check_connection(left[0], right[0], "Covale", "CONECT")

    def heavy(residue):
        atoms = [a for a in residue if a.occ > 0]
        alternates = {a.altloc for a in atoms if a.altloc not in {"\x00", " "}}
        if alternates:
            shared = [a for a in atoms if a.altloc in {"\x00", " "}]

            def score(altloc):
                conformer = [a for a in atoms if a.altloc == altloc]
                names = {a.name for a in [*shared, *conformer]}
                return (
                    -len(names & {"N", "CA", "C"}),
                    -sum(a.occ for a in conformer),
                    altloc != "A",
                    altloc,
                )

            selected = min(alternates, key=score)
            atoms = [a for a in atoms if a.altloc in {"\x00", " ", selected}]
        return [a for a in atoms if not a.element.is_hydrogen]

    peptide_atoms = [a for r in peptide for a in heavy(r)]
    peptide_xyz = np.asarray([[a.pos.x, a.pos.y, a.pos.z] for a in peptide_atoms])
    if not len(peptide_atoms):
        return dict(nearby_excluded_residues=[], checked_connections=checked)
    if not np.isfinite(peptide_xyz).all():
        raise ValueError("atom coordinates must be finite")
    owners = np.array([i for i, r in enumerate(peptide) for a in heavy(r)])
    for i, atom in enumerate(peptide_atoms):
        distances = np.linalg.norm(peptide_xyz[i + 1 :] - peptide_xyz[i], axis=-1)
        for offset in np.flatnonzero(distances < 1.9):
            j = i + 1 + int(offset)
            if owners[i] == owners[j]:
                continue
            other = peptide_atoms[j]
            normal_link = owners[j] == owners[i] + 1 and atom.name == "C" and other.name == "N"
            if normal_link or atom.element.is_metal or other.element.is_metal:
                continue
            threshold = min(1.9, float(atom.element.covalent_r + other.element.covalent_r) + 0.15)
            if distances[offset] < threshold:
                raise ValueError("possible_peptide_internal_crosslink_or_severe_clash")
    # Catch head-to-tail closure and disulfide crosslinks even without LINK.
    if len(peptide) > 1:
        first_n, last_c = peptide[0].find_atom("N", "*"), peptide[-1].find_atom("C", "*")
        if (
            first_n
            and last_c
            and first_n.occ > 0
            and last_c.occ > 0
            and first_n.pos.dist(last_c.pos) < 1.9
        ):
            raise ValueError("possible_cyclic_peptide: terminal C-N proximity")
    sulfurs = [a for r in peptide for a in heavy(r) if a.name == "SG"]
    if any(a.pos.dist(b.pos) < 2.3 for i, a in enumerate(sulfurs) for b in sulfurs[i + 1 :]):
        raise ValueError("possible_peptide_disulfide_crosslink")
    nearby = []
    allowed = STANDARD_AMINO_ACIDS | frozenset(RECEPTOR_RESIDUE_MAPPINGS)
    for chain in model:
        for residue in chain:
            if _identity(chain.name, residue) in identities:
                continue
            atoms = heavy(residue)
            if not atoms:
                continue
            xyz = np.asarray([[a.pos.x, a.pos.y, a.pos.z] for a in atoms])
            if not np.isfinite(xyz).all():
                raise ValueError("full-model environment coordinates must be finite")
            # Small residue-sized blocks keep memory bounded for large models.
            distances = np.linalg.norm(xyz[:, None] - peptide_xyz[None], axis=-1)
            minimum = float(distances.min())
            for i, j in np.argwhere(distances < 2.4):
                a, b = atoms[i], peptide_atoms[j]
                if a.element.is_metal or b.element.is_metal:
                    continue
                threshold = min(1.9, float(a.element.covalent_r + b.element.covalent_r) + 0.15)
                if distances[i, j] < threshold:
                    raise ValueError(
                        f"possible_peptide_external_covalent_or_severe_clash: {chain.name}:{residue.seqid}:{a.name} distance={distances[i, j]:.3f}"
                    )
            included_protein = chain.name in receptor_ids and residue.name in allowed
            if not included_protein and minimum <= radius:
                nearby.append(
                    dict(
                        chain_id=chain.name,
                        auth_seq_id=int(residue.seqid.num),
                        insertion_code=residue.seqid.icode.strip(),
                        residue_name=residue.name,
                        min_peptide_distance_angstrom=minimum,
                        category="protein_chain"
                        if residue.name in allowed
                        else "nonprotein_or_unsupported",
                    )
                )
    return dict(
        nearby_excluded_residues=nearby,
        checked_connections=checked,
        environment_radius_angstrom=radius,
        assembly_status="as_supplied_no_assembly_expansion",
        available_assembly_ids=[a.name for a in structure.assemblies],
        gemmi_version=gemmi.__version__,
        selected_model_name=str(model.name),
        total_model_count=len(structure),
    )
