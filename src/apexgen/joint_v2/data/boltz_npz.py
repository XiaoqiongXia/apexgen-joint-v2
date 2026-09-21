"""Boltz processed structures -> observed Joint-v2 complex records.

The supported wire format is Boltz's structured-array Structure NPZ, not its
tokenized model batch. Roles are explicit assembly-chain names. No Boltz import,
sequence offset arithmetic, reference conformer, MSA or inferred author numbering
is used. This first adapter supports canonical, linear protein peptides only.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from apexgen.joint_v2.contracts.contract import JOINT_V2_CONTRACT_SHA256
from apexgen.joint_v2.data.batch import POCKET_ATOM_NAMES
from apexgen.joint_v2.data.static_features import precompute_record
from apexgen.joint_v2.data.boltz_schema import decode_atom_name
from apexgen.joint_v2.data.dataset import COMPLEX_RECORD_SCHEMA, NATIVE_TARGET_SCHEMA
from apexgen.joint_v2.data.native_targets import observed_backbone_frames
from apexgen.joint_v2.data.preprocessing.quality import backbone_links
from apexgen.joint_v2.data.preprocessing.records import PocketParameters
from apexgen.joint_v2.data.preprocessing.stereochemistry import (
    validate_elements, validate_link_angles, validate_residue_geometry,
)
from apexgen.joint_v2.data.preprocessing.structure import AtomRecord, ResidueKey, ResidueRecord
from apexgen.shared.geometry.joint_residue_constants import (
    AA1_ORDER, AA3_ORDER, AA3_TO_INDEX, ATOM14_NAMES, ATOM14_CONSTANTS_SHA256,
)
from apexgen.shared.geometry.torsion import extract_backbone_torsions
from apexgen.shared.storage.features import virtual_cb
from apexgen.shared.storage.store import RECORD_SCHEMA_VERSION


ADAPTER_SCHEMA = "apexgen.boltz_structure_adapter.v2"
# Independent source vocabulary: cross-check names AND IDs, never assume target
# vocabulary will retain its current order. Boltz data/const.py tokens[2:22].
BOLTZ_AA3 = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
)
BOLTZ_AA_IDS = {name: index + 2 for index, name in enumerate(BOLTZ_AA3)}
ELEMENTS = {6: "C", 7: "N", 8: "O", 16: "S"}


def mapping_manifest():
    """Serializable mapping contract, including residue-dependent atom14 slots."""
    payload = dict(
        schema=ADAPTER_SCHEMA,
        atom_name_encoding="Boltz: four zero-padded integers, ord(character)-32; BoltzGen: Unicode name",
        connection_chain_identity="zero-based chains table row, never asym_id",
        canonical_element_policy="AA atom-name template; cross-check explicit element when present",
        amino_acids=[dict(name=n, one_letter=AA1_ORDER[AA3_TO_INDEX[n]],
                         boltz_id=BOLTZ_AA_IDS[n], apexgen_id=AA3_TO_INDEX[n],
                         atom14=list(ATOM14_NAMES[AA3_TO_INDEX[n]])) for n in BOLTZ_AA3],
        pocket_atom_names=list(POCKET_ATOM_NAMES),
        atom14_constants_sha256=ATOM14_CONSTANTS_SHA256,
        joint_v2_contract_sha256=JOINT_V2_CONTRACT_SHA256,
    )
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {**payload, "mapping_sha256": digest}


def canonical_aatype(name, boltz_id):
    if name not in BOLTZ_AA_IDS:
        raise ValueError(f"unsupported canonical residue: {name}")
    if not isinstance(boltz_id, (int, np.integer)) or isinstance(boltz_id, (bool, np.bool_)):
        raise ValueError("Boltz res_type must be an integer")
    if int(boltz_id) != BOLTZ_AA_IDS[name]:
        raise ValueError(f"Boltz residue name/token mismatch: {name} / {boltz_id}")
    return AA3_TO_INDEX[name]


def validate_boltz_record_contract(record):
    """Reject stale or foreign Boltz mappings at both dataset and batch entry."""
    if "boltz_adapter" not in record:
        return
    adapter = record["boltz_adapter"]
    if (not isinstance(adapter, dict) or adapter.get("schema") != ADAPTER_SCHEMA
            or adapter.get("mapping_sha256") != mapping_manifest()["mapping_sha256"]):
        raise ValueError("Boltz mapping contract mismatch; rebuild with the current adapter")
    if record.get("joint_v2_target", {}).get("joint_v2_contract_sha256") != JOINT_V2_CONTRACT_SHA256:
        raise ValueError("Boltz target Joint-v2 contract mismatch")


def _table(data, name, fields):
    if name not in data:
        raise ValueError(f"missing Boltz table: {name}")
    table = data[name]
    if table.ndim != 1 or table.dtype.names is None or not set(fields) <= set(table.dtype.names):
        raise ValueError(f"unsupported Boltz {name} schema; requires {fields}")
    for field, kind in fields.items():
        dtype = table.dtype.fields[field][0]
        if dtype.subdtype:
            dtype = dtype.subdtype[0]
        if dtype.kind not in kind:
            raise ValueError(f"invalid {name}.{field} dtype")
    return table


def _range(start, count, size, label):
    start, count = int(start), int(count)
    if start < 0 or count <= 0 or start + count > size:
        raise ValueError(f"invalid {label} span: {start}, {count}, size={size}")
    return range(start, start + count)


def _xyz(residue, name):
    return np.asarray(next(a.xyz for a in residue.atoms if a.name == name), dtype=np.float64)


def _cb(residue):
    if residue.name == "GLY":
        return _xyz(residue, "CA")
    if any(a.name == "CB" for a in residue.atoms):
        return _xyz(residue, "CB")
    return virtual_cb(*(torch.from_numpy(_xyz(residue, n)) for n in ("N", "CA", "C"))).numpy()


def _key(residue, chains, residue_row, chain_row):
    # No auth_seq_id/auth_chain_id in exported keys: NPZ discarded those fields.
    # Internal ResidueKey uses a surrogate only for shared geometry QC.
    chain = chains[chain_row]
    return dict(polymer_index=residue.key.polymer_index, sample_index=residue.key.sample_index,
                label_asym_id=f"boltz_asym:{int(chain['asym_id'])}",
                index_source="boltz_res_idx_zero_based", boltz_chain_name=str(chain["name"]),
                boltz_chain_row=chain_row, boltz_asym_id=int(chain["asym_id"]),
                boltz_entity_id=int(chain["entity_id"]), boltz_sym_id=int(chain["sym_id"]),
                boltz_residue_row=residue_row)


def _validated_tables(data):
    atoms = _table(data, "atoms", dict(name="iuU", coords="f", is_present="b"))
    native_names = atoms["name"].dtype.kind == "U"
    source_schema = "boltzgen_string_names" if native_names else "boltz_integer_names"
    if not native_names or "element" in atoms.dtype.names:
        _table(data, "atoms", dict(element="iu"))
    residues = _table(data, "residues", dict(name="U", res_type="iu", res_idx="iu",
        atom_idx="iu", atom_num="iu", is_standard="b", is_present="b"))
    chains = _table(data, "chains", dict(name="U", mol_type="iu", entity_id="iu", sym_id="iu",
        asym_id="iu", atom_idx="iu", atom_num="iu", res_idx="iu", res_num="iu"))
    chain_mask = data.get("mask")
    if chain_mask is None or chain_mask.shape != (len(chains),) or chain_mask.dtype.kind != "b":
        raise ValueError("mask must be a boolean per-chain validity mask")
    name_shape = (len(atoms),) if native_names else (len(atoms), 4)
    if atoms["name"].shape != name_shape or atoms["coords"].shape != (len(atoms), 3):
        raise ValueError("invalid atom name/coordinate dimensions")
    # Boltz-1 NPZ may have only atoms.coords. Newer schema duplicates them in
    # coords + ensemble. Ambiguous or inconsistent duplicate sources fail closed.
    if ("coords" in data) != ("ensemble" in data):
        raise ValueError("coords and ensemble must be supplied together")
    if "ensemble" in data:
        ensemble = _table(data, "ensemble", dict(atom_coord_idx="iu", atom_num="iu"))
        coords = _table(data, "coords", dict(coords="f"))
        if (len(ensemble) != 1 or int(ensemble[0]["atom_coord_idx"]) != 0
                or int(ensemble[0]["atom_num"]) != len(atoms) or len(coords) != len(atoms)):
            raise ValueError("only a single complete coordinate model is supported")
        present = atoms["is_present"]
        if not np.array_equal(coords["coords"][present], atoms["coords"][present]):
            raise ValueError("atoms.coords and ensemble coords disagree")
    if not np.isfinite(atoms["coords"][atoms["is_present"]]).all():
        raise ValueError("nonfinite observed coordinates")
    endpoint_fields = dict(chain_1="iu", chain_2="iu", res_1="iu", res_2="iu", atom_1="iu", atom_2="iu")
    if native_names:
        bonds = _table(data, "bonds", dict(**endpoint_fields, type="iu"))
        connections = (_table(data, "connections", endpoint_fields) if "connections" in data
                       else np.empty(0, dtype=[(k, "i4") for k in endpoint_fields]))
    else:
        connections = _table(data, "connections", endpoint_fields)
        bonds = _table(data, "bonds", dict(atom_1="iu", atom_2="iu", type="iu"))
    return atoms, residues, chains, chain_mask, native_names, source_schema, endpoint_fields, bonds, connections


def adapt_boltz_npz(
    path, *, sample_id, source_pdb_id, receptor_chain_ids, peptide_chain_id,
    split="smoke", parameters=None, target_residue_range=None, source=None,
):
    """Create an embedded-label record accepted by collate_joint_v2_records.

    Receptor residues without N/CA/C are omitted with provenance; their original
    polymer positions still prevent false adjacency. The generated region must
    be complete and continuous. By default this is the entire source chain;
    target_residue_range=(start, stop) selects a contiguous zero-based, half-open
    polymer interval, retaining every residue between its endpoints.
    Only single-model NPZs are accepted because the format
    does not provide independent atom-presence masks for each ensemble member.
    """
    parameters = parameters or PocketParameters()
    if not all(isinstance(v, str) and v for v in (sample_id, source_pdb_id, split, peptide_chain_id)):
        raise ValueError("sample/source/split/peptide identities must be nonempty strings")
    if isinstance(receptor_chain_ids, str):
        raise ValueError("receptor_chain_ids must be a sequence of exact chain names")
    roles = tuple(receptor_chain_ids)
    if not roles or len(set(roles)) != len(roles) or peptide_chain_id in roles:
        raise ValueError("receptor roles must be unique and disjoint from peptide")
    path = Path(path).resolve()
    if source is None:
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
    else:
        source.check_path(path)
        data = source.data
    validated = source.memo.get('validated_tables') if source is not None else None
    if validated is None:
        validated = _validated_tables(data)
        if source is not None:
            source.memo['validated_tables'] = validated
    atoms, residues, chains, chain_mask, native_names, source_schema, endpoint_fields, bonds, connections = validated
    selected = {}
    for name in (*roles, peptide_chain_id):
        hits = np.flatnonzero(chains["name"] == name)
        if len(hits) != 1:
            raise ValueError(f"chain name must identify exactly one assembly copy: {name}")
        ci = int(hits[0])
        if not chain_mask[ci] or int(chains[ci]["mol_type"]) != 0:
            raise ValueError(f"selected chain is masked or is not protein: {name}")
        selected[name] = ci
    if len({int(chains[i]["asym_id"]) for i in selected.values()}) != len(selected):
        raise ValueError("selected assembly chains have duplicate asym_id")

    target_chain = chains[selected[peptide_chain_id]]
    if "cyclic_period" in chains.dtype.names and int(target_chain["cyclic_period"]) > 0:
        raise ValueError("declared cyclic target chain is outside the linear peptide contract")
    target_length = int(target_chain["res_num"])
    if target_residue_range is None:
        target_start, target_stop = 0, target_length
    else:
        if (not isinstance(target_residue_range, (tuple, list)) or len(target_residue_range) != 2
                or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                       for v in target_residue_range)):
            raise ValueError("target_residue_range must contain two integer polymer positions")
        target_start, target_stop = map(int, target_residue_range)
        if not 0 <= target_start < target_stop <= target_length:
            raise ValueError("target_residue_range outside source chain")

    peptide, receptor, omitted, chirality = [], [], [], []
    # Each entry retains global source row identity, never cropped-array identity.
    provenance = {}
    peptide_source_atoms = {}
    used_atoms, used_residues = set(), set()
    for chain_name, ci in selected.items():
        cache_key = ('receptor', ci)
        cached = source.memo.get(cache_key) if source is not None and chain_name != peptide_chain_id else None
        if cached is not None:
            saved_residues, saved_keys, saved_omitted, saved_atoms, saved_rows = cached
            if used_atoms.intersection(saved_atoms) or used_residues.intersection(saved_rows):
                raise ValueError("overlapping residue span or duplicate atom identity")
            receptor.extend(saved_residues)
            provenance.update(saved_keys)
            omitted.extend(saved_omitted)
            used_atoms.update(saved_atoms)
            used_residues.update(saved_rows)
            continue
        before_receptor, before_omitted = len(receptor), len(omitted)
        before_atoms, before_rows = used_atoms.copy(), used_residues.copy()
        chain = chains[ci]
        crange = _range(chain["atom_idx"], chain["atom_num"], len(atoms), "chain atom")
        rrange = _range(chain["res_idx"], chain["res_num"], len(residues), "chain residue")
        expected_atom = crange.start
        for position, ri in enumerate(rrange):
            r = residues[ri]
            if ri in used_residues or int(r["res_idx"]) != position:
                raise ValueError("overlapping residue span or noncontiguous Boltz polymer indices")
            used_residues.add(ri)
            arange = _range(r["atom_idx"], r["atom_num"], len(atoms), "residue atom")
            if arange.start != expected_atom or arange.stop > crange.stop:
                raise ValueError("residue atom span does not partition its parent chain")
            expected_atom = arange.stop
            if chain_name == peptide_chain_id and not target_start <= position < target_stop:
                continue
            aa = canonical_aatype(str(r["name"]), r["res_type"])
            if not r["is_standard"]:
                raise ValueError(f"nonstandard residue chemistry: {chain_name}/{position}")
            names, observed = set(), []
            for ai in arange:
                atom = atoms[ai]
                name = decode_atom_name(atom["name"])
                if name in names or ai in used_atoms:
                    raise ValueError(f"duplicate atom identity: {chain_name}/{position}/{name}")
                names.add(name)
                used_atoms.add(ai)
                if chain_name == peptide_chain_id:
                    peptide_source_atoms[ai] = (position, name)
                if name not in set(ATOM14_NAMES[aa]) | {"OXT"}:
                    raise ValueError(f"unsupported atom for {r['name']}: {name}")
                # This inference is allowed only AFTER the canonical residue and
                # residue-specific atom-name template have both been validated.
                element = name[0]
                if "element" in atoms.dtype.names and ELEMENTS.get(int(atom["element"])) != element:
                    raise ValueError(f"atom element/name mismatch: {name}/{atom['element']}")
                if atom["is_present"]:
                    observed.append(AtomRecord(name, element, "", 1., 0., ai,
                                               tuple(float(v) for v in atom["coords"])))
            if not r["is_present"] and observed:
                raise ValueError("residue is absent but contains observed atoms")
            key = ResidueKey(chain_name, position, "", f"boltz_asym:{int(chain['asym_id'])}",
                             None, position, polymer_index=position, index_source="boltz_res_idx_zero_based")
            residue = ResidueRecord(key, str(r["name"]), "Polymer", tuple(observed))
            provenance[id(residue)] = _key(residue, chains, ri, ci)
            if not {"N", "CA", "C"} <= {a.name for a in observed}:
                if chain_name == peptide_chain_id:
                    raise ValueError(f"peptide missing observed backbone: {chain_name}/{position}")
                omitted.append(dict(**provenance[id(residue)], reason="missing_observed_backbone"))
                continue
            (peptide if chain_name == peptide_chain_id else receptor).append(residue)
        if expected_atom != crange.stop:
            raise ValueError("residue atoms do not exhaust their chain span")
        if source is not None and chain_name != peptide_chain_id:
            saved = receptor[before_receptor:]
            source.memo[cache_key] = (saved, {id(r): provenance[id(r)] for r in saved},
                omitted[before_omitted:], used_atoms - before_atoms, used_residues - before_rows)
    if not peptide or not receptor:
        raise ValueError("empty peptide or observed receptor")
    links, breaks = backbone_links(peptide, parameters.geometry)
    if breaks:
        raise ValueError(f"peptide backbone discontinuity: {breaks[0]}")
    validate_link_angles(peptide, links, parameters.geometry)

    peptide_atoms = {a.serial for r in peptide for a in r.atoms}
    boundary_bonds = []

    def source_target_atom(atom_row):
        """Resolve a boundary bond against the ORIGINAL chain, not packed slots."""
        r0 = int(target_chain["res_idx"])
        rr = residues[r0:r0 + target_length]
        position = int(np.searchsorted(rr["atom_idx"], atom_row, side="right")) - 1
        if (position < 0 or position >= target_length
                or not int(rr[position]["atom_idx"]) <= atom_row
                < int(rr[position]["atom_idx"] + rr[position]["atom_num"])):
            return None
        return position, decode_atom_name(atoms[atom_row]["name"])

    # Explicit crosslinks, even to an omitted/masked chain, cannot be represented
    # by the current linear peptide contract. Standard sequential C-N is allowed.
    for table in (connections, bonds):
        for connection in table:
            ai, aj = int(connection["atom_1"]), int(connection["atom_2"])
            if not (0 <= ai < len(atoms) and 0 <= aj < len(atoms)):
                raise ValueError("bond/connection atom index out of bounds")
            if set(endpoint_fields) <= set(table.dtype.names):
                for side, atom_row in ((1, ai), (2, aj)):
                    residue_row = int(connection[f"res_{side}"])
                    chain_row = int(connection[f"chain_{side}"])
                    if not 0 <= residue_row < len(residues) or not 0 <= chain_row < len(chains):
                        raise ValueError("invalid connection residue/chain identity")
                    r, c = residues[residue_row], chains[chain_row]
                    if (not int(r["atom_idx"]) <= atom_row < int(r["atom_idx"]) + int(r["atom_num"])
                            or not int(c["res_idx"]) <= residue_row < int(c["res_idx"]) + int(c["res_num"])):
                        raise ValueError("connection atom/residue/chain identities disagree")
            if ai not in peptide_source_atoms and aj not in peptide_source_atoms:
                continue
            left, right = peptide_source_atoms.get(ai), peptide_source_atoms.get(aj)
            if target_residue_range is not None and (left is None or right is None):
                full_left, full_right = source_target_atom(ai), source_target_atom(aj)
                if full_left is not None and full_right is not None:
                    pi, ni = full_left
                    pj, nj = full_right
                    if (ni, nj, pj - pi) in {("C", "N", 1), ("N", "C", -1)}:
                        boundary_bonds.append([ai, aj])
                        continue
            if left is not None and right is not None:
                pi, ni = left
                pj, nj = right
                if pi == pj and table is bonds:
                    raise ValueError("unexpected explicit bond chemistry in a standard peptide residue")
                if (ni, nj, pj - pi) in {("C", "N", 1), ("N", "C", -1)}:
                    continue
            raise ValueError("unsupported peptide covalent crosslink")
    if len(peptide) >= 3:
        end_distance = np.linalg.norm(_xyz(peptide[-1], "C") - _xyz(peptide[0], "N"))
        if parameters.geometry.min_cn_angstrom <= end_distance <= parameters.geometry.max_cn_angstrom:
            raise ValueError("possible cyclic peptide is outside the linear peptide contract")

    peptide_xyz = np.asarray([a.xyz for r in peptide for a in r.atoms])
    receptor_geometry_key = ('receptor_geometry', tuple(selected[n] for n in roles))
    receptor_geometry = source.memo.get(receptor_geometry_key) if source is not None else None
    if source is not None and receptor_geometry is None:
        receptor_geometry = (np.asarray([a.xyz for r in receptor for a in r.atoms]),
            np.cumsum([0] + [len(r.atoms) for r in receptor])[:-1],
            np.stack([_cb(r) for r in receptor]))
        source.memo[receptor_geometry_key] = receptor_geometry
    if source is None:
        core = np.asarray([np.any(np.linalg.norm(np.asarray([a.xyz for a in r.atoms])[:, None]
                             - peptide_xyz[None], axis=-1) <= parameters.core_cutoff_angstrom)
                           for r in receptor])
    else:
        peptide_tree = cKDTree(peptide_xyz)
        receptor_xyz, starts, _ = receptor_geometry
        nearest = peptide_tree.query(receptor_xyz, workers=1)[1]
        touches = np.linalg.norm(receptor_xyz - peptide_xyz[nearest], axis=-1) <= parameters.core_cutoff_angstrom
        core = np.logical_or.reduceat(touches, starts)
    if not core.any():
        raise ValueError("selected receptor has no peptide contact within core cutoff")
    cb = (receptor_geometry[2] if receptor_geometry is not None else np.stack([_cb(r) for r in receptor]))
    distances = np.linalg.norm(cb[:, None] - cb[core][None], axis=-1).min(axis=1)
    keep = core | (distances <= parameters.context_radius_angstrom)
    origin = np.stack([_xyz(r, "CA") for r, yes in zip(receptor, core) if yes]).mean(axis=0)
    pocket = [r for r, yes in zip(receptor, keep) if yes]
    receptor_ids = {id(r) for r in receptor}
    for r in [*pocket, *peptide]:
        geometry_key = ('geometry', id(r), parameters.geometry)
        status = source.memo.get(geometry_key) if source is not None and id(r) in receptor_ids else None
        if status is None:
            validate_elements(r)
            status = validate_residue_geometry(r, parameters.geometry)
            if source is not None and id(r) in receptor_ids:
                source.memo[geometry_key] = status
        chirality.append(dict(key=provenance[id(r)], status=status))
    pocket_links, pocket_breaks = backbone_links(pocket, parameters.geometry)
    validate_link_angles(pocket, pocket_links, parameters.geometry)

    def pack(residue_list, peptide_mode):
        slots = 14 if peptide_mode else len(POCKET_ATOM_NAMES)
        xyz = np.zeros((len(residue_list), slots, 3), dtype=np.float32)
        mask = np.zeros((len(residue_list), slots), dtype=bool)
        source_indices = np.full((len(residue_list), slots), -1, dtype=np.int64)
        excluded = []
        for i, r in enumerate(residue_list):
            names = ATOM14_NAMES[AA3_TO_INDEX[r.name]] if peptide_mode else POCKET_ATOM_NAMES
            for atom in r.atoms:
                if peptide_mode and atom.name == "OXT":
                    excluded.append(dict(residue_row=provenance[id(r)]["boltz_residue_row"],
                                         atom_row=atom.serial, name="OXT", reason="outside_atom14"))
                    continue
                slot = names.index(atom.name)
                xyz[i, slot] = np.asarray(atom.xyz, dtype=np.float64) - origin
                mask[i, slot] = True
                source_indices[i, slot] = atom.serial
        return xyz, mask, source_indices, excluded

    p_xyz, p_mask, p_indices, _ = pack(pocket, False)
    q_xyz, q_mask, q_indices, excluded = pack(peptide, True)
    frames = observed_backbone_frames(torch.from_numpy(p_xyz[:, :3]), torch.from_numpy(p_mask[:, :3]))
    q_frames = observed_backbone_frames(torch.from_numpy(q_xyz[:, :3]), torch.from_numpy(q_mask[:, :3]))
    torsions, _ = extract_backbone_torsions(*(torch.from_numpy(q_xyz[:, i]) for i in range(3)))
    target = dict(schema_version=NATIVE_TARGET_SCHEMA, sample_id=sample_id, peptide_length=len(peptide),
        aatype=np.asarray([AA3_TO_INDEX[r.name] for r in peptide], dtype=np.int64),
        translation=q_frames.translation.numpy(), rotation=q_frames.rotation.numpy(),
        backbone_torsion=torsions.numpy(), experimental_atom14=q_xyz, experimental_atom14_mask=q_mask,
        supervision_source="observed_boltz_processed_structure",
        joint_v2_contract_sha256=JOINT_V2_CONTRACT_SHA256)
    # Explicitly disclose other nearby source atoms that are outside the model.
    represented = set(p_indices[p_indices >= 0].tolist()) | peptide_atoms
    nearby_excluded = []
    if source is None:
        for start in range(0, len(atoms), 1024):
            rows = np.arange(start, min(start + 1024, len(atoms)))
            rows = rows[atoms["is_present"][rows]]
            near = np.any(np.linalg.norm(atoms["coords"][rows, None].astype(np.float64)
                                        - peptide_xyz[None], axis=-1) <= parameters.environment_radius_angstrom, axis=1)
            nearby_excluded.extend(int(i) for i in rows[near] if int(i) not in represented)
    else:
        spatial = source.memo.get('observed_atom_tree')
        if spatial is None:
            observed_rows = np.flatnonzero(atoms['is_present'])
            observed_xyz = atoms['coords'][observed_rows].astype(np.float64)
            spatial = (observed_rows, observed_xyz, cKDTree(observed_xyz))
            source.memo['observed_atom_tree'] = spatial
        observed_rows, observed_xyz, tree = spatial
        groups = tree.query_ball_point(peptide_xyz,
            np.nextafter(parameters.environment_radius_angstrom, np.inf), workers=1)
        candidate = sorted({i for group in groups for i in group})
        if candidate:
            candidate = np.asarray(candidate)
            nearest = peptide_tree.query(observed_xyz[candidate], workers=1)[1]
            near = np.linalg.norm(observed_xyz[candidate] - peptide_xyz[nearest], axis=-1) <= parameters.environment_radius_angstrom
            nearby_excluded = [int(i) for i in observed_rows[candidate[near]] if int(i) not in represented]
    record = dict(schema_version=RECORD_SCHEMA_VERSION, complex_schema_version=COMPLEX_RECORD_SCHEMA,
        sample_id=sample_id, source_pdb_id=source_pdb_id, split=split, peptide_length=len(peptide),
        receptor_chain_id=",".join(roles), receptor_chain_ids=list(roles), peptide_chain_id=peptide_chain_id,
        raw_path=str(path), raw_file_sha256=(source.sha256 if source is not None
            else hashlib.sha256(path.read_bytes()).hexdigest()),
        site_origin=origin, selected_model_index=0, preprocessing_parameters=asdict(parameters),
        pocket_residue_keys=[provenance[id(r)] for r in pocket],
        peptide_residue_keys=[provenance[id(r)] for r in peptide],
        pocket_original_residue_names=[r.name for r in pocket], pocket_normalized_residue_names=[r.name for r in pocket],
        pocket_aatype=np.asarray([AA3_TO_INDEX[r.name] for r in pocket], dtype=np.int64),
        pocket_atom_xyz=p_xyz, pocket_atom_mask=p_mask, pocket_residue_translation=frames.translation.numpy(),
        pocket_residue_rotation=frames.rotation.numpy(), pocket_core_mask=core[keep],
        pocket_backbone_link_mask=np.asarray(pocket_links, dtype=bool),
        pocket_cb_to_core_distance=distances[keep].astype(np.float32), joint_v2_target=target,
        boltz_adapter=dict(schema=ADAPTER_SCHEMA, source_schema=source_schema,
            mapping_sha256=mapping_manifest()["mapping_sha256"],
            pocket_atom_source_indices=p_indices, peptide_atom_source_indices=q_indices,
            peptide_sequence="".join(AA1_ORDER[AA3_TO_INDEX[r.name]] for r in peptide),
            target_source_length=target_length, target_residue_range=[target_start, target_stop],
            target_is_complete_source_chain=(target_start == 0 and target_stop == target_length),
            cut_boundary_peptide_bonds=boundary_bonds,
            source_chain_roles={"receptor": [selected[n] for n in roles], "peptide": selected[peptide_chain_id]},
            author_numbering_available=False, coordinate_source="atoms.coords", coordinate_unit="angstrom",
            omitted_receptor_residues=omitted, excluded_peptide_atoms=excluded,
            nearby_excluded_atom_rows=nearby_excluded,
            source_limitations=["occupancy, altloc and author numbering cannot be recovered from this NPZ",
                               "chain roles are supplied, not inferred biological binder annotations"]),
        structure_quality=dict(pocket_backbone_breaks=pocket_breaks, chirality=chirality,
            warnings=(["nearby_full_structure_environment_not_in_condition"] if nearby_excluded else [])))
    return precompute_record(record)


def audit_boltz_record(record, *, source=None):
    """Independently round-trip every represented residue/atom to the source NPZ."""
    if source is None:
        with np.load(record["raw_path"], allow_pickle=False) as archive:
            atoms, residues, chains = archive["atoms"], archive["residues"], archive["chains"]
        digest = hashlib.sha256(Path(record["raw_path"]).read_bytes()).hexdigest()
    else:
        source.check_path(record["raw_path"])
        atoms, residues, chains = (source.data[key] for key in ("atoms", "residues", "chains"))
        digest = source.sha256
    if digest != record["raw_file_sha256"]:
        raise ValueError("source NPZ changed after adaptation")
    if record["boltz_adapter"]["mapping_sha256"] != mapping_manifest()["mapping_sha256"]:
        raise ValueError("mapping contract changed after adaptation")
    maximum_error, atom_count, residue_count = 0., 0, 0
    for role in ("pocket", "peptide"):
        target = record if role == "pocket" else record["joint_v2_target"]
        xyz = target["pocket_atom_xyz" if role == "pocket" else "experimental_atom14"]
        mask = target["pocket_atom_mask" if role == "pocket" else "experimental_atom14_mask"]
        aa = target["pocket_aatype" if role == "pocket" else "aatype"]
        indices = record["boltz_adapter"][f"{role}_atom_source_indices"]
        if not np.array_equal(indices >= 0, mask) or np.any(xyz[~mask] != 0):
            raise ValueError("source index/mask/zero-padding mismatch")
        for i, key in enumerate(record[f"{role}_residue_keys"]):
            residue_count += 1
            r, c = residues[key["boltz_residue_row"]], chains[key["boltz_chain_row"]]
            if (str(r["name"]) != AA3_ORDER[int(aa[i])]
                    or canonical_aatype(str(r["name"]), r["res_type"]) != int(aa[i])
                    or key["polymer_index"] != int(r["res_idx"])
                    or key["boltz_asym_id"] != int(c["asym_id"])
                    or key["boltz_entity_id"] != int(c["entity_id"])
                    or key["boltz_sym_id"] != int(c["sym_id"])
                    or key["boltz_chain_name"] != str(c["name"])
                    or not int(c["res_idx"]) <= key["boltz_residue_row"] < int(c["res_idx"]) + int(c["res_num"])):
                raise ValueError("residue/chain identity round-trip failed")
            names = POCKET_ATOM_NAMES if role == "pocket" else ATOM14_NAMES[int(aa[i])]
            for slot in np.flatnonzero(mask[i]):
                ai = int(indices[i, slot])
                if not int(r["atom_idx"]) <= ai < int(r["atom_idx"] + r["atom_num"]):
                    raise ValueError("atom mapped into another residue")
                if not atoms[ai]["is_present"] or decode_atom_name(atoms[ai]["name"]) != names[slot]:
                    raise ValueError("atom name/presence round-trip failed")
                error = float(np.max(np.abs(xyz[i, slot].astype(np.float64) + record["site_origin"]
                                           - atoms[ai]["coords"].astype(np.float64))))
                maximum_error = max(maximum_error, error)
                atom_count += 1
    if maximum_error > 1e-4:
        raise ValueError(f"coordinate round-trip failed: {maximum_error} angstrom")
    return dict(sample_id=record["sample_id"], residues_checked=residue_count, atoms_checked=atom_count,
                max_coordinate_error_angstrom=maximum_error, sequence=record["boltz_adapter"]["peptide_sequence"],
                pocket_length=len(record["pocket_aatype"]), peptide_length=record["peptide_length"],
                pocket_chains=sorted({k["boltz_chain_name"] for k in record["pocket_residue_keys"]}),
                source_sha256=record["raw_file_sha256"], mapping_sha256=record["boltz_adapter"]["mapping_sha256"])
