"""Identity-sensitive tests: permuted source atoms, absent atoms and broken schemas."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from apexgen.joint_v2.data.boltz_npz import (
    adapt_boltz_npz, audit_boltz_record, canonical_aatype, decode_atom_name,
)
from apexgen.joint_v2.data.batch import collate_joint_v2_records
from apexgen.shared.geometry.joint_residue_constants import AA3_ORDER, ATOM14_NAMES
from apexgen.shared.geometry.sidechain import build_atom14


ATOM_DTYPE = np.dtype([("name", "i1", (4,)), ("element", "i1"), ("coords", "f4", (3,)),
                       ("conformer", "f4", (3,)), ("is_present", "?")])
RES_DTYPE = np.dtype([(n, t) for n, t in [
    ("name", "U5"), ("res_type", "i1"), ("res_idx", "i4"), ("atom_idx", "i4"),
    ("atom_num", "i4"), ("is_standard", "?"), ("is_present", "?")]])
CHAIN_DTYPE = np.dtype([(n, t) for n, t in [
    ("name", "U5"), ("mol_type", "i1"), ("entity_id", "i4"), ("sym_id", "i4"),
    ("asym_id", "i4"), ("atom_idx", "i4"), ("atom_num", "i4"), ("res_idx", "i4"), ("res_num", "i4")]])
CONNECTION_DTYPE = np.dtype([(n, "i4") for n in ("chain_1", "chain_2", "res_1", "res_2", "atom_1", "atom_2")])


def encoded(name):
    return [ord(c) - 32 for c in name] + [0] * (4 - len(name))


def source_arrays(peptide_aa=0):
    # Ideal local residue geometry is used only to make test fixtures, never to
    # construct production supervision. Every chain has one residue here.
    atoms, residues, chains = [], [], []
    for ci, (name, aa, offset) in enumerate([("A1", 0, [0, 3, 0]),
                                            ("B1", 7, [0, -3, 0]),
                                            ("P1", peptide_aa, [0, 0, 0])]):
        bb = torch.tensor([[[-0.525, 1.363, 0.], [0., 0., 0.], [1.526, 0., 0.],
                            [2.153, -1.062, 0.]]])
        xyz, mask = build_atom14(bb, torch.tensor([aa]), torch.zeros(1, 4))
        start = len(atoms)
        # Reverse ordering deliberately: never let passing tests depend on
        # Boltz and OpenFold having the same slot ordering.
        for slot in reversed(np.flatnonzero(mask[0].numpy()).tolist()):
            atom_name = ATOM14_NAMES[aa][slot]
            atoms.append((encoded(atom_name), {"C": 6, "N": 7, "O": 8, "S": 16}[atom_name[0]],
                          xyz[0, slot].numpy() + offset, [999., 999., 999.], True))
        count = len(atoms) - start
        residues.append((AA3_ORDER[aa], aa + 2, 0, start, count, True, True))
        # asym_id, entity_id and chain-row intentionally differ.
        chains.append((name, 0, 7 if ci < 2 else 9, ci, 10 + ci, start, count, ci, 1))
    atoms = np.array(atoms, dtype=ATOM_DTYPE)
    return dict(atoms=atoms, residues=np.array(residues, dtype=RES_DTYPE),
                chains=np.array(chains, dtype=CHAIN_DTYPE), mask=np.ones(3, dtype=bool),
                bonds=np.empty(0, dtype=[("atom_1", "i4"), ("atom_2", "i4"), ("type", "i1")]),
                connections=np.empty(0, dtype=CONNECTION_DTYPE),
                coords=np.array([(tuple(a["coords"]),) for a in atoms], dtype=[("coords", "f4", (3,))]),
                ensemble=np.array([(0, len(atoms))], dtype=[("atom_coord_idx", "i4"), ("atom_num", "i4")]))


def adapt(tmp_path, arrays):
    path = tmp_path / "fixture.npz"
    np.savez(path, **arrays)
    return adapt_boltz_npz(path, sample_id="fixture", source_pdb_id="synthetic",
                           receptor_chain_ids=("A1", "B1"), peptide_chain_id="P1")


@pytest.mark.parametrize("aa", range(20))
def test_all_twenty_residues_reordered_by_atom_identity(tmp_path, aa):
    arrays = source_arrays(aa)
    record = adapt(tmp_path, arrays)
    audit = audit_boltz_record(record)
    assert record["joint_v2_target"]["aatype"].tolist() == [aa]
    assert audit["max_coordinate_error_angstrom"] < 1e-5
    for slot, name in enumerate(ATOM14_NAMES[aa]):
        mask = record["joint_v2_target"]["experimental_atom14_mask"][0, slot]
        assert bool(mask) == bool(name)
        if name:
            atom_row = record["boltz_adapter"]["peptide_atom_source_indices"][0, slot]
            assert decode_atom_name(arrays["atoms"][atom_row]["name"]) == name
    batch = collate_joint_v2_records([record])
    assert batch.condition.chain_index[0].tolist() == [0, 1, 2]
    assert not record["pocket_backbone_link_mask"].any()
    assert batch.condition.aatype[batch.condition.peptide_mask].tolist() == [20]
    assert not batch.condition.pocket_atom_mask[batch.condition.peptide_mask].any()
    assert batch.targets.endpoint_aatype[batch.condition.peptide_mask].tolist() == [aa]
    assert "auth_seq_id" not in record["peptide_residue_keys"][0]
    assert record["peptide_residue_keys"][0]["boltz_asym_id"] == 12


def test_fixed_external_vocabulary():
    # Includes late alphabet entries whose indexes differ in alphabetical/ESM order.
    assert canonical_aatype("ALA", 2) == 0
    assert canonical_aatype("ARG", 3) == 1
    assert canonical_aatype("GLY", 9) == 7
    assert canonical_aatype("TRP", 19) == 17
    assert canonical_aatype("VAL", 21) == 19
    with pytest.raises(ValueError, match="mismatch"):
        canonical_aatype("ALA", 0)
    with pytest.raises(ValueError, match="unsupported"):
        canonical_aatype("MSE", 14)


@pytest.mark.parametrize("encoded_name", [[46, 0, 33, 0], [0, 0, 0, 0], [-1, 0, 0, 0], [46, 33]])
def test_invalid_atom_encoding(encoded_name):
    with pytest.raises(ValueError):
        decode_atom_name(encoded_name)


def test_absent_sidechain_does_not_become_an_observation(tmp_path):
    arrays = source_arrays(1)
    ri = arrays["residues"][-1]
    ai = next(i for i in range(int(ri["atom_idx"]), len(arrays["atoms"]))
              if decode_atom_name(arrays["atoms"][i]["name"]) == "NH2")
    arrays["atoms"][ai]["is_present"] = False
    arrays["atoms"][ai]["coords"] = np.nan
    record = adapt(tmp_path, arrays)
    slot = ATOM14_NAMES[1].index("NH2")
    assert not record["joint_v2_target"]["experimental_atom14_mask"][0, slot]
    assert np.all(record["joint_v2_target"]["experimental_atom14"][0, slot] == 0)
    assert record["boltz_adapter"]["peptide_atom_source_indices"][0, slot] == -1
    audit_boltz_record(record)


@pytest.mark.parametrize("failure", ["residue_token", "duplicate_atom", "wrong_element", "chain_mask",
    "missing_backbone", "bad_span", "residue_index", "ensemble", "coordinates", "crosslink"])
def test_rejects_identity_and_topology_corruption(tmp_path, failure):
    arrays = source_arrays()
    start = int(arrays["residues"][-1]["atom_idx"])
    n_row = next(i for i in range(start, len(arrays["atoms"]))
                 if decode_atom_name(arrays["atoms"][i]["name"]) == "N")
    if failure == "residue_token":
        arrays["residues"][-1]["res_type"] = 0
    elif failure == "duplicate_atom":
        arrays["atoms"][start]["name"] = arrays["atoms"][start + 1]["name"]
    elif failure == "wrong_element":
        arrays["atoms"][n_row]["element"] = 6
    elif failure == "chain_mask":
        arrays["mask"][-1] = False
    elif failure == "missing_backbone":
        arrays["atoms"][n_row]["is_present"] = False
    elif failure == "bad_span":
        arrays["residues"][-1]["atom_idx"] += 1
    elif failure == "residue_index":
        arrays["residues"][-1]["res_idx"] = 9
    elif failure == "ensemble":
        arrays["ensemble"] = np.repeat(arrays["ensemble"], 2)
    elif failure == "coordinates":
        arrays["coords"][n_row]["coords"] += 1
    elif failure == "crosslink":
        arrays["connections"] = np.array([(12, 10, 2, 0, n_row, 0)], dtype=CONNECTION_DTYPE)
    with pytest.raises(ValueError):
        adapt(tmp_path, arrays)


def test_old_schema_and_batch_padding_and_lmdb_serialization(tmp_path):
    from apexgen.shared.storage.store import pack_record, unpack_record
    arrays = source_arrays(17)
    del arrays["coords"], arrays["ensemble"]
    record = adapt(tmp_path, arrays)
    restored = unpack_record(pack_record(record))
    audit_boltz_record(restored)
    smaller = deepcopy(record)
    for key in ("pocket_aatype", "pocket_atom_xyz", "pocket_atom_mask", "pocket_residue_translation",
                "pocket_residue_rotation", "pocket_core_mask", "pocket_residue_keys"):
        smaller[key] = smaller[key][:1]
    smaller["pocket_backbone_link_mask"] = np.zeros(0, dtype=bool)
    from apexgen.joint_v2.data.static_features import precompute_record
    precompute_record(smaller)  # Re-cropping invalidates the original cache.
    batch = collate_joint_v2_records([restored, smaller])
    assert batch.condition.residue_mask.tolist() == [[True, True, True], [True, True, False]]
    assert batch.condition.chain_index[1, :2].tolist() == [0, 1]


def test_audit_detects_atom14_slot_swap(tmp_path):
    record = adapt(tmp_path, source_arrays(1))
    indices = record["boltz_adapter"]["peptide_atom_source_indices"]
    indices[0, [0, 1]] = indices[0, [1, 0]]
    with pytest.raises(ValueError, match="atom name"):
        audit_boltz_record(record)


def test_peptide_chain_break_is_not_concatenated(tmp_path):
    arrays = source_arrays()
    residue = arrays["residues"][-1].copy()
    extra = arrays["atoms"][int(residue["atom_idx"]):].copy()
    extra["coords"] += 50
    residue["atom_idx"] = len(arrays["atoms"])
    residue["res_idx"] = 1
    arrays["atoms"] = np.concatenate([arrays["atoms"], extra])
    arrays["residues"] = np.concatenate([arrays["residues"], np.array([residue], dtype=RES_DTYPE)])
    arrays["chains"][-1]["atom_num"] *= 2
    arrays["chains"][-1]["res_num"] = 2
    del arrays["coords"], arrays["ensemble"]
    with pytest.raises(ValueError, match="backbone discontinuity"):
        adapt(tmp_path, arrays)


def test_missing_receptor_backbone_and_atom14_oxt_policy(tmp_path):
    arrays = source_arrays()
    arrays["atoms"]["is_present"][:int(arrays["residues"][0]["atom_num"])] = False
    arrays["residues"][0]["is_present"] = False
    carbon = next(a for a in arrays["atoms"][int(arrays["residues"][-1]["atom_idx"]):]
                  if decode_atom_name(a["name"]) == "C")
    extra = np.array([(encoded("OXT"), 8, carbon["coords"] + [0, 1.25, 0], [999]*3, True)], dtype=ATOM_DTYPE)
    arrays["atoms"] = np.concatenate([arrays["atoms"], extra])
    arrays["residues"][-1]["atom_num"] += 1
    arrays["chains"][-1]["atom_num"] += 1
    del arrays["coords"], arrays["ensemble"]
    record = adapt(tmp_path, arrays)
    assert len(record["boltz_adapter"]["omitted_receptor_residues"]) == 1
    assert record["boltz_adapter"]["excluded_peptide_atoms"][0]["name"] == "OXT"
    assert len(record["pocket_aatype"]) == 1
    audit_boltz_record(record)


def test_panel_builder_produces_native_loader_dataset(tmp_path):
    from scripts.data.prepare_joint_v2_boltz_npz import build_panel
    from apexgen.joint_v2.data.dataset import JointV2Dataset
    from apexgen.joint_v2.runtime.lineage import joint_v2_dataset_identity
    source = tmp_path / "source.npz"
    np.savez(source, **source_arrays())
    rows = [dict(path=str(source), sample_id="test", source_pdb_id="synthetic",
                 receptor_chain_ids=["A1", "B1"], peptide_chain_id="P1")]
    output = tmp_path / "dataset"
    build_panel(rows, output)
    joint_v2_dataset_identity(output)
    dataset = JointV2Dataset(output, split="smoke")
    try:
        assert len(dataset) == 1
        record = dataset[0]
        assert audit_boltz_record(record)["pocket_chains"] == ["A1", "B1"]
        collate_joint_v2_records([record])
    finally:
        dataset.close()
    with pytest.raises(FileExistsError):
        build_panel(rows, output)
