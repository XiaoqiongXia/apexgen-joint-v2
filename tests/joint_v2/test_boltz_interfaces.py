import csv
import io
from pathlib import Path
import tarfile

import numpy as np
import pyarrow.parquet as pq
import pytest
from scipy.spatial.distance import cdist

from apexgen.joint_v2.data.boltz_interfaces import describe_interfaces


def fixture_arrays():
    atoms = np.array([
        ([0, 0, 0], 6, True), ([0, 1, 0], 7, True), ([10, 0, 0], 6, True),
        ([25, 0, 0], 6, False), ([5, 0, 0], 6, True), ([5, 0.1, 0], 8, True),
        ([100, 0, 0], 6, True), ([200, 0, 0], 6, True), ([0, 0, 0], 1, True),
    ], dtype=[("coords", "f4", (3,)), ("element", "i1"), ("is_present", "?")])
    residues = np.array([
        (0, 0, 2, True, True), (1, 2, 1, True, True), (2, 3, 1, False, True),
        (0, 4, 2, True, False), (1, 6, 1, True, True), (0, 7, 2, True, True),
    ], dtype=[("res_idx", "i4"), ("atom_idx", "i4"), ("atom_num", "i4"),
              ("is_present", "?"), ("is_standard", "?")])
    chains = np.array([
        ("A1", 0, 0, 0, 42, 0, 4, 0, 3), ("B1", 0, 0, 1, 17, 4, 3, 3, 2),
        ("C1", 0, 1, 0, 9, 7, 2, 5, 1),
    ], dtype=[("name", "U5"), ("mol_type", "i1"), ("entity_id", "i4"), ("sym_id", "i4"),
              ("asym_id", "i4"), ("atom_idx", "i4"), ("atom_num", "i4"),
              ("res_idx", "i4"), ("res_num", "i4")])
    return dict(atoms=atoms, residues=residues, chains=chains, mask=np.array([True, False, True]),
                interfaces=np.array([(0, 1), (1, 0), (0, 2)],
                                    dtype=[("chain_1", "i4"), ("chain_2", "i4")]))


def serialized(arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    payload = stream.getvalue()
    source = dict(source_archive="test.tar", source_member="structures/test.npz",
                  source_npz_filename="test.npz", source_structure_id="test",
                  source_offset=512, source_size=len(payload))
    return payload, source


def test_residue_counts_inclusive_cutoff_and_distinct_id_spaces():
    rows, audit = describe_interfaces(*serialized(fixture_arrays()))
    assert len(rows) == 2  # reverse duplicate collapsed; masked/nonstandard kept
    ab = rows[0]
    assert (ab["chain_a_id"], ab["chain_b_id"]) == ("A1", "B1")
    assert (ab["chain_a_asym_id"], ab["chain_b_asym_id"]) == (42, 17)
    assert (ab["chain_a_length"], ab["chain_b_length"]) == (3, 2)
    assert ab["chain_a_observed_residue_count"] == 2
    assert ab["chain_a_interface_residue_count"] == 2
    assert ab["chain_b_interface_residue_count"] == 1
    assert ab["chain_a_interface_residue_indices"] == [0, 1]
    assert ab["chain_b_interface_residue_indices"] == [0]
    assert ab["chain_b_interface_residue_rows"] == [3]
    assert ab["minimum_contact_distance_angstrom"] == 5.0
    assert ab["contact_verified"] and not ab["both_source_masks_true"]
    assert ab["chain_b_nonstandard_residue_count"] == 1
    assert not rows[1]["contact_verified"]  # a close hydrogen is not a heavy atom
    assert rows[1]["chain_a_interface_residue_count"] == 0
    assert rows[1]["minimum_contact_distance_angstrom"] is None
    assert audit["stored_interface_rows"] == 3


@pytest.mark.parametrize("cutoff", [0.5, 4.99999, 5.0, 5.1, 30.0])
def test_kdtree_matches_independent_brute_force(cutoff):
    arrays = fixture_arrays()
    rows, _ = describe_interfaces(*serialized(arrays), cutoff=cutoff)
    atom_to_residue = np.repeat(np.arange(len(arrays["residues"])), arrays["residues"]["atom_num"])
    for row in rows:
        identities, coordinates = [], []
        for side in ("a", "b"):
            chain = arrays["chains"][row[f"chain_{side}_row"]]
            ids = np.arange(chain["atom_idx"], chain["atom_idx"] + chain["atom_num"])
            keep = arrays["atoms"]["is_present"][ids] & (arrays["atoms"]["element"][ids] > 1)
            ids = ids[keep]
            identities.append(atom_to_residue[ids])
            coordinates.append(arrays["atoms"]["coords"][ids].astype(float))
        contacts = cdist(*coordinates) <= cutoff
        assert row["chain_a_interface_residue_rows"] == np.unique(identities[0][contacts.any(1)]).tolist()
        assert row["chain_b_interface_residue_rows"] == np.unique(identities[1][contacts.any(0)]).tolist()


def test_absent_atoms_do_not_count_and_polymer_indices_survive():
    arrays = fixture_arrays()
    arrays["atoms"]["is_present"][4:6] = False
    arrays["residues"]["res_idx"][:3] = [10, 20, 30]
    rows, _ = describe_interfaces(*serialized(arrays))
    assert not rows[0]["contact_verified"]
    arrays["atoms"]["is_present"][4] = True
    rows, _ = describe_interfaces(*serialized(arrays))
    assert rows[0]["chain_a_interface_residue_indices"] == [10, 20]


def test_nonprotein_partners_excluded_but_multimodel_is_annotated():
    arrays = fixture_arrays()
    arrays["chains"]["mol_type"][2] = 2
    arrays["ensemble"] = np.zeros(3, dtype=[("atom_coord_idx", "i4"), ("atom_num", "i4")])
    rows, _ = describe_interfaces(*serialized(arrays))
    assert len(rows) == 1 and rows[0]["ensemble_model_count"] == 3


@pytest.mark.parametrize("problem", ["bad_pair", "bad_span", "nan", "wrong_mask"])
def test_malformed_identity_or_coordinates_rejected(problem):
    arrays = fixture_arrays()
    if problem == "bad_pair":
        arrays["interfaces"][0]["chain_1"] = 42  # asym ID is not chain ROW
    elif problem == "bad_span":
        arrays["residues"][0]["atom_idx"] = 1
    elif problem == "nan":
        arrays["atoms"][0]["coords"] = np.nan
    else:
        arrays["mask"] = arrays["mask"].astype(int)
    with pytest.raises(ValueError):
        describe_interfaces(*serialized(arrays))


def test_indexed_tar_to_csv_parquet_with_source_error_accounting(tmp_path):
    from scripts.data.build_boltz_interface_manifest import build_manifest
    archive = tmp_path / "source.tar"
    payload, _ = serialized(fixture_arrays())
    with tarfile.open(archive, "w") as tar:
        for name, data in (("structures/test.npz", payload), ("structures/bad.npz", b"broken")):
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            tar.addfile(entry, io.BytesIO(data))
    members = tmp_path / "members.csv"
    with members.open("w", newline="") as f, tarfile.open(archive) as tar:
        writer = csv.DictWriter(f, fieldnames=["member", "offset", "size"])
        writer.writeheader()
        for entry in tar:
            writer.writerow(dict(member=entry.name, offset=entry.offset_data, size=entry.size))
    output = tmp_path / "manifest"
    summary = build_manifest(archive, members, output, workers=1, block_size=1)
    assert summary["counts"]["structures_attempted"] == 2
    assert summary["counts"]["structures_failed"] == 1
    assert summary["counts"]["pair_rows"] == 2
    table = pq.read_table(output / "interfaces.parquet")
    rows = list(csv.DictReader((output / "interfaces.csv").open()))
    assert len(rows) == table.num_rows == 2
    assert table["chain_a_interface_residue_indices"].to_pylist() == [[0, 1], []]
    assert len((output / "errors.jsonl").read_text().splitlines()) == 1
    with pytest.raises(FileExistsError):
        build_manifest(archive, members, output, workers=1)
    assert Path(table["source_archive"][0].as_py()) == archive.resolve()
