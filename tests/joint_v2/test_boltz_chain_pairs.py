"""Direction roles, source identity and continuous (not concatenated) fragments."""

from copy import deepcopy
import csv
import io
import json
import tarfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from apexgen.joint_v2.data.boltz_chain_pairs import adapt_interface_direction, fragment_selections
from apexgen.joint_v2.data.boltz_interfaces import PAIR_SCHEMA, describe_interfaces
from apexgen.joint_v2.data.boltz_npz import adapt_boltz_npz, audit_boltz_record, decode_atom_name
from apexgen.joint_v2.data.dataset import JointV2Dataset
from scripts.data.prepare_joint_v2_boltz_chain_pairs import build_dataset
from apexgen.shared.geometry.backbone import build_backbone
from test_boltz_npz import ATOM_DTYPE, CHAIN_DTYPE, RES_DTYPE, encoded, source_arrays


def make_pair(tmp_path, *, model_length=False, missing_second=False):
    arrays = source_arrays()
    if model_length:
        length = 4 if model_length is True else model_length
        bb = build_backbone(torch.tensor([[-1.0, -0.8, 3.14159265]] * length)).numpy()
        atoms, residues, chains = [], [], []
        for ci, name in enumerate(("A1", "B1", "P1")):
            a0, r0 = len(atoms), len(residues)
            for position in range(length):
                start = len(atoms)
                for slot, atom_name in enumerate(("N", "CA", "C", "O")):
                    shift = [100, 100, 100] if length == 9 and ci == 2 and position == 4 else [0, 0, ci * 1.5]
                    present = not (missing_second and ci == 2 and position == 5 and atom_name == "N")
                    atoms.append((encoded(atom_name), {"N": 7, "C": 6, "O": 8}[atom_name[0]],
                        bb[position, slot] + shift, [0, 0, 0], present))
                residues.append(("GLY", 9, position, start, 4, True, True))
            chains.append((name, 0, ci, 0, ci + 10, a0, length * 4, r0, length))
        arrays.update(atoms=np.array(atoms, dtype=ATOM_DTYPE), residues=np.array(residues, dtype=RES_DTYPE),
                      chains=np.array(chains, dtype=CHAIN_DTYPE))
        del arrays["coords"], arrays["ensemble"]
    arrays["interfaces"] = np.array([(0, 2)], dtype=[("chain_1", "i4"), ("chain_2", "i4")])
    path = tmp_path / "synthetic.npz"
    np.savez(path, **arrays)
    payload = path.read_bytes()
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as tar:
        member = tarfile.TarInfo("structures/synthetic.npz")
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))
    with tarfile.open(archive) as tar:
        member = tar.getmembers()[0]
    source = dict(source_archive=str(archive), source_member=member.name,
        source_npz_filename=path.name, source_structure_id="synthetic",
        source_offset=member.offset_data, source_size=len(payload))
    pairs, _ = describe_interfaces(payload, source)
    return path, pairs[0]


def test_both_directions_swap_roles_and_keep_original_source_identity(tmp_path):
    path, pair = make_pair(tmp_path)
    forward, _ = adapt_interface_direction(path, pair, "a_to_b", split="validation")
    reverse, _ = adapt_interface_direction(path, pair, "b_to_a", split="validation")
    assert forward["peptide_chain_id"] == "P1"
    assert reverse["peptide_chain_id"] == "A1"
    assert forward["receptor_chain_ids"] == ["A1"]
    assert reverse["receptor_chain_ids"] == ["P1"]
    assert forward["peptide_residue_keys"][0]["boltz_asym_id"] == 12
    assert reverse["peptide_residue_keys"][0]["boltz_asym_id"] == 10
    assert forward["split"] == reverse["split"] == "validation"
    assert forward["interface_pair"]["split_group"] == reverse["interface_pair"]["split_group"]


def test_all_contact_runs_are_not_an_envelope_or_concatenation():
    pair = dict(chain_b_interface_residue_indices=[2, 3, 7, 9], chain_b_length=14,
                chain_a_id="A1", chain_b_id="B1")
    results = fragment_selections(pair, "a_to_b")
    assert [(r["target_start"], r["target_stop"]) for r in results] == [(2, 4), (7, 8), (9, 10)]
    result = results[0]
    assert (result["target_start"], result["target_stop"], result["fragment_length"]) == (2, 4, 2)
    assert result["noninterface_residues_in_fragment"] == 0
    assert result["contact_runs"] == [[2, 4], [7, 8], [9, 10]]
    assert result["omitted_interface_residue_count"] == 2
    assert result["largest_internal_noncontact_gap"] == 3
    assert result["interface_density"] == 1.
    assert result["interface_coverage"] == .5
    assert not result["target_is_complete_source_chain"]


@pytest.mark.parametrize("positions,expected", [
    ([10, 11, 12, 30, 31], [(10, 13), (30, 32)]),
    ([2, 3, 8, 9], [(2, 4), (8, 10)]),
    ([0, 3, 4, 5, 6, 20], [(0, 1), (3, 7), (20, 21)]),
    ([4], [(4, 5)]),
])
def test_all_maximal_runs_including_equal_length(positions, expected):
    pair = dict(chain_a_interface_residue_indices=positions, chain_a_length=40,
                chain_a_id="A1", chain_b_id="B1")
    results = fragment_selections(pair, "b_to_a")
    assert [(r["target_start"], r["target_stop"]) for r in results] == expected


@pytest.mark.parametrize("field,value", [
    ("chain_b_length", 9), ("chain_b_asym_id", 2), ("source_npz_sha256", "0" * 64),
    ("chain_b_interface_residue_rows", [0]), ("contact_cutoff_angstrom", 8.),
])
def test_mismatched_inventory_is_rejected(tmp_path, field, value):
    path, pair = make_pair(tmp_path)
    pair[field] = value
    with pytest.raises(ValueError):
        adapt_interface_direction(path, pair, "a_to_b")


def doubled_target():
    arrays = source_arrays()
    residue = arrays["residues"][-1].copy()
    extra = arrays["atoms"][int(residue["atom_idx"]):].copy()
    # Both residues have valid local geometry, but no intervening peptide bond.
    extra["coords"] += [0, 0, 3]
    residue["atom_idx"] = len(arrays["atoms"])
    residue["res_idx"] = 1
    arrays["atoms"] = np.concatenate([arrays["atoms"], extra])
    arrays["residues"] = np.concatenate([arrays["residues"], np.array([residue], dtype=RES_DTYPE)])
    arrays["chains"][-1]["atom_num"] *= 2
    arrays["chains"][-1]["res_num"] = 2
    del arrays["coords"], arrays["ensemble"]
    return arrays


def test_explicit_fragment_ignores_missing_outside_but_keeps_original_indices(tmp_path):
    arrays = doubled_target()
    first = arrays["residues"][-2]
    arrays["atoms"]["is_present"][int(first["atom_idx"]):int(first["atom_idx"] + first["atom_num"])] = False
    arrays["residues"][-2]["is_present"] = False
    path = tmp_path / "source.npz"
    np.savez(path, **arrays)
    kwargs = dict(sample_id="fragment", source_pdb_id="synthetic", receptor_chain_ids=["A1"], peptide_chain_id="P1")
    record = adapt_boltz_npz(path, **kwargs, target_residue_range=(1, 2))
    assert record["peptide_length"] == 1
    assert record["peptide_residue_keys"][0]["polymer_index"] == 1
    assert record["peptide_residue_keys"][0]["boltz_residue_row"] == 3
    assert record["boltz_adapter"]["target_residue_range"] == [1, 2]
    audit_boltz_record(record)
    with pytest.raises(ValueError, match="missing observed backbone"):
        adapt_boltz_npz(path, **kwargs, target_residue_range=(0, 2))


def test_discontinuous_fragment_rejected_not_joined(tmp_path):
    path = tmp_path / "broken.npz"
    np.savez(path, **doubled_target())
    with pytest.raises(ValueError, match="backbone discontinuity"):
        adapt_boltz_npz(path, sample_id="broken", source_pdb_id="synthetic",
            receptor_chain_ids=["A1"], peptide_chain_id="P1", target_residue_range=(0, 2))


def test_cut_boundary_peptide_bond_allowed_but_crosslink_rejected(tmp_path):
    arrays = doubled_target()

    def atom_row(residue, name):
        return next(i for i in range(int(residue["atom_idx"]), int(residue["atom_idx"] + residue["atom_num"]))
                    if decode_atom_name(arrays["atoms"][i]["name"]) == name)

    left = atom_row(arrays["residues"][-2], "C")
    right = atom_row(arrays["residues"][-1], "N")
    arrays["bonds"] = np.array([(left, right, 1)], dtype=arrays["bonds"].dtype)
    path = tmp_path / "boundary.npz"
    np.savez(path, **arrays)
    kwargs = dict(sample_id="fragment", source_pdb_id="synthetic", receptor_chain_ids=["A1"],
                  peptide_chain_id="P1", target_residue_range=(1, 2))
    record = adapt_boltz_npz(path, **kwargs)
    assert record["boltz_adapter"]["cut_boundary_peptide_bonds"] == [[left, right]]
    arrays["bonds"][0]["atom_2"] = atom_row(arrays["residues"][-1], "CA")
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="unsupported peptide covalent crosslink"):
        adapt_boltz_npz(path, **kwargs)


@pytest.mark.parametrize("interval", [(-1, 1), (0, 3), (1, 1), (0, True), (0, 1, 2)])
def test_invalid_fragment_interval(tmp_path, interval):
    path = tmp_path / "source.npz"
    np.savez(path, **doubled_target())
    with pytest.raises(ValueError, match="target_residue_range"):
        adapt_boltz_npz(path, sample_id="bad", source_pdb_id="synthetic",
            receptor_chain_ids=["A1"], peptide_chain_id="P1", target_residue_range=interval)


def test_builder_loadable_shards_grouped_splits_and_rejection_log(tmp_path):
    path, pair = make_pair(tmp_path, model_length=True)
    inventory = tmp_path / "interfaces.parquet"
    pq.write_table(pa.Table.from_pylist([pair], schema=PAIR_SCHEMA), inventory)
    output = tmp_path / "dataset"
    result = build_dataset(inventory, output, split_map={"synthetic": "validation"}, shard_size=1)
    assert result["counts"] == dict(pairs_attempted=1, directions_attempted=2, contact_runs=2,
                                  skipped_no_contact=0, skipped_short=0, fragments_attempted=2, included=2, rejected=0)
    assert len(json.loads((output / "metadata.json").read_text())["shards"]) == 2
    dataset = JointV2Dataset(output, split="validation")
    try:
        assert len(dataset) == 2
        assert {dataset[i]["peptide_chain_id"] for i in range(2)} == {"P1", "A1"}
        for i in range(2):
            audit_boltz_record(dataset[i])
    finally:
        dataset.close()
    with pytest.raises(FileExistsError):
        build_dataset(inventory, output)
    with pytest.raises(ValueError, match="missing source-structure split"):
        build_dataset(inventory, tmp_path / "missing", split_map={"other": "train"})
    invalid = deepcopy(pair)
    invalid["contact_verified"] = False
    pq.write_table(pa.Table.from_pylist([invalid], schema=PAIR_SCHEMA), inventory)
    result = build_dataset(inventory, tmp_path / "rejected")
    assert result["counts"]["rejected"] == 2
    assert result["counts"]["included"] == 0
    log = [json.loads(line) for line in (tmp_path / "rejected" / "directions.jsonl").read_text().splitlines()]
    assert all(row["status"] == "rejected" and row["reason"] for row in log)


def test_builder_rejects_short_run_without_extending_it(tmp_path):
    _, pair = make_pair(tmp_path)
    inventory = tmp_path / "interfaces.parquet"
    pq.write_table(pa.Table.from_pylist([pair], schema=PAIR_SCHEMA), inventory)
    output = tmp_path / "short"
    result = build_dataset(inventory, output)
    assert result["counts"]["included"] == 0
    assert result["counts"]["skipped_short"] == 2
    log = [json.loads(line) for line in (output / "directions.jsonl").read_text().splitlines()]
    assert all(row["status"] == "skipped_short" and "fragment length < 4" in row["reason"] for row in log)
    assert all(row["selection"]["fragment_length"] == 1 for row in log)


@pytest.mark.parametrize("missing_second", [False, True])
def test_multiple_fragments_have_distinct_ids_and_independent_qc(tmp_path, missing_second):
    _, pair = make_pair(tmp_path, model_length=9, missing_second=missing_second)
    assert pair["chain_b_interface_residue_indices"] == [0, 1, 2, 3, 5, 6, 7, 8]
    inventory = tmp_path / "interfaces.parquet"
    pq.write_table(pa.Table.from_pylist([pair], schema=PAIR_SCHEMA), inventory)
    output = tmp_path / "multi"
    build_dataset(inventory, output, split_map={"synthetic": "train"})
    rows = pq.read_table(output / "manifest.parquet").to_pylist()
    forward = [r for r in rows if r["direction"] == "a_to_b"]
    assert [(r["target_start"], r["target_stop"]) for r in forward] == (
        [(0, 4)] if missing_second else [(0, 4), (5, 9)])
    assert len({r["sample_id"] for r in rows}) == len(rows)
    assert {r["split"] for r in rows} == {"train"}
    assert {r["split_group"] for r in rows} == {"synthetic"}
    dataset = JointV2Dataset(output, split="train")
    try:
        for i in range(len(dataset)):
            record = dataset[i]
            chosen = record["interface_pair"]
            assert [k["polymer_index"] for k in record["peptide_residue_keys"]] == list(
                range(chosen["target_start"], chosen["target_stop"]))
            audit_boltz_record(record)
    finally:
        dataset.close()
    if missing_second:
        log = [json.loads(line) for line in (output / "directions.jsonl").read_text().splitlines()]
        assert any(r["direction"] == "a_to_b" and r["selection"]["target_start"] == 5
                   and r["status"] == "rejected" and "missing observed backbone" in r["reason"] for r in log)


def test_three_residue_threshold_is_strict_by_default(tmp_path):
    _, pair = make_pair(tmp_path, model_length=True)
    # Exercise filtering at 3 vs 4 separately from geometrical checks.
    for side in ("a", "b"):
        pair[f"chain_{side}_interface_residue_indices"] = [0, 1, 2]
        start = pair[f"chain_{side}_residue_table_start"]
        pair[f"chain_{side}_interface_residue_rows"] = list(range(start, start + 3))
        pair[f"chain_{side}_interface_residue_count"] = 3
    inventory = tmp_path / "interfaces.parquet"
    pq.write_table(pa.Table.from_pylist([pair], schema=PAIR_SCHEMA), inventory)
    default = build_dataset(inventory, tmp_path / "default")
    assert default["counts"]["included"] == 0
    assert default["counts"]["skipped_short"] == 2
    relaxed = build_dataset(inventory, tmp_path / "relaxed", min_fragment_length=3)
    assert relaxed["counts"]["included"] == 2


def test_csv_visible_before_next_sample_and_interrupted_build_is_preserved(tmp_path, monkeypatch):
    from scripts.data import prepare_joint_v2_boltz_chain_pairs as builder
    from apexgen.shared.storage.store import unpack_record
    _, pair = make_pair(tmp_path, model_length=True)
    inventory = tmp_path/'interfaces.parquet'
    pq.write_table(pa.Table.from_pylist([pair], schema=PAIR_SCHEMA), inventory)
    output = tmp_path/'dataset'
    stage = tmp_path/'dataset.inprogress'
    original_adapt = builder.adapt_interface_direction
    original_open = builder.lmdb.open
    environments = []

    def capture_environment(*args, **kwargs):
        environment = original_open(*args, **kwargs)
        environments.append(environment)
        return environment

    calls = 0

    def interrupt_after_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            with (stage/'sample_inventory.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) == 1
            with environments[-1].begin() as txn:
                record = unpack_record(txn.get(rows[0]['tensor_key'].encode()))
            assert record['sample_id'] == rows[0]['sample_id']
            assert record['peptide_length'] == int(rows[0]['target_length'])
            raise KeyboardInterrupt('test interrupted after first committed sample')
        return original_adapt(*args, **kwargs)

    monkeypatch.setattr(builder.lmdb, 'open', capture_environment)
    monkeypatch.setattr(builder, 'adapt_interface_direction', interrupt_after_first)
    with pytest.raises(KeyboardInterrupt):
        builder.build_dataset(inventory, output)
    assert not output.exists() and stage.is_dir()
    status = json.loads((stage/'build_status.json').read_text())
    assert status['status'] == 'interrupted' and status['counts']['included'] == 1
    assert status['usable_for_training'] is False
    assert (stage/'shards/shard-00000.lmdb/data.mdb').is_file()
    # A repeat command must not erase the saved partial build.
    before = (stage/'sample_inventory.csv').read_bytes()
    with pytest.raises(FileExistsError):
        builder.build_dataset(inventory, output)
    assert (stage/'sample_inventory.csv').read_bytes() == before


def test_csv_write_failure_never_publishes_lmdb_as_complete(tmp_path, monkeypatch):
    from scripts.data import prepare_joint_v2_boltz_chain_pairs as builder
    _, pair = make_pair(tmp_path, model_length=True)
    inventory = tmp_path/'interfaces.parquet'
    pq.write_table(pa.Table.from_pylist([pair], schema=PAIR_SCHEMA), inventory)
    original = csv.DictWriter.writerow

    def fail_data_row(self, row):
        if 'tensor_key' in row and row.get('sample_id') != 'sample_id':
            raise OSError('injected CSV write failure')
        return original(self, row)

    monkeypatch.setattr(csv.DictWriter, 'writerow', fail_data_row)
    with pytest.raises(OSError, match='CSV write failure'):
        builder.build_dataset(inventory, tmp_path/'dataset')
    assert not (tmp_path/'dataset').exists()
    stage = tmp_path/'dataset.inprogress'
    assert json.loads((stage/'build_status.json').read_text())['status'] == 'failed'
    assert not (stage/'metadata.json').exists()
    # LMDB may contain the committed record: CSV is intentionally not claimed
    # to be in the same atomic transaction. Inspection/reconciliation is required.
    with builder.lmdb.open(str(stage/'shards/shard-00000.lmdb'), readonly=True, lock=False) as env:
        with env.begin() as txn:
            assert txn.stat()['entries'] == 1


def test_large_shard_map_growth_preserves_records_and_rejects_duplicates(tmp_path):
    from scripts.data import prepare_joint_v2_boltz_chain_pairs as builder
    with builder.lmdb.open(str(tmp_path/'large.lmdb'), map_size=32768) as env:
        builder._put_sample(env, b'first', b'a' * 1000)
        builder._put_sample(env, b'second', b'b' * 200000)
        assert env.info()['map_size'] > 32768
        with env.begin() as txn:
            assert txn.get(b'first') == b'a' * 1000
            assert txn.get(b'second') == b'b' * 200000
            assert txn.stat()['entries'] == 2
        with pytest.raises(ValueError, match='duplicate directed sample'):
            builder._put_sample(env, b'first', b'changed')
        with env.begin() as txn:
            assert txn.get(b'first') == b'a' * 1000


def test_batch_map_growth_and_duplicate_rollback(tmp_path):
    from scripts.data import prepare_joint_v2_boltz_chain_pairs as builder
    with builder.lmdb.open(str(tmp_path/'batch.lmdb'), map_size=32768) as env:
        builder._put_samples(env, [(b'a', b'x'*100000), (b'b', b'y'*100000)])
        with pytest.raises(ValueError, match='duplicate directed sample'):
            builder._put_samples(env, [(b'c', b'z'), (b'a', b'changed')])
        with env.begin() as txn:
            assert txn.stat()['entries'] == 2
            assert txn.get(b'c') is None
            assert txn.get(b'a') == b'x'*100000
