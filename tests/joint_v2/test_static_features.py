"""Static caching must preserve observed labels and reject stale geometry."""
from copy import deepcopy
from dataclasses import fields
import json

import lmdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from apexgen.joint_v2.data.batch import collate_joint_v2_records
from apexgen.joint_v2.data.dataset import JointV2Dataset, COMPLEX_RECORD_SCHEMA, COMPLEX_DATASET_SCHEMA, NATIVE_TARGET_SCHEMA
from apexgen.joint_v2.data.static_features import (
    STATIC_FEATURES_KEY, STATIC_FEATURES_SCHEMA, UPGRADE_MARKER, precompute_record,
)
from apexgen.joint_v2.runtime.lineage import sha256_file
from apexgen.shared.storage.store import pack_record, unpack_record, RECORD_SCHEMA_VERSION
from scripts.data.upgrade_joint_v2_static_features import run_upgrade


def assert_batches_equal(actual, expected):
    for role in ("condition", "targets"):
        for field in fields(getattr(expected, role)):
            torch.testing.assert_close(getattr(getattr(actual, role), field.name),
                getattr(getattr(expected, role), field.name), rtol=0, atol=0)


def test_cached_mixed_padding_roundtrip_and_no_geometry_recomputation(record_factory, monkeypatch):
    records = [record_factory("first", peptide_length=4), record_factory("second", peptide_length=6)]
    # Chain breaks and missing sidechains must keep exactly the same masks.
    records[0]["pocket_residue_keys"][1]["auth_chain_id"] = "B"
    records[1]["joint_v2_target"]["experimental_atom14_mask"][2, 4:] = False
    expected = collate_joint_v2_records(records)
    cached = []
    for record in deepcopy(records):
        record["schema_version"] = RECORD_SCHEMA_VERSION
        cached.append(unpack_record(pack_record(precompute_record(record, verify=True), compression="zlib")))
    assert_batches_equal(collate_joint_v2_records([cached[0], records[1]]), expected)
    def forbidden(*args, **kwargs):
        raise AssertionError("cached collation must not re-extract static features")
    for name in ("_pocket_angles", "_residue_topology", "extract_backbone_torsions", "extract_chi",
                 "observed_backbone_frames", "_angle_sin_cos", "_peptide_angle_mask"):
        monkeypatch.setattr("apexgen.joint_v2.data.batch." + name, forbidden)
    assert_batches_equal(collate_joint_v2_records(cached), expected)


@pytest.mark.parametrize("change", ["xyz", "sequence", "mask", "topology", "link"])
def test_stale_cache_rejected(record_factory, change):
    record = precompute_record(record_factory())
    if change == "xyz":
        record["joint_v2_target"]["experimental_atom14"][0, 0, 0] += 1
    elif change == "sequence":
        record["joint_v2_target"]["aatype"][0] = 3
    elif change == "mask":
        record["pocket_atom_mask"][0, 4] = True
    elif change == "topology":
        record["pocket_residue_keys"][1]["auth_chain_id"] = "B"
    else:
        record["pocket_backbone_link_mask"] = np.array([False])
    with pytest.raises(ValueError, match="stale static features"):
        collate_joint_v2_records([record])
    precompute_record(record)
    collate_joint_v2_records([record])


@pytest.mark.parametrize("change", ["schema", "contract", "field", "shape", "dtype", "value"])
def test_cache_corruption_fails_closed(record_factory, change):
    record = precompute_record(record_factory())
    cache = record[STATIC_FEATURES_KEY]
    if change == "schema":
        cache["schema_version"] = "future"
    elif change == "contract":
        cache["joint_v2_contract_sha256"] = "bad"
    elif change == "field":
        del cache["pocket"]["sequence_index"]
    elif change == "shape":
        cache["peptide"]["rotation"] = cache["peptide"]["rotation"][:1]
    elif change == "dtype":
        cache["peptide"]["rotation"] = cache["peptide"]["rotation"].astype(np.float64)
    else:
        cache["pocket"]["backbone_angles_sin_cos"][0, 0, 0] += .1
    with pytest.raises(ValueError):
        collate_joint_v2_records([record])


def test_cached_forward_backward_matches_legacy(record_factory, small_model):
    from apexgen.joint_v2.sampling.base import sample_base_state
    from apexgen.joint_v2.training.step import joint_endpoint_training_step
    records = [record_factory()]
    old = collate_joint_v2_records(records)
    new = collate_joint_v2_records([precompute_record(deepcopy(records[0]))])
    outputs, gradients = [], []
    for batch in (old, new):
        small_model.zero_grad(set_to_none=True)
        base = sample_base_state(batch.condition, generator=torch.Generator().manual_seed(11))
        output = joint_endpoint_training_step(small_model, base, batch.condition, batch.targets,
            generator=torch.Generator().manual_seed(12))
        output.losses["total"].backward()
        outputs.append(output.losses["total"].detach())
        gradients.append({n: p.grad.clone() for n, p in small_model.named_parameters() if p.grad is not None})
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
    assert gradients[0] and any(g.abs().sum() > 0 for g in gradients[0].values())
    for key in gradients[0]:
        assert torch.isfinite(gradients[1][key]).all()
        torch.testing.assert_close(gradients[0][key], gradients[1][key], atol=0, rtol=0)


def make_dataset(root, record_factory, shards=2):
    root.mkdir()
    entries, metadata_shards, records = [], [], []
    for index in range(shards):
        record = record_factory(f"sample{index}")
        record.update(schema_version=RECORD_SCHEMA_VERSION, complex_schema_version=COMPLEX_RECORD_SCHEMA)
        record["joint_v2_target"].update(schema_version=NATIVE_TARGET_SCHEMA,
            sample_id=record["sample_id"], peptide_length=record["peptide_length"])
        name = f"shard-{index:05d}.lmdb"
        path = root / "shards" / name
        path.mkdir(parents=True)
        env = lmdb.open(str(path), map_size=1 << 24)
        with env.begin(write=True) as transaction:
            transaction.put(record["sample_id"].encode(), pack_record(record))
        env.close()
        entries.append(dict(sample_id=record["sample_id"], status="included", split="validation",
            shard_id=name, tensor_key=record["sample_id"]))
        metadata_shards.append(dict(shard_id=name, data_mdb_sha256=sha256_file(path / "data.mdb")))
        records.append(record)
    pq.write_table(pa.Table.from_pylist(entries), root / "manifest.parquet")
    (root / "sample_inventory.csv").write_text("sample_id\n" + "".join(r["sample_id"]+"\n" for r in records))
    metadata = dict(schema_version=COMPLEX_DATASET_SCHEMA, record_schema_version=COMPLEX_RECORD_SCHEMA,
        record_count=shards, shards=metadata_shards, preprocessing={},
        manifest_sha256=sha256_file(root / "manifest.parquet"),
        sample_inventory_sha256=sha256_file(root / "sample_inventory.csv"))
    (root / "metadata.json").write_text(json.dumps(metadata))
    (root / "build.json").write_text("{}")
    return records


def test_migration_preserves_csv_manifest_and_tensors(tmp_path, record_factory):
    output = tmp_path / "dataset"
    records = make_dataset(output, record_factory)
    csv_sha = sha256_file(output / "sample_inventory.csv")
    manifest_sha = sha256_file(output / "manifest.parquet")
    result = run_upgrade(output, workers=2, min_free_gib=0)
    assert result["records"] == 2 and result["status"] == "complete"
    assert sha256_file(output / "sample_inventory.csv") == csv_sha
    assert sha256_file(output / "manifest.parquet") == manifest_sha
    assert not (output / UPGRADE_MARKER).exists()
    dataset = JointV2Dataset(output, split="validation")
    try:
        actual = [dataset[index] for index in range(2)]
        assert all(record[STATIC_FEATURES_KEY]["schema_version"] == STATIC_FEATURES_SCHEMA for record in actual)
        assert_batches_equal(collate_joint_v2_records(actual), collate_joint_v2_records(records))
    finally:
        dataset.close()


def test_live_migration_leaves_producer_files_unchanged_and_waits(tmp_path, record_factory):
    output = tmp_path / "dataset"
    stage = tmp_path / "dataset.inprogress"
    make_dataset(stage, record_factory)
    original = [sha256_file(path) for path in sorted(stage.glob("shards/*/data.mdb"))]
    with pytest.raises(RuntimeError, match="unfinished"):
        run_upgrade(output, workers=1, min_free_gib=0)
    assert [sha256_file(path) for path in sorted(stage.glob("shards/*/data.mdb"))] == original
    status = json.loads((tmp_path / ".dataset.static-upgrade/status.json").read_text())
    assert set(status["prepared"]) == {"shard-00000.lmdb"}
    with pytest.raises(RuntimeError, match="incomplete"):
        JointV2Dataset(stage, split="validation")
    stage.rename(output)
    result = run_upgrade(output, workers=1, min_free_gib=0)
    assert result["status"] == "complete"


def test_recover_after_atomic_replacement_before_checkpoint(tmp_path, record_factory, monkeypatch):
    from scripts.data import upgrade_joint_v2_static_features as module
    output = tmp_path / "dataset"
    make_dataset(output, record_factory, shards=1)
    original_replace = module.os.replace
    def interrupted(source, destination):
        original_replace(source, destination)
        if str(destination).endswith("data.mdb"):
            raise RuntimeError("simulated crash after replacement")
    with monkeypatch.context() as patch:
        patch.setattr(module.os, "replace", interrupted)
        with pytest.raises(RuntimeError, match="simulated crash"):
            run_upgrade(output, workers=1, min_free_gib=0)
    assert (output / UPGRADE_MARKER).exists()
    assert run_upgrade(output, workers=1, min_free_gib=0)["status"] == "complete"


def test_declared_cached_dataset_cannot_silently_fall_back(tmp_path, record_factory):
    output = tmp_path / "dataset"
    make_dataset(output, record_factory, shards=1)
    metadata = json.loads((output / "metadata.json").read_text())
    metadata["preprocessing"]["static_features_schema"] = STATIC_FEATURES_SCHEMA
    (output / "metadata.json").write_text(json.dumps(metadata))
    dataset = JointV2Dataset(output, split="validation")
    try:
        with pytest.raises(ValueError, match="record cache is missing"):
            dataset[0]
    finally:
        dataset.close()
