"""Verified training skips static geometry audits without changing optimization."""
from copy import deepcopy
from dataclasses import dataclass
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import lmdb
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from apexgen.joint_v2.data.batch import collate_joint_v2_records
from apexgen.joint_v2.data.dataset import JointV2Dataset, JointV2DatasetView
from apexgen.joint_v2.data.static_features import STATIC_FEATURES_KEY, STATIC_FEATURES_SCHEMA, precompute_record
from apexgen.joint_v2.data.training_collate import prepare_training_collator
from apexgen.joint_v2.runtime.lineage import sha256_file, joint_v2_dataset_identity
from apexgen.shared.storage.store import pack_record, unpack_record
from test_static_features import make_dataset, assert_batches_equal


@pytest.fixture
def cached_dataset(tmp_path, record_factory):
    root = tmp_path / "dataset"
    make_dataset(root, record_factory, shards=2)
    metadata = json.loads((root / "metadata.json").read_text())
    for index, shard in enumerate(metadata["shards"]):
        path = root / "shards" / shard["shard_id"]
        env = lmdb.open(str(path), map_size=1 << 24)
        with env.begin(write=True) as transaction:
            key = f"sample{index}".encode()
            record = unpack_record(transaction.get(key))
            if index == 1:
                record["pocket_residue_keys"][1]["auth_chain_id"] = "B"
                record["joint_v2_target"]["experimental_atom14_mask"][2, 4:] = False
            precompute_record(record)
            transaction.put(key, pack_record(record, compression="zlib"))
        env.close()
        shard["data_mdb_sha256"] = sha256_file(path / "data.mdb")
    metadata["preprocessing"]["static_features_schema"] = STATIC_FEATURES_SCHEMA
    (root / "metadata.json").write_text(json.dumps(metadata))
    dataset = JointV2Dataset(root, split="validation")
    yield dataset
    dataset.close()


def forbid_audits(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("expensive geometry/hash audit ran in training loop")
    for path in (
        "apexgen.joint_v2.contracts.contract.observed_backbone_frames",
        "apexgen.joint_v2.contracts.contract.PeptideNativeTargets.validate_for",
        "apexgen.joint_v2.contracts.contract.UnifiedComplexCondition.validate_invariants",
        "apexgen.joint_v2.contracts.state.JointFlowState.validate",
        "apexgen.joint_v2.data.static_features.input_fingerprint",
        "apexgen.joint_v2.data.static_features._digest",
        "apexgen.joint_v2.data.batch.extract_backbone_torsions",
        "apexgen.joint_v2.data.batch.extract_chi",
        "apexgen.joint_v2.data.batch._residue_topology",
    ):
        monkeypatch.setattr(path, forbidden)


def test_light_collation_exact_and_no_repeated_audits(cached_dataset, monkeypatch):
    before = torch.get_rng_state()
    collator = prepare_training_collator(cached_dataset)
    assert torch.equal(before, torch.get_rng_state())
    assert collator.report["mode"] == "light"
    assert collator.report["startup_checked_samples"] == 2
    records = [cached_dataset[i] for i in range(2)]
    # Different pocket lengths exercise padding, not just fixed-size toy layouts.
    short = deepcopy(records[1])
    for key in ("pocket_aatype", "pocket_atom_xyz", "pocket_atom_mask", "pocket_residue_translation",
                "pocket_residue_rotation", "pocket_core_mask", "pocket_residue_keys"):
        short[key] = short[key][:1]
    precompute_record(short)
    records[1] = short
    expected = collate_joint_v2_records(records)
    collator = pickle.loads(pickle.dumps(collator))
    forbid_audits(monkeypatch)
    actual = collator(records)
    assert_batches_equal(actual, expected)
    assert not actual.condition.pocket_atom_mask[actual.condition.peptide_mask].any()
    assert (actual.condition.aatype[actual.condition.peptide_mask] == 20).all()


def test_light_forward_backward_exact(cached_dataset, small_model, monkeypatch):
    from apexgen.joint_v2.sampling.base import sample_base_state
    from apexgen.joint_v2.training.step import joint_endpoint_training_step
    collator = prepare_training_collator(cached_dataset)
    records = [cached_dataset[i] for i in range(2)]
    strict = collate_joint_v2_records(records)
    def run(batch):
        small_model.zero_grad(set_to_none=True)
        base = sample_base_state(batch.condition, generator=torch.Generator().manual_seed(91))
        output = joint_endpoint_training_step(small_model, base, batch.condition, batch.targets,
            generator=torch.Generator().manual_seed(92))
        output.losses["total"].backward()
        return output.losses, {n:p.grad.clone() for n,p in small_model.named_parameters() if p.grad is not None}
    expected, expected_gradients = run(strict)
    forbid_audits(monkeypatch)
    actual, gradients = run(collator(records))
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    for key in expected_gradients:
        assert torch.isfinite(gradients[key]).all()
        torch.testing.assert_close(gradients[key], expected_gradients[key], atol=0, rtol=0)


def test_debug_override_and_explicit_full(cached_dataset, monkeypatch):
    collator = prepare_training_collator(cached_dataset)
    record = cached_dataset[0]
    record["joint_v2_target"]["experimental_atom14"][0, 0, 0] += 1
    monkeypatch.setenv("APEXGEN_JOINT_V2_DEBUG_INVARIANTS", "1")
    with pytest.raises(ValueError, match="stale static features"):
        collator([record])
    assert prepare_training_collator(cached_dataset).report["mode"] == "full"
    monkeypatch.setenv("APEXGEN_JOINT_V2_DEBUG_INVARIANTS", "0")
    assert prepare_training_collator(cached_dataset, geometry_checks="full").report["mode"] == "full"


@pytest.mark.parametrize("corrupt", ["cache_missing", "schema", "contract", "shape", "dtype", "index", "too_short"])
def test_light_keeps_structural_checks(cached_dataset, corrupt):
    collator = prepare_training_collator(cached_dataset)
    record = cached_dataset[0]
    if corrupt == "cache_missing":
        del record[STATIC_FEATURES_KEY]
    elif corrupt == "schema":
        record[STATIC_FEATURES_KEY]["schema_version"] = "wrong"
    elif corrupt == "contract":
        record[STATIC_FEATURES_KEY]["joint_v2_contract_sha256"] = "wrong"
    elif corrupt == "shape":
        record[STATIC_FEATURES_KEY]["peptide"]["rotation"] = np.eye(3,dtype=np.float32)
    elif corrupt == "dtype":
        record["pocket_atom_xyz"] = record["pocket_atom_xyz"].astype(np.float64)
    elif corrupt == "index":
        record["joint_v2_target"]["aatype"][0] = 20
    else:
        record["peptide_length"] = 2
    with pytest.raises(ValueError):
        collator([record])


def test_legacy_uses_full_checks(tmp_path, record_factory):
    root = tmp_path / "legacy"
    make_dataset(root, record_factory, shards=1)
    dataset = JointV2Dataset(root, split="validation")
    try:
        assert prepare_training_collator(dataset).report["mode"] == "full"
    finally:
        dataset.close()


def test_identity_tampering_fails_before_light_mode(cached_dataset):
    identity = joint_v2_dataset_identity(cached_dataset.pockets.root)
    path = cached_dataset.pockets.root / "metadata.json"
    path.write_text(path.read_text()+"\n")
    with pytest.raises(ValueError, match="identity changed"):
        prepare_training_collator(cached_dataset, verified_identity=identity)


@dataclass
class _NumpyBatchCollator:
    collator: object

    def __call__(self, records):
        batch = self.collator(records)
        # Exercise real collation inside a spawned worker. Return NumPy copies
        # because sandboxed tensor IPC cannot create its shared-memory sockets,
        # and this project's absolute TMPDIR also exceeds AF_UNIX path limits.
        return {role: {name: value.numpy().copy() for name, value in vars(getattr(batch, role)).items()}
                for role in ("condition", "targets")}


def test_views_and_spawn_workers(cached_dataset):
    view = JointV2DatasetView(cached_dataset, (1, 0))
    collator = prepare_training_collator(view)
    records = [view[i] for i in range(2)]
    expected = collate_joint_v2_records(records)
    view.close()
    loader = DataLoader(view, batch_size=2, num_workers=1, collate_fn=_NumpyBatchCollator(collator),
                        multiprocessing_context="spawn", timeout=30)
    batch = next(iter(loader))
    for role in ("condition", "targets"):
        for name, value in vars(getattr(expected, role)).items():
            torch.testing.assert_close(torch.from_numpy(batch[role][name]), value, atol=0, rtol=0)


def test_portable_optimization_identical_full_and_light(cached_dataset, tmp_path):
    from apexgen.joint_v2.runtime.portable import train, read_checkpoint
    config = Path(__file__).resolve().parents[2] / "configs/joint_v2/portable/simplex_tiny.yaml"
    args = SimpleNamespace(config=config, dataset=cached_dataset.pockets.root, split="validation",
        device="cpu", cpu_threads=1, resume=None, steps=2, output=tmp_path/"full", geometry_checks="full")
    cached_dataset.close()
    train(args)
    args.output, args.geometry_checks = tmp_path/"light", "auto"
    train(args)
    full = read_checkpoint(tmp_path/"full/checkpoint_00000002.pt")
    light = read_checkpoint(tmp_path/"light/checkpoint_00000002.pt")
    for key in full["model_state"]:
        torch.testing.assert_close(full["model_state"][key],light["model_state"][key],atol=0,rtol=0)
    assert torch.equal(full["generator_state"], light["generator_state"])
    assert torch.equal(full["torch_rng_state"], light["torch_rng_state"])
    report = json.loads((tmp_path/"light/run.json").read_text())["input_validation"]
    assert report["mode"] == "light" and not report["per_batch_geometry_checks"]
