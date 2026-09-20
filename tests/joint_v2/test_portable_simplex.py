from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from apexgen.joint_v2.data.boltz_npz import audit_boltz_record
from apexgen.joint_v2.data.dataset import JointV2Dataset
from apexgen.joint_v2.runtime.portable import (
    infer, load_config, read_checkpoint, train, training_batches, validate_config,
)
from apexgen.joint_v2.runtime.lineage import joint_v2_dataset_identity, sha256_file


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/joint_v2/portable/simplex_tiny.yaml"
EXAMPLE = ROOT / "examples/boltzgen19/dataset"


def test_batches_cover_epoch_and_resume_tail_without_global_rng():
    before = torch.get_rng_state()
    all_batches = list(training_batches(19, 4, 3, 0, 10))
    assert sorted(i for _, batch in all_batches[:5] for i in batch) == list(range(19))
    assert len(all_batches[4][1]) == 3
    assert list(training_batches(19, 4, 3, 3, 10)) == all_batches[3:]
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("section,key,value", [
    ("training", "batch_size", 0), ("training", "learning_rate", float("nan")),
    ("training", "precision", "float16"), ("sampling", "steps", 0),
])
def test_config_rejects_invalid_values(section, key, value):
    config = deepcopy(load_config(CONFIG))
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_example_relocates_and_preserves_source_audits(tmp_path):
    moved = tmp_path / "server_b/dataset"
    shutil.copytree(EXAMPLE, moved)
    assert joint_v2_dataset_identity(moved) == joint_v2_dataset_identity(EXAMPLE)
    dataset = JointV2Dataset(moved, split="smoke")
    try:
        assert len(dataset) == 19
        for i in range(len(dataset)):
            record = dataset[i]
            assert Path(record["raw_path"]).is_relative_to(moved)
            audit_boltz_record(record)
            target = record['joint_v2_target']
            context = record['pocket_atom_xyz'][record['pocket_atom_mask'].astype(bool)]
            for xyz, mask in zip(target['experimental_atom14'], target['experimental_atom14_mask']):
                distances = np.linalg.norm(xyz[mask.astype(bool)][:, None] - context[None], axis=-1)
                assert distances.min() <= 5.0 + 1e-5
    finally:
        dataset.close()


def test_identity_requires_every_consumed_shard_and_detects_its_changes(tmp_path):
    moved = tmp_path / 'dataset'
    shutil.copytree(EXAMPLE, moved)
    table = pq.read_table(moved / 'manifest.parquet')
    rows = table.to_pylist()
    extra = 'additional-shard'
    shutil.copytree(moved / 'shards' / rows[0]['shard_id'], moved / 'shards' / extra)
    rows[0]['shard_id'] = extra
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), moved / 'manifest.parquet')
    metadata_path = moved / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['manifest_sha256'] = sha256_file(moved / 'manifest.parquet')
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='manifest references undeclared tensor shards'):
        joint_v2_dataset_identity(moved)
    shard_path = moved / 'shards' / extra / 'data.mdb'
    metadata['shards'].append(dict(shard_id=extra, data_mdb_sha256=sha256_file(shard_path)))
    metadata_path.write_text(json.dumps(metadata))
    joint_v2_dataset_identity(moved)
    with shard_path.open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='tensor shard digest mismatch'):
        joint_v2_dataset_identity(moved)


def test_identity_rejects_duplicate_shard_declarations(tmp_path):
    moved = tmp_path / 'dataset'
    shutil.copytree(EXAMPLE, moved)
    metadata_path = moved / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['shards'].append(metadata['shards'][0])
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='duplicate shard identity'):
        joint_v2_dataset_identity(moved)


def test_training_resume_and_independent_inference_after_relocation(tmp_path):
    args = SimpleNamespace(config=CONFIG, dataset=EXAMPLE, split="smoke", device="cpu",
                           cpu_threads=2, resume=None, steps=2, output=tmp_path / "full")
    train(args)
    args.steps, args.output = 1, tmp_path / "first"
    train(args)
    checkpoint = args.output / "checkpoint_00000001.pt"
    moved = tmp_path / "another_server"
    moved.mkdir()
    shutil.copy2(checkpoint, moved / "checkpoint.pt")
    shutil.copytree(EXAMPLE, moved / "dataset")
    args.steps, args.resume = 2, moved / "checkpoint.pt"
    args.dataset, args.output = moved / "dataset", tmp_path / "resumed"
    train(args)
    full = read_checkpoint(tmp_path / "full/checkpoint_00000002.pt")
    resumed = read_checkpoint(args.output / "checkpoint_00000002.pt")
    for key, value in full["model_state"].items():
        assert torch.equal(value, resumed["model_state"][key]), key
    assert torch.equal(full["generator_state"], resumed["generator_state"])
    assert torch.equal(full["torch_rng_state"], resumed["torch_rng_state"])

    inference = SimpleNamespace(command="evaluate", checkpoint=args.output / "checkpoint_00000002.pt",
        dataset=moved / "dataset", split="smoke", output=tmp_path / "eval", device="cpu",
        cpu_threads=2, bases=1, limit=1, seed=42, sampling_steps=2)
    infer(inference)
    summary = json.loads((inference.output / "summary.json").read_text())
    assert summary["counts"]["rollout"] == 1
    assert summary["counts"]["denoise_t0"] == 1
    inference.command, inference.output = "sample", tmp_path / "samples"
    infer(inference)
    with np.load(inference.output / "sample_000000_base_000.npz", allow_pickle=False) as arrays:
        assert arrays["generated_backbone"].shape[-2:] == (3, 3)
        assert np.isfinite(arrays["generated_backbone"]).all()
        assert "native_aatype" not in arrays.files
    args.resume, args.split, args.output = moved / "checkpoint.pt", "train", tmp_path / "bad"
    with pytest.raises(ValueError, match="empty"):
        train(args)

    distributed = read_checkpoint(moved / "checkpoint.pt")
    distributed["distributed_training"] = dict(world_size=3, continuation_supported=False)
    torch.save(distributed, moved / "ddp.pt")
    args.resume, args.split = moved / "ddp.pt", "smoke"
    with pytest.raises(ValueError, match="not exact-state continuation"):
        train(args)
