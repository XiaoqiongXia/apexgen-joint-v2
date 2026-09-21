#!/usr/bin/env python
"""Build a small identity-audited NPZ panel and optionally smoke-test simplex FM.

Input: JSON list of {path, sample_id, source_pdb_id, receptor_chain_ids,
peptide_chain_id, split?}. Chain names are exact Boltz assembly-copy names.
Source files are copied into the dataset; no downloads or GPU use.
"""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root), str(root / "src")]

import argparse
from copy import deepcopy
from dataclasses import fields
import json
from pathlib import Path
import shutil
import tempfile

import lmdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from apexgen.joint_v2.data.boltz_npz import adapt_boltz_npz, audit_boltz_record, mapping_manifest
from apexgen.joint_v2.data.batch import collate_joint_v2_records
from apexgen.joint_v2.data.static_features import STATIC_FEATURES_SCHEMA, STATIC_FEATURES_KEY
from apexgen.joint_v2.data.dataset import COMPLEX_DATASET_SCHEMA, COMPLEX_RECORD_SCHEMA, JointV2Dataset
from apexgen.joint_v2.runtime.lineage import canonical_sha256, joint_v2_dataset_identity, sha256_file
from apexgen.shared.storage.store import pack_record


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def build_panel(rows, output):
    """Publish a fully audited, immutable small panel through a local staging dir."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("panel must have nonempty, unique sample identities")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.build-", dir=output.parent) as temp:
        stage = Path(temp)
        (stage / "sources").mkdir()
        shard_name = "shard-00000.lmdb"
        shard = stage / "shards" / shard_name
        shard.mkdir(parents=True)
        environment = lmdb.open(str(shard), map_size=1024**3, subdir=True)
        manifest, audits = [], []
        try:
            with environment.begin(write=True) as transaction:
                for index, row in enumerate(rows):
                    source_name = f"{index:04d}.npz"
                    shutil.copyfile(Path(row["path"]).resolve(), stage / "sources" / source_name)
                    record = adapt_boltz_npz(stage / "sources" / source_name,
                        sample_id=row["sample_id"], source_pdb_id=row["source_pdb_id"],
                        receptor_chain_ids=row["receptor_chain_ids"], peptide_chain_id=row["peptide_chain_id"],
                        split=row.get("split", "smoke"))
                    audit = audit_boltz_record(record)
                    audit.update(receptor_chain_ids=list(row["receptor_chain_ids"]), peptide_chain_id=row["peptide_chain_id"],
                        omitted_receptor_residues=len(record["boltz_adapter"]["omitted_receptor_residues"]),
                        excluded_peptide_atoms=record["boltz_adapter"]["excluded_peptide_atoms"],
                        nearby_excluded_atom_count=len(record["boltz_adapter"]["nearby_excluded_atom_rows"]))
                    audits.append(audit)
                    record["raw_path"] = str(output / "sources" / source_name)
                    transaction.put(row["sample_id"].encode(), pack_record(record), overwrite=False)
                    manifest.append(dict(schema_version="apexgen.manifest.v0", sample_id=row["sample_id"],
                        source_pdb_id=row["source_pdb_id"], split=record["split"], status="included", reason=None,
                        raw_path=record["raw_path"], raw_file_sha256=record["raw_file_sha256"],
                        shard_id=shard_name, tensor_key=row["sample_id"], record_index=index,
                        peptide_length=record["peptide_length"], pocket_size=len(record["pocket_aatype"])))
        finally:
            environment.close()
        pq.write_table(pa.Table.from_pylist(manifest), stage / "manifest.parquet")
        mappings = mapping_manifest()
        write_json(stage / "mapping.json", mappings)
        policy = dict(static_features_schema=STATIC_FEATURES_SCHEMA, adapter=mappings["schema"], mapping_sha256=mappings["mapping_sha256"],
            core_cutoff_angstrom=5.0, context_radius_angstrom=11.0,
            site_selection="native interface crop; inference requires a supplied target site",
            coordinate_system="subtract core CA centroid; observed atoms only",
            roles="explicit Boltz assembly-chain names", msa_used=False,
            supported="canonical linear protein peptide; one coordinate model; explicit receptor chains")
        write_json(stage / "metadata.json", dict(schema_version=COMPLEX_DATASET_SCHEMA,
            record_schema_version=COMPLEX_RECORD_SCHEMA, record_count=len(rows), formal=False,
            manifest_sha256=sha256_file(stage / "manifest.parquet"),
            shards=[dict(shard_id=shard_name, data_mdb_sha256=sha256_file(shard / "data.mdb"))],
            preprocessing=policy, preprocessing_config_sha256=canonical_sha256(policy)))
        identity = joint_v2_dataset_identity(stage)
        write_json(stage / "build.json", dict(formal=False, purpose="adapter smoke panel, not a train/validation split",
            inputs=rows, audits=audits, dataset_identity=identity))
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    return audits


def smoke_model(output, config_path, *, record_indices=None, report_name="model_smoke.json"):
    """CPU forward/backward; no parameter updates and no training checkpoint."""
    from apexgen.joint_v2.contracts.task_contract import TaskObservation
    from apexgen.joint_v2.model.simplex_codesign import SimplexCodesignModel
    from apexgen.joint_v2.runtime.config import load_joint_v2_config
    from apexgen.joint_v2.sampling.simplex_runtime import (
        sample_simplex_base, simplex_training_path, simplex_losses,
    )

    torch.set_num_threads(2)
    torch.manual_seed(20260920)
    rows = pq.read_table(output / "manifest.parquet").to_pylist()
    if len({r["split"] for r in rows}) != 1:
        raise ValueError("smoke panel must use one split")
    dataset = JointV2Dataset(output, split=rows[0]["split"])
    try:
        indices = range(len(dataset)) if record_indices is None else record_indices
        records = [dataset[i] for i in indices]
    finally:
        dataset.close()
    audits = [audit_boltz_record(record) for record in records]
    batch = collate_joint_v2_records(records)
    condition = batch.condition
    # Crop is already fixed. Native peptide labels must not enter static model
    # conditions; the supervised noisy training state intentionally uses labels.
    changed = deepcopy(records)
    for record in changed:
        record.pop(STATIC_FEATURES_KEY, None)  # Deliberately edited labels require rebuilding.
        target = record["joint_v2_target"]
        target["experimental_atom14"][target["experimental_atom14_mask"]] += 19.0
        target["aatype"] = (target["aatype"] + 7) % 20
    changed_condition = collate_joint_v2_records(changed).condition
    for field in fields(condition):
        if not torch.equal(getattr(condition, field.name), getattr(changed_condition, field.name)):
            raise AssertionError(f"native labels leaked into static condition: {field.name}")
    assert (condition.aatype[condition.peptide_mask] == 20).all()
    assert not condition.pocket_atom_mask[condition.peptide_mask].any()
    config = load_joint_v2_config(config_path)
    model = SimplexCodesignModel(config)
    generator = torch.Generator().manual_seed(20260920)
    time = torch.linspace(0.15, 0.85, len(records))
    base = sample_simplex_base(condition, generator=generator)
    state = simplex_training_path(base, batch, time, generator=generator)
    state.validate(condition)
    prediction = model(state, time, TaskObservation("J", condition))
    losses = simplex_losses(prediction, batch, state)
    if not all(torch.isfinite(value).all() for value in losses.values()):
        raise AssertionError("nonfinite model loss")
    losses["total"].mean().backward()
    gradients = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
    if not gradients or not all(torch.isfinite(value).all() for value in gradients.values()):
        raise AssertionError("missing or nonfinite model gradients")
    active = [name for name, gradient in gradients.items() if bool(torch.count_nonzero(gradient))]
    if not active:
        raise AssertionError("all model gradients are zero")
    assert torch.equal(prediction.translation[condition.pocket_mask], condition.pocket_translation[condition.pocket_mask])
    assert torch.equal(prediction.rotation[condition.pocket_mask], condition.pocket_rotation[condition.pocket_mask])
    # At default zero-head initialization, upstream gradients are exactly zero.
    # A separate diagnostic perturbation tests the entire feature-to-loss path.
    # This model is discarded; no optimizer, checkpoint or training run is used.
    model.zero_grad(set_to_none=True)
    structure = model.network.decoder.structure_module
    with torch.no_grad():
        structure.backbone_update.linear.weight.normal_(std=0.01)
        structure.sequence_update.weight.normal_(std=0.01)
    probe_prediction = model(state, time, TaskObservation("J", condition))
    assert torch.equal(probe_prediction.translation[condition.pocket_mask], condition.pocket_translation[condition.pocket_mask])
    assert torch.equal(probe_prediction.rotation[condition.pocket_mask], condition.pocket_rotation[condition.pocket_mask])
    probe_losses = simplex_losses(probe_prediction, batch, state)
    if not all(torch.isfinite(value).all() for value in probe_losses.values()):
        raise AssertionError("nonfinite gradient-probe loss")
    probe_losses["total"].mean().backward()
    probe_active = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise AssertionError(f"nonfinite gradient in {name}")
            if torch.count_nonzero(parameter.grad):
                probe_active.append(name)
    if not any("encoder" in name for name in probe_active):
        raise AssertionError("probe did not backpropagate into the encoder")
    summary = dict(status="passed", formal=False, device="cpu", optimizer_steps=0,
        model="SimplexCodesignModel", config=str(Path(config_path).resolve()), config_sha256=sha256_file(config_path),
        sample_ids=list(batch.sample_ids), layout=list(condition.layout),
        mapping_audits=audits, static_condition_independent_of_native_labels=True,
        fixed_pocket_unchanged=True, finite_backward=True, nonzero_gradient_parameters=len(active),
        active_encoder_gradient_parameters=sum("encoder" in name for name in active),
        perturbed_head_gradient_probe=dict(nonzero_gradient_parameters=len(probe_active),
            active_encoder_gradient_parameters=sum("encoder" in name for name in probe_active),
            losses_per_sample={k: v.detach().tolist() for k, v in probe_losses.items()}),
        losses_per_sample={k: v.detach().tolist() for k, v in losses.items()},
        torch_version=torch.__version__, numpy_version=np.__version__,
        dataset_identity=joint_v2_dataset_identity(output))
    write_json(output / report_name, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-smoke", action="store_true")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2]
                        / "configs/joint_v2/experiments/sequence_structure_tiny_encoder_bottleneck_v1.yaml")
    args = parser.parse_args()
    rows = json.loads(args.index.read_text())
    audits = build_panel(rows, args.output)
    result = dict(output=str(args.output.resolve()), samples=audits)
    if args.model_smoke:
        smoke = smoke_model(args.output.resolve(), args.config)
        result["model_smoke"] = {k: smoke[k] for k in ("status", "layout", "finite_backward", "nonzero_gradient_parameters")}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
