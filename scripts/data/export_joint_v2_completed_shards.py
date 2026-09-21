#!/usr/bin/env python
"""Publish a portable subset of closed shards while the full builder runs."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root), str(root / "src")]

import argparse
import csv
import json
from pathlib import Path
import shutil

import lmdb
import pyarrow as pa
import pyarrow.parquet as pq

from apexgen.joint_v2.data.boltz_npz import mapping_manifest
from apexgen.joint_v2.data.dataset import COMPLEX_DATASET_SCHEMA, COMPLEX_RECORD_SCHEMA, JointV2Dataset
from apexgen.joint_v2.data.batch import collate_joint_v2_records
from apexgen.joint_v2.data.sample_inventory import SAMPLE_FIELDS
from apexgen.joint_v2.runtime.lineage import canonical_sha256, joint_v2_dataset_identity, sha256_file
from apexgen.shared.storage.store import unpack_record
from scripts.data.prepare_joint_v2_boltz_chain_pairs import MANIFEST_SCHEMA
from scripts.data.prepare_joint_v2_boltz_npz import write_json


def export_completed_shards(source, output, count=3):
    source, output = Path(source).resolve(), Path(output).resolve()
    state = json.loads((source / 'build_status.json').read_text())
    if count < 1:
        raise ValueError('count must be positive')
    if state.get('status') != 'running':
        raise ValueError('this exporter requires a running build with declared shard size')
    size = state['shard_size']
    # A following shard must exist, proving the writer closed the selected one.
    if not (source / 'shards' / f'shard-{count:05d}.lmdb' / 'data.mdb').is_file():
        raise ValueError('selected shards are not all closed; select fewer shards')
    if state['counts']['included'] < count * size:
        raise ValueError('not enough committed samples')
    stage = output.with_name(output.name + '.inprogress')
    if output.exists() or stage.exists():
        raise FileExistsError(output)
    stage.mkdir(parents=True)
    shards = [f'shard-{i:05d}.lmdb' for i in range(count)]
    selected = set(shards)
    with (source / 'sample_inventory.csv').open() as stream:
        rows = []
        for row in csv.DictReader(stream):
            if row['shard_id'] in selected:
                rows.append(row)
                if len(rows) == count * size:
                    break
    if len(rows) != count * size or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('CSV snapshot has missing or duplicate samples')
    identities = []
    for name in shards:
        original = source / 'shards' / name / 'data.mdb'
        digest = sha256_file(original)
        target = stage / 'shards' / name / 'data.mdb'
        target.parent.mkdir(parents=True)
        shutil.copyfile(original, target)
        if sha256_file(target) != digest or sha256_file(original) != digest:
            raise ValueError('shard changed during snapshot')
        identities.append(dict(shard_id=name, data_mdb_sha256=digest))
    manifest, environments, sources = [], {}, set()
    try:
        for index, row in enumerate(rows):
            name = row['shard_id']
            if name not in environments:
                env = lmdb.open(str(stage / 'shards' / name), readonly=True, lock=False)
                environments[name] = env
                with env.begin() as txn:
                    if txn.stat()['entries'] != size:
                        raise ValueError('selected shard is not full')
            with environments[name].begin() as txn:
                payload = txn.get(row['tensor_key'].encode())
                if payload is None:
                    raise ValueError('CSV key is missing from shard')
                record = unpack_record(payload)
            if (record['sample_id'] != row['sample_id'] or record['split'] != row['split']
                    or record['raw_file_sha256'] != row['source_npz_sha256']
                    or record['peptide_length'] != int(row['target_length'])):
                raise ValueError('CSV/record identity mismatch')
            meta = record['interface_pair']
            digest = record['raw_file_sha256']
            local_source = f'sources/{digest}.npz'
            sources.add(digest)
            manifest.append(dict(schema_version='apexgen.manifest.v0', sample_id=record['sample_id'],
                source_pdb_id=record['source_pdb_id'], split=record['split'], status='included', reason=None,
                raw_path=local_source, raw_file_sha256=digest, shard_id=name, tensor_key=record['sample_id'],
                record_index=index, pair_id=meta['pair_id'], direction=meta['direction'],
                condition_chain_id=meta['condition_chain_id'], target_chain_id=meta['target_chain_id'],
                split_group=record['source_pdb_id'], peptide_length=record['peptide_length'],
                pocket_size=len(record['pocket_aatype']), target_source_length=meta['target_source_length'],
                target_start=meta['target_start'], target_stop=meta['target_stop'], fragment_index=meta['fragment_index'],
                target_interface_residue_count=len(meta['target_interface_residue_rows'])))
            row['record_index'] = index
            row['dataset_source_npz'] = local_source
    finally:
        for env in environments.values():
            env.close()
    (stage / 'sources').mkdir()
    for digest in sorted(sources):
        target = stage / 'sources' / f'{digest}.npz'
        shutil.copyfile(source / 'sources' / target.name, target)
        if sha256_file(target) != digest:
            raise ValueError('source NPZ hash mismatch')
    pq.write_table(pa.Table.from_pylist(manifest, schema=MANIFEST_SCHEMA), stage / 'manifest.parquet', compression='zstd')
    with (stage / 'sample_inventory.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    mapping = mapping_manifest()
    write_json(stage / 'mapping.json', mapping)
    policy = dict(formal=False, selection='first closed shards; not a representative random split',
        split_policy='preserve source smoke labels; no train/validation separation',
        minimum_target_length=4, mapping_sha256=mapping['mapping_sha256'])
    write_json(stage / 'metadata.json', dict(schema_version=COMPLEX_DATASET_SCHEMA,
        record_schema_version=COMPLEX_RECORD_SCHEMA, record_count=len(rows), formal=False,
        record_compression=state['compression'], shards=identities,
        manifest_sha256=sha256_file(stage / 'manifest.parquet'),
        sample_inventory_sha256=sha256_file(stage / 'sample_inventory.csv'),
        preprocessing=policy, preprocessing_config_sha256=canonical_sha256(policy)))
    identity = joint_v2_dataset_identity(stage)
    dataset = JointV2Dataset(stage, split='smoke')
    try:
        if len(dataset) != len(rows):
            raise ValueError('loader sample count mismatch')
        indices = sorted({0, len(rows) - 1, *(i * size for i in range(count)),
            *((i + 1) * size - 1 for i in range(count))})
        for i in indices:
            collate_joint_v2_records([dataset[i]]).condition.validate_model_input()
    finally:
        dataset.close()
    summary = dict(status='complete',export_kind='closed_shard_subset',samples=len(rows),sources=len(sources),
        selected_shards=shards,split='smoke',formal=False,source_build=str(source),
        source_progress_snapshot=state,dataset_identity=identity,loader_checked_indices=indices)
    write_json(stage / 'export.json', summary)
    write_json(stage / 'build_status.json', dict(status='complete',counts=dict(included=len(rows)),export_kind='subset'))
    stage.rename(output)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(export_completed_shards(args.source, args.output, args.count), indent=2))
