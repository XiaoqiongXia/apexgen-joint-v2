#!/usr/bin/env python
"""Interface Parquet -> bidirectional continuous-fragment Joint-v2 LMDB dataset.

Default split is smoke. Real split assignments must be supplied per source
structure, never independently for the two directions. No optimizer or GPU use.
"""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root), str(root / "src")]

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import shutil

import lmdb
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from apexgen.joint_v2.data.batch import collate_joint_v2_records
from apexgen.joint_v2.data.static_features import STATIC_FEATURES_SCHEMA, STATIC_FEATURES_KEY
from apexgen.joint_v2.data.boltz_chain_pairs import (
    DIRECTIONS, SCHEMA, adapt_interface_direction, fragment_selections,
)
from apexgen.joint_v2.data.boltz_npz import mapping_manifest
from apexgen.joint_v2.data.boltz_source import BoltzSource
from apexgen.joint_v2.data.dataset import COMPLEX_DATASET_SCHEMA, COMPLEX_RECORD_SCHEMA
from apexgen.joint_v2.data.sample_inventory import SAMPLE_FIELDS, _sample_inventory_row
from apexgen.joint_v2.runtime.lineage import canonical_sha256, joint_v2_dataset_identity, sha256_file
from apexgen.shared.storage.store import pack_record
from scripts.data.prepare_joint_v2_boltz_npz import smoke_model, write_json


MANIFEST_SCHEMA = pa.schema([
    (key, pa.string()) for key in (
        "schema_version", "sample_id", "source_pdb_id", "split", "status", "reason",
        "raw_path", "raw_file_sha256", "shard_id", "tensor_key", "pair_id", "direction",
        "condition_chain_id", "target_chain_id", "split_group")
] + [(key, pa.int64()) for key in (
    "record_index", "peptide_length", "pocket_size", "target_interface_residue_count",
    "target_source_length", "target_start", "target_stop", "fragment_index")])


def _write_build_status(stage, status, counts, **detail):
    temporary = stage / ".build_status.json.tmp"
    write_json(temporary, dict(status=status, counts=dict(counts), **detail))
    temporary.replace(stage / "build_status.json")


@contextmanager
def _build_stage(output, counts):
    """Retain incomplete products for inspection; never publish after an error."""
    stage = output.with_name(output.name + ".inprogress")
    stage.mkdir()  # Existing partial work must not be overwritten implicitly.
    _write_build_status(stage, "running", counts, final_output=str(output))
    try:
        yield stage
    except BaseException as error:
        if stage.exists():
            _write_build_status(stage, "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                counts, error_type=type(error).__name__, error=str(error), final_output=str(output),
                usable_for_training=False, automatic_resume_supported=False)
        raise


def _put_sample(environment, key, payload):
    """Grow the address-space limit if a large shard exceeds its initial map."""
    _put_samples(environment, [(key, payload)])


def _put_samples(environment, records):
    while True:
        try:
            with environment.begin(write=True) as transaction:
                for key, payload in records:
                    if not transaction.put(key, payload, overwrite=False):
                        raise ValueError("duplicate directed sample")
            return
        except lmdb.MapFullError:
            environment.set_mapsize(environment.info()['map_size'] * 2)


_WORKER_SOURCE = None


def _worker_init():
    import torch
    torch.set_num_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


def _pair_events(pair, source_path, split, minimum, parsed=None):
    for direction in DIRECTIONS:
        yield dict(directions_attempted=1), None, None, None
        if (not pair['contact_verified'] and all(
                not pair[f'chain_{side}_interface_residue_indices']
                and not pair[f'chain_{side}_interface_residue_rows']
                and pair[f'chain_{side}_interface_residue_count'] == 0 for side in ('a', 'b'))):
            yield dict(skipped_no_contact=1), dict(pair_id=pair['pair_id'],
                source_structure_id=pair['source_structure_id'], direction=direction, split=split,
                status='skipped_no_contact',
                reason='no observed heavy-atom contact within the inventory cutoff'), None, None
            continue
        selections = fragment_selections(pair, direction)
        yield dict(contact_runs=len(selections)), None, None, None
        for selection in selections:
            base = dict(pair_id=pair['pair_id'], source_structure_id=pair['source_structure_id'],
                direction=direction, split=split, selection=selection)
            if selection['fragment_length'] < minimum:
                yield dict(skipped_short=1), dict(**base, status='skipped_short',
                    reason=f'fragment length < {minimum}'), None, None
                continue
            yield dict(fragments_attempted=1), None, None, None
            try:
                kwargs = dict(fragment_index=selection['fragment_index'], split=split)
                if parsed is not None:
                    kwargs['source'] = parsed
                record, audit = adapt_interface_direction(source_path, pair, direction, **kwargs)
                if parsed is None:
                    collate_joint_v2_records([record]).condition.validate_model_input()
                elif record['peptide_length'] < 3:
                    # The adapter already collated/validated all tensors; the
                    # remaining model-input check is solely the minimum length.
                    raise ValueError('model input requires peptide length >= 3 (greater than 2)')
            except ValueError as error:
                yield dict(rejected=1), dict(**base, status='rejected', reason=str(error)), None, None
                continue
            yield {}, dict(**base, status='included'), record, audit


def _convert_task(task):
    global _WORKER_SOURCE
    result = []
    for pair, path, split, minimum in task:
        if _WORKER_SOURCE is None or _WORKER_SOURCE.path != Path(path):
            _WORKER_SOURCE = BoltzSource(path)
        _WORKER_SOURCE.check_path(path)
        result.append((pair, path, split, list(_pair_events(pair, path, split, minimum, _WORKER_SOURCE))))
    return result


def _prepared_tasks(scanner, stage, split_map, minimum, min_free_gib, seen_pairs, seen_sources):
    task, last_source = [], None
    for batch in scanner.to_batches():
        for pair in batch.to_pylist():
            if pair['pair_id'] in seen_pairs:
                raise ValueError(f"duplicate interface pair: {pair['pair_id']}")
            seen_pairs.add(pair['pair_id'])
            sid = pair['source_structure_id']
            if split_map is not None and sid not in split_map:
                raise ValueError(f'missing source-structure split assignment: {sid}')
            split = 'smoke' if split_map is None else split_map[sid]
            seen_sources.add(sid)
            digest = pair['source_npz_sha256']
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('invalid source content hash')
            source = stage/'sources'/f'{digest}.npz'
            if task and (last_source != source or len(task) >= 8):
                yield task
                task = []
            if shutil.disk_usage(stage).free < min_free_gib * 1024**3:
                raise OSError(f'free disk space below reserve of {min_free_gib} GiB')
            if not source.exists():
                offset, size = pair['source_offset'], pair['source_size']
                if offset < 0 or size <= 0:
                    raise ValueError('invalid source byte span')
                with Path(pair['source_archive']).open('rb') as archive:
                    archive.seek(offset)
                    payload = archive.read(size)
                if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
                    raise ValueError(f'archive/inventory identity mismatch: {sid}')
                source.write_bytes(payload)
            task.append((pair, str(source), split, minimum))
            last_source = source
    if task:
        yield task


@contextmanager
def _conversion_batches(tasks, workers):
    if workers == 1:
        def serial():
            for task in tasks:
                for pair, path, split, minimum in task:
                    yield [(pair, path, split, _pair_events(pair, path, split, minimum))]
        yield serial()
        return
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
        initializer=_worker_init)
    pending = deque()
    def parallel():
        iterator = iter(tasks)
        for _ in range(workers * 2):
            task = next(iterator, None)
            if task is None:
                break
            pending.append(pool.submit(_convert_task, task))
        while pending:
            # Consume in inventory order for deterministic records, CSV and shards.
            yield pending.popleft().result()
            task = next(iterator, None)
            if task is not None:
                pending.append(pool.submit(_convert_task, task))
    try:
        yield parallel()
    finally:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


def build_dataset(inventory, output, *, structure_ids=None, split_map=None, shard_size=8192, min_fragment_length=4,
                  min_free_gib=0, workers=1, commit_size=1, compression=None):
    inventory, output = Path(inventory).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if isinstance(commit_size, bool) or not isinstance(commit_size, int) or commit_size < 1:
        raise ValueError("commit_size must be a positive integer")
    if compression not in (None, 'zlib'):
        raise ValueError("compression must be None or zlib")
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    if split_map is not None and (not isinstance(split_map, dict) or not split_map
            or any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in split_map.items())):
        raise ValueError("split map must assign nonempty split names by source structure ID")
    if isinstance(min_fragment_length, bool) or not isinstance(min_fragment_length, int) or min_fragment_length < 3:
        raise ValueError("min_fragment_length must be an integer >= 3 (model minimum)")
    structure_ids = sorted(set(structure_ids)) if structure_ids else None
    selection = pads.field("source_structure_id").isin(structure_ids) if structure_ids else None
    scanner = pads.dataset(inventory, format="parquet").scanner(filter=selection, batch_size=64)
    started = time.monotonic()
    inventory_sha = sha256_file(inventory)
    counts, reasons = Counter(), Counter()
    seen_pairs, seen_sources = set(), set()
    shards, manifest_buffer = [], []
    output.parent.mkdir(parents=True, exist_ok=True)
    with _build_stage(output, counts) as stage:
        (stage / "sources").mkdir()
        environment = None
        shard_name = None
        try:
            with (pq.ParquetWriter(stage / "manifest.parquet", MANIFEST_SCHEMA, compression="zstd") as writer,
                  (stage / "directions.jsonl").open("w", buffering=1) as audits,
                  (stage / "sample_inventory.csv").open("w", newline="", encoding="utf-8") as sample_stream):
                sample_writer = csv.DictWriter(sample_stream, fieldnames=SAMPLE_FIELDS)
                sample_writer.writeheader()
                sample_stream.flush()
                pending, audit_buffer = [], []
                def flush_pending():
                    if pending:
                        if shutil.disk_usage(stage).free < min_free_gib * 1024**3:
                            raise OSError(f'free disk space below reserve of {min_free_gib} GiB')
                        _put_samples(environment, [(entry['tensor_key'].encode(), payload)
                            for entry, row, payload in pending])
                        for entry, row, payload in pending:
                            sample_writer.writerow(row)
                            manifest_buffer.append(entry)
                            counts['included'] += 1
                        sample_stream.flush()
                        pending.clear()
                    for event in audit_buffer:
                        audits.write(json.dumps(event) + '\n')
                    audit_buffer.clear()
                tasks = _prepared_tasks(scanner, stage, split_map, min_fragment_length,
                    min_free_gib, seen_pairs, seen_sources)
                with _conversion_batches(tasks, workers) as batches:
                    for converted in batches:
                        for pair, source, split, events in converted:
                            sid = pair['source_structure_id']
                            counts['pairs_attempted'] += 1
                            for delta, base, record, audit in events:
                                counts.update(delta)
                                if record is None:
                                    if base is not None:
                                        audit_buffer.append(base)
                                        if base['status'] == 'rejected':
                                            reasons[base['reason'].split(':', 1)[0]] += 1
                                    continue
                                if not pending and counts["included"] % shard_size == 0:
                                    if environment is not None:
                                        environment.close()
                                    shard_name = f"shard-{len(shards):05d}.lmdb"
                                    shard_path = stage / "shards" / shard_name
                                    shard_path.mkdir(parents=True)
                                    environment = lmdb.open(str(shard_path), map_size=8 * 1024**3, subdir=True)
                                    shards.append(shard_name)
                                record["raw_path"] = str(output / "sources" / Path(source).name)
                                meta = record["interface_pair"]
                                entry = dict(schema_version="apexgen.manifest.v0",
                                    sample_id=record["sample_id"], source_pdb_id=sid, split=split,
                                    status="included", reason=None, raw_path=record["raw_path"],
                                    raw_file_sha256=record["raw_file_sha256"], shard_id=shard_name,
                                    tensor_key=record["sample_id"], record_index=counts["included"] + len(pending),
                                    pair_id=pair["pair_id"], direction=base["direction"],
                                    condition_chain_id=meta["condition_chain_id"], target_chain_id=meta["target_chain_id"],
                                    split_group=sid, peptide_length=record["peptide_length"],
                                    pocket_size=len(record["pocket_aatype"]),
                                    target_source_length=meta["target_source_length"],
                                    target_start=meta["target_start"], target_stop=meta["target_stop"],
                                    fragment_index=meta["fragment_index"],
                                    target_interface_residue_count=len(meta["target_interface_residue_rows"]))
                                sample_row = _sample_inventory_row(record, pair, entry)
                                pending.append((entry, sample_row, pack_record(record, compression=compression)))
                                audit_buffer.append({**base, 'status': 'included', 'audit': audit})
                                if len(pending) >= commit_size or (counts['included'] + len(pending)) % shard_size == 0:
                                    flush_pending()
                        flush_pending()
                        if manifest_buffer:
                            writer.write_table(pa.Table.from_pylist(manifest_buffer, schema=MANIFEST_SCHEMA))
                            manifest_buffer.clear()
                        print(json.dumps(dict(progress=dict(counts))), flush=True)
                        _write_build_status(stage, "running", counts, final_output=str(output), workers=workers,
                            elapsed_seconds=round(time.monotonic() - started, 1), shard_size=shard_size,
                            commit_size=commit_size, compression=compression)
        finally:
            if environment is not None:
                environment.close()
        if not seen_pairs:
            raise ValueError("selection contains no interface pairs")
        if structure_ids and set(structure_ids) != seen_sources:
            raise ValueError(f"requested structures not found: {set(structure_ids) - seen_sources}")
        if sha256_file(inventory) != inventory_sha:
            raise ValueError("inventory changed during build")
        mapping = mapping_manifest()
        write_json(stage / "mapping.json", mapping)
        policy = dict(static_features_schema=STATIC_FEATURES_SCHEMA, schema=SCHEMA, mapping_sha256=mapping["mapping_sha256"],
            condition="single partner chain: native 5-A contact core plus 11-A CB context",
            target="each maximal sequence-consecutive interface run; observed N/CA/C throughout; no concatenation",
            target_length_limit=None, target_selection="all maximal contact runs as independent samples in source order",
            minimum_target_length=min_fragment_length,
            directions=list(DIRECTIONS), missing_sidechain="masked; never filled",
            split_policy="explicit source-structure map" if split_map is not None else "smoke only",
            split_map=split_map, msa_used=False, formal=False,
            limitations=["no sequence-cluster split or homology deduplication supplied by this builder",
                "other chains/ligands omitted from condition; nearby excluded atoms recorded",
                "linear canonical chemistry only; unsupported crosslinks rejected"])
        write_json(stage / "metadata.json", dict(schema_version=COMPLEX_DATASET_SCHEMA,
            record_schema_version=COMPLEX_RECORD_SCHEMA, record_count=counts["included"], formal=False,
            record_compression=compression,
            manifest_sha256=sha256_file(stage / "manifest.parquet"),
            sample_inventory_sha256=sha256_file(stage / "sample_inventory.csv"),
            shards=[dict(shard_id=s, data_mdb_sha256=sha256_file(stage / "shards" / s / "data.mdb")) for s in shards],
            preprocessing=policy, preprocessing_config_sha256=canonical_sha256(policy)))
        summary = dict(schema=SCHEMA, formal=False, inventory=str(inventory), inventory_sha256=inventory_sha,
            shard_size=shard_size, workers=workers, commit_size=commit_size, compression=compression,
            elapsed_seconds=round(time.monotonic() - started, 1),
            structure_selection=structure_ids, source_structures=len(seen_sources),
            counts={k: counts[k] for k in ("pairs_attempted", "directions_attempted", "contact_runs", "skipped_no_contact", "skipped_short",
                "fragments_attempted", "included", "rejected")},
            rejection_reasons=dict(reasons),
            code_sha256={str(path): sha256_file(path) for path in (
                Path(__file__).resolve(),
                Path(__file__).resolve().with_name("prepare_joint_v2_boltz_npz.py"),
                Path(__file__).resolve().parents[2] / "src/apexgen/joint_v2/data/sample_inventory.py",
                Path(__file__).resolve().parents[2] / "src/apexgen/joint_v2/data/boltz_chain_pairs.py",
                Path(__file__).resolve().parents[2] / "src/apexgen/joint_v2/data/boltz_npz.py",
                Path(__file__).resolve().parents[2] / "src/apexgen/joint_v2/data/boltz_source.py",
                Path(__file__).resolve().parents[2] / "src/apexgen/joint_v2/data/static_features.py",
                Path(__file__).resolve().parents[2] / "src/apexgen/joint_v2/data/batch.py",
                Path(__file__).resolve().parents[2] / "src/apexgen/shared/storage/store.py")},
            directions_sha256=sha256_file(stage / "directions.jsonl"),
            sample_inventory=dict(csv="sample_inventory.csv", samples=counts["included"],
                sha256=sha256_file(stage / "sample_inventory.csv")),
            dataset_identity=joint_v2_dataset_identity(stage) if shards else None)
        write_json(stage / "build.json", summary)
        if output.exists():
            raise FileExistsError(output)
        _write_build_status(stage, "complete", counts, final_output=str(output))
        stage.rename(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--structure-ids", nargs="+")
    parser.add_argument("--split-map", type=Path, help="JSON {source_structure_id: split}; otherwise smoke only")
    parser.add_argument("--commit-size", type=int, default=64)
    parser.add_argument("--compression", choices=["none", "zlib"], default="zlib")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard-size", type=int, default=8192)
    parser.add_argument("--min-free-gib", type=int, default=20)
    parser.add_argument("--min-fragment-length", type=int, default=4, help="inclusive minimum; default 4 (>3)")
    parser.add_argument("--model-smoke", action="store_true", help="CPU forward/backward in batches of 8")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2]
                        / "configs/joint_v2/experiments/sequence_structure_tiny_encoder_bottleneck_v1.yaml")
    args = parser.parse_args()
    summary = build_dataset(args.inventory, args.output, structure_ids=args.structure_ids,
        split_map=json.loads(args.split_map.read_text()) if args.split_map else None, shard_size=args.shard_size,
        min_fragment_length=args.min_fragment_length, min_free_gib=args.min_free_gib, workers=args.workers, commit_size=args.commit_size,
        compression=None if args.compression == "none" else args.compression)
    if args.model_smoke:
        if not summary["counts"]["included"]:
            raise ValueError("dataset published; no included records to smoke-test")
        batches = []
        for start in range(0, summary["counts"]["included"], 8):
            indices = list(range(start, min(start + 8, summary["counts"]["included"])))
            name = f"model_smoke_batch_{start // 8:04d}.json"
            result = smoke_model(args.output.resolve(), args.config, record_indices=indices, report_name=name)
            batches.append(dict(report=name, sample_ids=result["sample_ids"], status=result["status"]))
        summary["model_smoke"] = dict(status="passed", samples=summary["counts"]["included"], batches=batches)
        write_json(args.output / "model_smoke.json", summary["model_smoke"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
