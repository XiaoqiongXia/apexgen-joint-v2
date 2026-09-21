#!/usr/bin/env python
"""Add observed static features to closed LMDB shards, including a live build.

A next shard proves the producer has closed the previous one. During conversion,
only compact feature sidecars are prepared. After the producer finishes, each
replacement is verified in a separate LMDB before atomic file replacement. A persistent
marker prevents JointV2Dataset reads until every shard and metadata hash agree.
No NPZ parsing, sample selection, CSV editing, or model parameter updates occur.
"""
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root), str(root / "src")]

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import lmdb
import pyarrow.parquet as pq
import torch

from apexgen.joint_v2.data.static_features import (
    STATIC_FEATURES_KEY, STATIC_FEATURES_SCHEMA, UPGRADE_MARKER,
    precompute_record, read_static_features,
)
from apexgen.joint_v2.runtime.lineage import canonical_sha256, joint_v2_dataset_identity, sha256_file
from apexgen.shared.storage.store import pack_record, unpack_record, RECORD_SCHEMA_VERSION


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _root(output):
    stage = output.with_name(output.name + ".inprogress")
    return output if output.exists() else stage


def _hash_record(digest, key, record):
    legacy = {name: value for name, value in record.items() if name != STATIC_FEATURES_KEY}
    data = pack_record(legacy)
    digest.update(len(key).to_bytes(8, "little"))
    digest.update(key)
    digest.update(len(data).to_bytes(8, "little"))
    digest.update(data)


def _init_worker():
    torch.set_num_threads(1)


def prepare_shard_cache(output, shard_name, work, min_free_gib):
    """Compute compact sidecars while leaving the producer's immutable shards alone."""
    output, work = Path(output), Path(work)
    source_path = _root(output) / "shards" / shard_name
    original_sha = sha256_file(source_path / "data.mdb")
    target = work / "prepared" / shard_name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    source = lmdb.open(str(source_path), readonly=True, lock=False, readahead=False)
    cache = lmdb.open(str(target), map_size=8 * 1024**3)
    count = 0
    try:
        with source.begin() as transaction:
            pending = []
            for key, payload in transaction.cursor():
                record = unpack_record(payload)
                if key.decode() != record["sample_id"]:
                    raise ValueError("LMDB key/sample identity mismatch")
                precompute_record(record, verify=True)
                envelope = dict(schema_version=RECORD_SCHEMA_VERSION,
                    sample_id=record["sample_id"], features=record[STATIC_FEATURES_KEY])
                pending.append((key, pack_record(envelope, compression="zlib")))
                count += 1
                if len(pending) == 64:
                    _commit(cache, pending, work, min_free_gib)
                    pending.clear()
            if pending:
                _commit(cache, pending, work, min_free_gib)
        cache.sync()
    finally:
        source.close()
        cache.close()
    if sha256_file(_root(output) / "shards" / shard_name / "data.mdb") != original_sha:
        raise ValueError("source shard changed while computing features")
    return dict(shard_id=shard_name, records=count, original_sha256=original_sha,
        cache_sha256=sha256_file(target / "data.mdb"))


def upgrade_shard(output, shard_name, work, min_free_gib, prepared):
    """Write/verify one closed shard without modifying its original file."""
    output, work = Path(output), Path(work)
    source_path = _root(output) / "shards" / shard_name
    original_sha = sha256_file(source_path / "data.mdb")
    if original_sha != prepared["original_sha256"]:
        raise ValueError("prepared cache source identity mismatch")
    cache_path = work / "prepared" / shard_name
    if sha256_file(cache_path / "data.mdb") != prepared["cache_sha256"]:
        raise ValueError("prepared cache checksum mismatch")
    cache_environment = lmdb.open(str(cache_path), readonly=True, lock=False, readahead=False)
    temporary = work / shard_name
    # Only this upgrader owns these unpublished scratch files; originals remain intact.
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    source = lmdb.open(str(source_path), readonly=True, lock=False, readahead=False)
    destination = lmdb.open(str(temporary), map_size=8 * 1024**3)
    count = 0
    before = hashlib.sha256()
    try:
        with source.begin() as read, cache_environment.begin() as cached:
            pending = []
            for key, payload in read.cursor():
                record = unpack_record(payload)
                if key.decode() != record["sample_id"]:
                    raise ValueError("LMDB key/sample identity mismatch")
                _hash_record(before, key, record)
                envelope_payload = cached.get(key)
                if envelope_payload is None:
                    raise ValueError("prepared cache is missing a sample")
                envelope = unpack_record(envelope_payload)
                if envelope["sample_id"] != record["sample_id"]:
                    raise ValueError("prepared cache sample identity mismatch")
                record[STATIC_FEATURES_KEY] = envelope["features"]
                read_static_features(record)
                pending.append((key, pack_record(record, compression="zlib")))
                count += 1
                if len(pending) == 64:
                    _commit(destination, pending, work, min_free_gib)
                    pending.clear()
            if pending:
                _commit(destination, pending, work, min_free_gib)
        destination.sync()
    finally:
        source.close()
        destination.close()
        cache_environment.close()
    after = hashlib.sha256()
    seen = 0
    verified = lmdb.open(str(temporary), readonly=True, lock=False, readahead=False)
    try:
        with verified.begin() as transaction:
            for key, payload in transaction.cursor():
                record = unpack_record(payload)
                read_static_features(record)
                _hash_record(after, key, record)
                seen += 1
    finally:
        verified.close()
    if seen != count or count != prepared["records"] or before.digest() != after.digest():
        raise ValueError("static upgrade changed an original record")
    if sha256_file(_root(output) / "shards" / shard_name / "data.mdb") != original_sha:
        raise ValueError("source shard changed during static upgrade")
    return dict(shard_id=shard_name, records=count, original_sha256=original_sha,
        data_mdb_sha256=sha256_file(temporary / "data.mdb"),
        original_record_stream_sha256=before.hexdigest(),
        bytes_before=(_root(output) / "shards" / shard_name / "data.mdb").stat().st_size,
        bytes_after=(temporary / "data.mdb").stat().st_size)


def _commit(environment, pending, work, min_free_gib):
    if shutil.disk_usage(work).free < min_free_gib * 1024**3:
        raise OSError(f"free disk space below {min_free_gib} GiB reserve")
    while True:
        try:
            with environment.begin(write=True) as transaction:
                for key, data in pending:
                    if not transaction.put(key, data, overwrite=False):
                        raise ValueError("duplicate key in upgraded shard")
            return
        except lmdb.MapFullError:
            environment.set_mapsize(environment.info()["map_size"] * 2)


@contextmanager
def _lock(path):
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def _source_complete(output, producer_status):
    if not output.exists():
        return False
    if producer_status is not None:
        status = json.loads(producer_status.read_text())
        if status["status"] in ("failed", "interrupted"):
            raise RuntimeError("source conversion stopped; preserved upgrade checkpoint for recovery")
        return status["status"] == "complete"
    return (output / "metadata.json").exists()


def _finalize(output, work, state, state_path):
    root = _root(output)
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    completed = state["shards"]
    actual = {path.name for path in (root / "shards").glob("*.lmdb")}
    declared = {item["shard_id"] for item in metadata["shards"]}
    if actual != declared or actual != set(completed):
        raise ValueError("upgrade shard set differs from dataset metadata")
    record_count = sum(item["records"] for item in completed.values())
    if record_count != metadata["record_count"]:
        raise ValueError("upgrade record count differs from metadata")
    if pq.ParquetFile(root / "manifest.parquet").metadata.num_rows != record_count:
        raise ValueError("upgrade record count differs from manifest")
    manifest_sha = sha256_file(root / "manifest.parquet")
    csv_path = root / "sample_inventory.csv"
    csv_sha = sha256_file(csv_path) if csv_path.exists() else None
    if metadata.get("manifest_sha256") != manifest_sha:
        raise ValueError("source manifest checksum mismatch")
    if metadata.get("sample_inventory_sha256") not in (None, csv_sha):
        raise ValueError("source sample CSV checksum mismatch")
    for item in metadata["shards"]:
        name = item["shard_id"]
        digest = sha256_file(root / "shards" / name / "data.mdb")
        if digest != completed[name]["data_mdb_sha256"]:
            raise ValueError(f"upgraded shard checksum changed: {name}")
        item["data_mdb_sha256"] = digest
    metadata["record_compression"] = "zlib"
    metadata["preprocessing"]["static_features_schema"] = STATIC_FEATURES_SCHEMA
    metadata["preprocessing_config_sha256"] = canonical_sha256(metadata["preprocessing"])
    # Keep the previous publication descriptors, not duplicate LMDB payloads.
    backup = work / "original_metadata.json"
    if not backup.exists():
        shutil.copyfile(metadata_path, backup)
    write_json(metadata_path, metadata)
    identity = joint_v2_dataset_identity(root)
    report = dict(schema_version=STATIC_FEATURES_SCHEMA, status="complete",
        records=record_count, shards=len(completed), dataset_identity=identity,
        manifest_sha256=manifest_sha, sample_inventory_sha256=csv_sha,
        code_sha256=state["code_sha256"], elapsed_seconds=round(time.time()-state["started_epoch"], 1),
        verification="every decoded original field and every condition/target tensor unchanged")
    build_path = root / "build.json"
    if build_path.exists():
        build = json.loads(build_path.read_text())
        if not (work / "original_build.json").exists():
            shutil.copyfile(build_path, work / "original_build.json")
        build.update(dataset_identity=identity, static_features_upgrade=report)
        write_json(build_path, build)
    # The BoltzGen launcher also records a dataset summary outside dataset/.
    summary_path = output.parent / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if Path(summary.get("dataset", "")) == output:
            summary["dataset_summary"] = json.loads(build_path.read_text())
            summary["static_features_upgrade"] = report
            write_json(summary_path, summary)
    write_json(root / "static_features_upgrade.json", report)
    state.update(status="complete", report=report)
    write_json(state_path, state)
    # This is the final publication step; readers fail closed until now.
    (root / UPGRADE_MARKER).unlink()
    return report


def run_upgrade(output, *, workers=4, min_free_gib=20, watch=False,
                producer_status=None, poll_seconds=10):
    output = Path(output).resolve()
    producer_status = Path(producer_status) if producer_status else None
    work = output.parent / ("." + output.name + ".static-upgrade")
    work.mkdir(exist_ok=True)
    state_path = work / "status.json"
    code_paths = [Path(__file__).resolve(),
        Path(__file__).resolve().parents[2] / "src/apexgen/joint_v2/data/static_features.py",
        Path(__file__).resolve().parents[2] / "src/apexgen/joint_v2/data/batch.py"]
    code_paths += [Path(__file__).resolve().parents[2] / path for path in (
        "src/apexgen/joint_v2/data/native_targets.py",
        "src/apexgen/shared/geometry/torsion.py",
        "src/apexgen/shared/geometry/sidechain.py",
        "src/apexgen/shared/storage/features.py",
    )]
    hashes = {str(path): sha256_file(path) for path in code_paths}
    with _lock(work / "upgrade.lock"):
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state["code_sha256"] != hashes:
                raise ValueError("upgrade code changed; checkpoint requires the original feature recipe")
            if state["status"] == "complete":
                # Recover a crash between the final status and marker removal.
                if joint_v2_dataset_identity(_root(output)) != state["report"]["dataset_identity"]:
                    raise ValueError("completed upgraded dataset identity changed")
                (_root(output) / UPGRADE_MARKER).unlink(missing_ok=True)
                return state["report"]
        else:
            state = dict(status="running", started_epoch=time.time(), output=str(output),
                         workers=workers, code_sha256=hashes, shards={}, prepared={})
        state["status"] = "running"
        marker = _root(output) / UPGRADE_MARKER
        if not marker.parent.exists():
            raise FileNotFoundError(marker.parent)
        write_json(marker, dict(schema_version=STATIC_FEATURES_SCHEMA, checkpoint=str(state_path),
                               usable_for_training=False))
        write_json(state_path, state)
        # Recover a crash after atomic replacement but before checkpoint commit.
        pending_commit = work / "pending_commit.json"
        if pending_commit.exists():
            result = json.loads(pending_commit.read_text())
            digest = sha256_file(_root(output) / "shards" / result["shard_id"] / "data.mdb")
            if digest == result["data_mdb_sha256"]:
                state["shards"][result["shard_id"]] = result
                write_json(state_path, state)
            elif digest != result["original_sha256"]:
                raise ValueError("interrupted shard replacement has an unknown identity")
            pending_commit.unlink()
        try:
            with ProcessPoolExecutor(max_workers=workers,
                    mp_context=multiprocessing.get_context("spawn"), initializer=_init_worker) as pool:
                pending = {}
                while True:
                    complete = _source_complete(output, producer_status)
                    if producer_status and not complete:
                        producer = json.loads(producer_status.read_text())
                        if producer["status"] in ("failed", "interrupted"):
                            raise RuntimeError("source conversion stopped; upgrade remains unpublished")
                    root = _root(output)
                    shards = sorted(path.name for path in (root / "shards").glob("*.lmdb"))
                    eligible = shards if complete else [name for i, name in enumerate(shards[:-1])
                        if int(shards[i+1][6:11]) == int(name[6:11]) + 1]
                    occupied = set(pending.values()) | set(state["prepared"])
                    for name in eligible:
                        if len(pending) >= workers:
                            break
                        if name in occupied:
                            continue
                        future = pool.submit(prepare_shard_cache, str(output), name, str(work), min_free_gib)
                        pending[future] = name
                    if not pending:
                        if complete:
                            break
                        if not watch:
                            raise RuntimeError("source dataset is unfinished; use --watch with producer status")
                        time.sleep(poll_seconds)
                        continue
                    finished, _ = wait(pending, timeout=poll_seconds, return_when=FIRST_COMPLETED)
                    for future in finished:
                        name = pending.pop(future)
                        result = future.result()
                        state["prepared"][name] = result
                        state.update(updated_epoch=time.time(), phase="precomputing",
                            records_precomputed=sum(item["records"] for item in state["prepared"].values()),
                            shards_precomputed=len(state["prepared"]), producer_complete=complete)
                        write_json(state_path, state)
                        print(json.dumps(dict(phase="precomputing", shard=name,
                            records=state["records_precomputed"], shards=state["shards_precomputed"],
                            elapsed=round(time.time()-state["started_epoch"],1))), flush=True)
            # No file replacement is permitted while the original producer can
            # still hash/publish its metadata. This prevents an identity race.
            source_metadata = json.loads((output / "metadata.json").read_text())
            declared = {item["shard_id"]: item["data_mdb_sha256"] for item in source_metadata["shards"]}
            if set(declared) != set(state["prepared"]):
                raise ValueError("source metadata differs from prepared shard set")
            for name, digest in declared.items():
                allowed = {state["prepared"][name]["original_sha256"]}
                if name in state["shards"]:
                    allowed.add(state["shards"][name]["data_mdb_sha256"])
                if digest not in allowed:
                    raise ValueError("source metadata shard identity changed")
            _init_worker()
            for name in sorted(state["prepared"]):
                if name in state["shards"]:
                    # A recovered atomic commit may have left scratch directories.
                    for scratch in (work / name, work / "prepared" / name):
                        if scratch.exists():
                            shutil.rmtree(scratch)
                    continue
                result = upgrade_shard(output, name, work, min_free_gib, state["prepared"][name])
                current = output / "shards" / name / "data.mdb"
                if sha256_file(current) != result["original_sha256"]:
                    raise ValueError("source shard changed before replacement")
                write_json(pending_commit, result)
                os.replace(work / name / "data.mdb", current)
                _sync_directory(current.parent)
                state["shards"][name] = result
                state.update(updated_epoch=time.time(), phase="merging",
                    records_upgraded=sum(item["records"] for item in state["shards"].values()),
                    shards_upgraded=len(state["shards"]))
                write_json(state_path, state)
                pending_commit.unlink()
                shutil.rmtree(work / name)
                shutil.rmtree(work / "prepared" / name)
                print(json.dumps(dict(phase="merging", shard=name, records=state["records_upgraded"],
                    shards=state["shards_upgraded"], elapsed=round(time.time()-state["started_epoch"],1))), flush=True)
            return _finalize(output, work, state, state_path)
        except BaseException as error:
            state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                         error_type=type(error).__name__, error=str(error))
            write_json(state_path, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-free-gib", type=int, default=20)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--producer-status", type=Path)
    args = parser.parse_args()
    if args.workers < 1 or args.min_free_gib < 0:
        parser.error("workers must be positive and reserve nonnegative")
    if args.watch and args.producer_status is None:
        parser.error("--watch requires --producer-status to distinguish completion from a failed producer")
    print(json.dumps(run_upgrade(args.dataset, workers=args.workers, min_free_gib=args.min_free_gib,
        watch=args.watch, producer_status=args.producer_status), indent=2))


if __name__ == "__main__":
    main()
