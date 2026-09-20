#!/usr/bin/env python
"""Build chain-pair metadata from indexed NPZ members of an uncompressed tar."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root), str(root / "src")]

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time

import pyarrow as pa
import pyarrow.parquet as pq

from apexgen.joint_v2.data.boltz_interfaces import (
    CHAIN_FIELDS, CSV_FIELDS, PAIR_SCHEMA, SCHEMA_VERSION, describe_interfaces,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(data)
    return digest.hexdigest()


def process_part(task):
    index, sources, directory, cutoff = task
    directory = Path(directory)
    prefix = directory / f"part-{index:05d}"
    counts = Counter()
    with (prefix.with_suffix(".csv")).open("w", newline="") as csv_file, \
            prefix.with_suffix(".jsonl").open("w") as audits, \
            pq.ParquetWriter(prefix.with_suffix(".parquet"), PAIR_SCHEMA, compression="zstd") as writer:
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        csv_writer.writeheader()
        buffer = []
        with Path(sources[0]["source_archive"]).open("rb") as archive:
            for source in sources:
                counts["structures_attempted"] += 1
                try:
                    archive.seek(source["source_offset"])
                    payload = archive.read(source["source_size"])
                    rows, audit = describe_interfaces(payload, source, cutoff=cutoff)
                except Exception as error:
                    counts["structures_failed"] += 1
                    audits.write(json.dumps(dict(**source, status="error",
                        error=f"{type(error).__name__}: {error}")) + "\n")
                    continue
                audits.write(json.dumps(audit) + "\n")
                counts["structures_with_protein_pairs"] += bool(rows)
                counts["structures_with_verified_contacts"] += any(r["contact_verified"] for r in rows)
                counts["pair_rows"] += len(rows)
                counts["verified_contact_pairs"] += sum(r["contact_verified"] for r in rows)
                counts["both_source_masks_true_pairs"] += sum(r["both_source_masks_true"] for r in rows)
                counts["verified_and_both_masks_true_pairs"] += sum(
                    r["contact_verified"] and r["both_source_masks_true"] for r in rows)
                counts["pairs_with_nonstandard_residues"] += sum(
                    bool(r["chain_a_nonstandard_residue_count"] or r["chain_b_nonstandard_residue_count"])
                    for r in rows)
                counts["pairs_with_multiple_coordinate_models"] += sum(r["ensemble_model_count"] > 1 for r in rows)
                csv_writer.writerows(rows)
                buffer.extend(rows)
                if len(buffer) >= 4096:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=PAIR_SCHEMA))
                    buffer.clear()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=PAIR_SCHEMA))
    return index, dict(counts)


def build_manifest(archive, members, output, *, workers=8, cutoff=5.0, block_size=128):
    archive, members, output = (Path(p).resolve() for p in (archive, members, output))
    if output.exists():
        raise FileExistsError(output)
    if workers < 1 or block_size < 1:
        raise ValueError("workers and block_size must be positive")
    stat = archive.stat()
    sources = []
    seen = set()
    with members.open(newline="") as stream:
        for entry in csv.DictReader(stream):
            member = PurePosixPath(entry["member"])
            if member.suffix != ".npz":
                continue
            offset, size = int(entry["offset"]), int(entry["size"])
            if offset < 0 or size <= 0 or offset + size > stat.st_size:
                raise ValueError(f"NPZ member outside archive bounds: {member}")
            if member.stem in seen:
                raise ValueError(f"duplicate structure identity: {member.stem}")
            seen.add(member.stem)
            sources.append(dict(source_archive=str(archive), source_member=str(member),
                source_npz_filename=member.name, source_structure_id=member.stem,
                source_offset=offset, source_size=size))
    if not sources:
        raise ValueError("inventory has no NPZ entries")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.build-", dir=output.parent) as temporary:
        stage = Path(temporary)
        parts = stage / "parts"
        parts.mkdir()
        tasks = [(i // block_size, sources[i:i + block_size], str(parts), cutoff)
                 for i in range(0, len(sources), block_size)]
        total = Counter()
        with ProcessPoolExecutor(max_workers=workers,
                mp_context=multiprocessing.get_context("spawn")) as pool:
            for index, counts in pool.map(process_part, tasks, chunksize=1):
                total.update(counts)
                if index % 10 == 0 or index == len(tasks) - 1:
                    print(json.dumps(dict(parts_complete=index + 1, parts_total=len(tasks),
                        **total, elapsed_seconds=round(time.monotonic() - started, 1))), flush=True)
        preview = []
        with (stage / "interfaces.csv").open("w", newline="") as csv_out, \
                (stage / "structures.jsonl").open("w") as audit_out, \
                (stage / "errors.jsonl").open("w") as error_out, \
                pq.ParquetWriter(stage / "interfaces.parquet", PAIR_SCHEMA, compression="zstd") as writer:
            for index, *_ in tasks:
                prefix = parts / f"part-{index:05d}"
                for batch in pq.ParquetFile(prefix.with_suffix(".parquet")).iter_batches(batch_size=8192):
                    writer.write_batch(batch)
                    if len(preview) < 20:
                        preview.extend(batch.slice(0, 20 - len(preview)).to_pylist())
                with prefix.with_suffix(".csv").open() as csv_in:
                    header = next(csv_in)
                    if index == 0:
                        csv_out.write(header)
                    shutil.copyfileobj(csv_in, csv_out)
                with prefix.with_suffix(".jsonl").open() as audit_in:
                    for line in audit_in:
                        audit_out.write(line)
                        if json.loads(line)["status"] == "error":
                            error_out.write(line)
        parquet_rows = pq.ParquetFile(stage / "interfaces.parquet").metadata.num_rows
        if parquet_rows != total["pair_rows"] or total["structures_attempted"] != len(sources):
            raise RuntimeError("manifest aggregation counts differ")
        now = archive.stat()
        if (now.st_size, now.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise RuntimeError("source archive changed during the build")
        with (stage / "preview.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(preview)
        (stage / "preview.json").write_text(json.dumps(preview, indent=2) + "\n")
        metadata = dict(schema=SCHEMA_VERSION, counts=dict(total),
            coordinate_source="atoms.coords", contact_cutoff_angstrom=cutoff,
            contact_definition="at least one present heavy-atom pair at distance <= cutoff (angstrom)",
            residue_count_definition="distinct residues, not atom contacts or solvent-accessible area",
            interface_source="NPZ interfaces chain-row pairs; protein/protein only; duplicate pair rows collapsed",
            filters="no chain length, chain mask, canonical chemistry or training geometry filter",
            source=dict(archive=str(archive), archive_size=stat.st_size, archive_mtime_ns=stat.st_mtime_ns,
                        archive_sha256=sha256_file(archive), members=str(members), members_sha256=sha256_file(members)),
            files={name: sha256_file(stage / name) for name in (
                "interfaces.parquet", "interfaces.csv", "structures.jsonl", "errors.jsonl")},
            code_sha256={str(Path(__file__).resolve()): sha256_file(__file__),
                str(Path(__file__).resolve().parents[2] / "src/apexgen/joint_v2/data/boltz_interfaces.py"):
                sha256_file(Path(__file__).resolve().parents[2] / "src/apexgen/joint_v2/data/boltz_interfaces.py")},
            chain_side_fields=list(CHAIN_FIELDS), source_author_numbering_available=False,
            directions="one row per unordered pair; derive a_to_b or b_to_a at sampling time",
            scope="only existing indexed members, not an assertion of full upstream archive completeness",
            elapsed_seconds=round(time.monotonic() - started, 2))
        (stage / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        shutil.rmtree(parts)
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--members", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(build_manifest(args.archive, args.members, args.output,
        workers=args.workers, cutoff=args.cutoff, block_size=args.block_size), indent=2))


if __name__ == "__main__":
    main()
