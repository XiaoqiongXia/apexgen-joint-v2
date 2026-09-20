#!/usr/bin/env python
"""Selected extracted BoltzGen NPZs -> interface inventory -> Joint-v2 dataset.

Select structure IDs or explicitly opt in to scanning all extracted NPZs.
The resulting split is smoke; optional model checks run forward/backward on CPU.
"""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root), str(root / "src")]

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from apexgen.joint_v2.data.boltz_interfaces import CSV_FIELDS, PAIR_SCHEMA, describe_interfaces
from apexgen.joint_v2.data.dataset import JointV2Dataset
from apexgen.joint_v2.data.sample_inventory import (
    SAMPLE_FIELDS, _source_ranges, _sample_inventory_row, write_inventory_markdown,
)
from scripts.data.prepare_joint_v2_boltz_chain_pairs import build_dataset
from scripts.data.prepare_joint_v2_boltz_npz import smoke_model, write_json



def write_sample_inventory(dataset_dir, inventory, output):
    """Backfill inventories for older datasets; new builds write CSV in the builder."""
    dataset_dir, output = Path(dataset_dir).resolve(), Path(output).resolve()
    csv_path, md_path = output / "sample_inventory.csv", output / "sample_inventory.md"
    for path in (csv_path, md_path):
        if path.exists():
            raise FileExistsError(path)
    pairs = {p["pair_id"]: p for p in pq.read_table(inventory).to_pylist()}
    manifest = pq.read_table(dataset_dir / "manifest.parquet").to_pylist()
    rows_by_id = {}
    for split in sorted({r["split"] for r in manifest if r["status"] == "included"}):
        dataset = JointV2Dataset(dataset_dir, split=split)
        try:
            for i, entry in enumerate(dataset.rows):
                record = dataset[i]
                sid = record["sample_id"]
                if sid in rows_by_id:
                    raise ValueError(f"duplicate sample inventory ID: {sid}")
                rows_by_id[sid] = _sample_inventory_row(
                    record, pairs[record["interface_pair"]["pair_id"]], entry)
                del record
        finally:
            dataset.close()
    rows = [rows_by_id[e["sample_id"]] for e in manifest if e["status"] == "included"]
    output.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_inventory_markdown(csv_path, md_path, dataset_dir)

    return dict(csv=csv_path.name, markdown=md_path.name, samples=len(rows))


def prepare_selected(structures_dir, structure_ids, output, *, model_config=None, min_free_gib=20):
    structures_dir, output = Path(structures_dir).resolve(), Path(output).resolve()
    if (not structure_ids or len(set(structure_ids)) != len(structure_ids)
            or any(not isinstance(sid, str) or not sid.isascii() or not sid.isalnum()
                   for sid in structure_ids)):
        raise ValueError("provide distinct alphanumeric structure IDs")
    paths = [structures_dir / f"{sid}.npz" for sid in structure_ids]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(output)
    # Read only explicitly selected files. A standalone NPZ is a byte source
    # with offset 0; never reuse compressed ZIP header offsets as data offsets.
    output.mkdir(parents=True)
    started = time.monotonic()
    def progress(phase, checked, **detail):
        status = dict(phase=phase, sources_checked=checked, sources_total=len(paths),
            elapsed_seconds=round(time.monotonic() - started, 1), **detail)
        temporary = output / '.progress.json.tmp'
        write_json(temporary, status)
        temporary.replace(output / 'progress.json')
        print(json.dumps(status), flush=True)
    progress('interfaces', 0)
    inventory = output / "interfaces.parquet"
    pair_count = 0
    with (pq.ParquetWriter(inventory, PAIR_SCHEMA, compression="zstd") as parquet,
          (output / "interfaces.csv").open("w", newline="", encoding="utf-8") as stream,
          (output / "source_audits.json").open("w", encoding="utf-8") as audit_stream):
        csv_writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        csv_writer.writeheader()
        stream.flush()
        audit_stream.write("[\n")
        try:
            for index, path in enumerate(paths):
                if shutil.disk_usage(output).free < min_free_gib * 1024**3:
                    raise OSError(f"free disk space below reserve of {min_free_gib} GiB")
                payload = path.read_bytes()
                source = dict(source_archive=str(path), source_member=path.name,
                    source_npz_filename=path.name, source_structure_id=path.stem,
                    source_offset=0, source_size=len(payload))
                rows, audit = describe_interfaces(payload, source)
                if rows:
                    parquet.write_table(pa.Table.from_pylist(rows, schema=PAIR_SCHEMA))
                    csv_writer.writerows(rows)
                    stream.flush()
                    pair_count += len(rows)
                if index:
                    audit_stream.write(",\n")
                audit_stream.write(json.dumps(audit, allow_nan=False))
                audit_stream.flush()
                if (index + 1) % 100 == 0 or index + 1 == len(paths):
                    progress('interfaces', index + 1, interface_pairs=pair_count)
        finally:
            audit_stream.write("\n]\n")
    if not pair_count:
        raise ValueError("selected structures contain no stored protein interface pairs")
    dataset_dir = output / "dataset"
    progress('dataset', len(paths), interface_pairs=pair_count)
    summary = build_dataset(inventory, dataset_dir, min_fragment_length=4, min_free_gib=min_free_gib)
    result = dict(structure_ids=list(structure_ids), source_storage="standalone_npz",
        sources_checked=len(paths), source_audits="source_audits.json", dataset=str(dataset_dir),
        dataset_summary=summary)
    # Save preprocessing results even if optional model verification fails.
    write_json(output / "summary.json", result)
    # The CSV was written alongside LMDB during conversion. Do not reread tensors.
    shutil.copyfile(dataset_dir / "sample_inventory.csv", output / "sample_inventory.csv")
    write_inventory_markdown(output / "sample_inventory.csv", output / "sample_inventory.md", dataset_dir)
    result["sample_inventory"] = dict(csv="sample_inventory.csv", markdown="sample_inventory.md",
        samples=summary["counts"]["included"])
    write_json(output / "summary.json", result)
    if model_config is not None:
        count = summary["counts"]["included"]
        if not count:
            raise ValueError("no accepted fragments available for model verification")
        batches = []
        for start in range(0, count, 8):
            name = f"model_smoke_batch_{start // 8:04d}.json"
            checked = smoke_model(dataset_dir, model_config,
                record_indices=list(range(start, min(start + 8, count))), report_name=name)
            batches.append(dict(report=name, sample_ids=checked["sample_ids"], status=checked["status"]))
        result["model_smoke"] = dict(status="passed", samples=count, device="cpu", batches=batches)
        write_json(dataset_dir / "model_smoke.json", result["model_smoke"])
        write_json(output / "summary.json", result)
    progress('complete', len(paths), samples=summary['counts']['included'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures-dir", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--structure-ids", nargs="+")
    selection.add_argument("--all-structures", action="store_true",
        help="explicitly process every *.npz in structures-dir")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-smoke", action="store_true")
    parser.add_argument("--min-free-gib", type=int, default=20,
        help="stop and retain partial products when free disk space falls below this reserve")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2]
        / "configs/joint_v2/experiments/sequence_structure_tiny_encoder_bottleneck_v1.yaml")
    args = parser.parse_args()
    if args.min_free_gib < 0:
        parser.error('--min-free-gib must be nonnegative')
    ids = (sorted(p.stem for p in args.structures_dir.glob('*.npz') if p.is_file())
           if args.all_structures else args.structure_ids)
    print(json.dumps(prepare_selected(args.structures_dir, ids, args.output,
        model_config=args.config if args.model_smoke else None,
        min_free_gib=args.min_free_gib), indent=2))


if __name__ == "__main__":
    main()
