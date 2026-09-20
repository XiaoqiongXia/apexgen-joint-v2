"""Checked metadata shared by streaming dataset builds and inventory exports."""

import csv
import json
from pathlib import Path

from apexgen.shared.geometry.joint_residue_constants import AA1_ORDER


SAMPLE_FIELDS = (
    "sample_id", "source_structure_id", "source_schema", "source_archive", "source_member",
    "source_npz_sha256", "split", "direction", "condition_chain_id", "condition_source_length",
    "pocket_length", "pocket_source_ranges_0based_half_open", "target_chain_id",
    "target_source_length", "target_length", "target_start_0based", "target_stop_0based_exclusive",
    "target_start_1based", "target_end_1based_inclusive", "target_sequence", "index_source",
    "record_index", "shard_id", "tensor_key", "dataset_source_npz",
)


def _source_ranges(positions):
    """Keep gaps visible: ranges are zero-based [start, stop), never an envelope."""
    if not positions or positions != sorted(set(positions)):
        raise ValueError("expected nonempty ordered unique polymer indices")
    runs = []
    start = previous = positions[0]
    for position in positions[1:]:
        if position != previous + 1:
            runs.append([start, previous + 1])
            start = position
        previous = position
    return runs + [[start, previous + 1]]


def _sample_inventory_row(record, pair, entry):
    """Reduce one checked model record to scalar/string CSV metadata."""
    meta = record["interface_pair"]
    side, target_side = meta["condition_side"], meta["target_side"]
    lo, hi = meta["target_start"], meta["target_stop"]
    positions = [k["polymer_index"] for k in record["peptide_residue_keys"]]
    pocket_positions = [k["polymer_index"] for k in record["pocket_residue_keys"]]
    sequence = "".join(AA1_ORDER[int(aa)] for aa in record["joint_v2_target"]["aatype"])
    if (record["raw_file_sha256"] != pair["source_npz_sha256"]
            or record["source_pdb_id"] != pair["source_structure_id"]
            or meta["condition_chain_id"] != pair[f"chain_{side}_id"]
            or meta["target_chain_id"] != pair[f"chain_{target_side}_id"]
            or meta["target_source_length"] != pair[f"chain_{target_side}_length"]
            or positions != list(range(lo, hi)) or len(positions) != record["peptide_length"]
            or len(sequence) != len(positions)
            or sequence != record["boltz_adapter"]["peptide_sequence"]
            or any(k["boltz_chain_name"] != meta["condition_chain_id"]
                   for k in record["pocket_residue_keys"])
            or any(k["boltz_chain_name"] != meta["target_chain_id"]
                   for k in record["peptide_residue_keys"])
            or any(entry[k] != value for k, value in dict(
                sample_id=record["sample_id"], source_pdb_id=record["source_pdb_id"],
                raw_file_sha256=record["raw_file_sha256"], split=record["split"],
                pair_id=meta["pair_id"], direction=meta["direction"],
                condition_chain_id=meta["condition_chain_id"], target_chain_id=meta["target_chain_id"],
                peptide_length=len(positions), pocket_size=len(pocket_positions),
                target_start=lo, target_stop=hi).items())):
        raise ValueError("sample inventory/source identity mismatch")
    return dict(
            sample_id=record["sample_id"], source_structure_id=record["source_pdb_id"],
            source_schema=record["boltz_adapter"]["source_schema"],
            source_archive=pair["source_archive"], source_member=pair["source_member"],
            source_npz_sha256=record["raw_file_sha256"], split=record["split"],
            direction=meta["direction"], condition_chain_id=meta["condition_chain_id"],
            condition_source_length=pair[f"chain_{side}_length"], pocket_length=len(pocket_positions),
            pocket_source_ranges_0based_half_open=json.dumps(_source_ranges(pocket_positions)),
            target_chain_id=meta["target_chain_id"], target_source_length=meta["target_source_length"],
            target_length=record["peptide_length"], target_start_0based=lo,
            target_stop_0based_exclusive=hi, target_start_1based=lo + 1,
            target_end_1based_inclusive=hi, target_sequence=sequence,
            index_source="NPZ residues.res_idx; not PDB auth_seq_id",
            record_index=entry["record_index"], shard_id=entry["shard_id"],
            tensor_key=entry["tensor_key"], dataset_source_npz=record["raw_path"],
        )


def write_inventory_markdown(csv_path, md_path, dataset_dir):
    """Render from the saved CSV with bounded sample memory; never reopen LMDB."""
    csv_path, md_path = Path(csv_path), Path(md_path)
    sources, count = set(), 0
    with csv_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            count += 1
            sources.add((row["source_structure_id"], row["source_archive"], row["source_member"]))
    with md_path.open("w", encoding="utf-8") as out:
        out.write(f"# BoltzGen model sample inventory\n\nAccepted samples: {count}. Dataset: `{dataset_dir}`.\n\n")
        out.write("Each row represents a saved sample; see dataset/directions.jsonl for short runs and rejection reasons.\n"
            "Chain IDs are NPZ assembly chain names. Ranges [start, stop) are zero-based half-open source-chain residues.res_idx positions, "
            "not PDB author residue numbers. The CSV also provides one-based inclusive ranges. Discontinuous pocket positions are listed as separate runs.\n\n"
            "## Data sources\n\n| Structure ID | Source NPZ / archive | Member |\n|---|---|---|\n")
        for sid, archive, member in sorted(sources):
            out.write(f"| {sid} | [{Path(archive).name}](<{archive}>) | {member} |\n")
        out.write("\nSee sample_inventory.csv for source formats and NPZ SHA256 hashes.\n\n## Samples\n\n"
            "| Sample ID | Structure ID | Context chain | Source context length | Pocket length | Target chain | Source target length | Target fragment length | Target source indices [start,stop) | Target sequence |\n"
            "|---|---|---|---:|---:|---|---:|---:|---|---|\n")
        with csv_path.open(newline="", encoding="utf-8") as stream:
            for r in csv.DictReader(stream):
                out.write(f"| {r['sample_id']} | {r['source_structure_id']} | {r['condition_chain_id']} | "
                    f"{r['condition_source_length']} | {r['pocket_length']} | {r['target_chain_id']} | "
                    f"{r['target_source_length']} | {r['target_length']} | "
                    f"[{r['target_start_0based']},{r['target_stop_0based_exclusive']}) | {r['target_sequence']} |\n")
        out.write("\n## Retained pocket source-chain ranges\n\nEach run below is a zero-based half-open range.\n\n"
            "| Sample ID | Context chain | Discontinuous ranges |\n|---|---|---|\n")
        with csv_path.open(newline="", encoding="utf-8") as stream:
            for r in csv.DictReader(stream):
                ranges = "; ".join(f"[{lo},{hi})" for lo, hi in json.loads(r["pocket_source_ranges_0based_half_open"]))
                out.write(f"| {r['sample_id']} | {r['condition_chain_id']} | {ranges} |\n")
