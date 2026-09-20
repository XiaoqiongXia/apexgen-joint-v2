"""Write independent, immutable datasets consumable by the current v2 loader."""

from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import uuid

import lmdb
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from apexgen.shared.storage.store import pack_record
from apexgen.joint_v2.contracts.contract import JOINT_V2_CONTRACT_SHA256
from apexgen.joint_v2.data.dataset import COMPLEX_RECORD_SCHEMA, COMPLEX_DATASET_SCHEMA
from apexgen.joint_v2.runtime.lineage import canonical_sha256, joint_v2_dataset_identity, sha256_file
from apexgen.joint_v2.data.preprocessing.records import PocketParameters, preprocess_complex
from apexgen.shared.geometry.joint_residue_constants import (
    ATOM14_CONSTANTS_SHA256,
    BACKBONE_STEREOCHEMISTRY_SHA256,
)
import gemmi


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _put(environment, key, payload):
    while True:
        try:
            with environment.begin(write=True) as txn:
                if not txn.put(key.encode(), payload, overwrite=False):
                    raise ValueError(f"duplicate tensor key {key}")
            return
        except lmdb.MapFullError:
            environment.set_mapsize(environment.info()["map_size"] * 2)


def load_input_manifest(path: str | Path):
    """Read a JSON list or JSONL; raw_path is resolved relative to the manifest."""
    path = Path(path).resolve()
    content = path.read_text()
    rows = (
        json.loads(content)
        if content.lstrip().startswith("[")
        else [json.loads(line) for line in content.splitlines() if line.strip()]
    )
    for row in rows:
        raw = Path(row["raw_path"]).expanduser()
        row["raw_path"] = str(
            (path.parent / raw).resolve() if not raw.is_absolute() else raw.resolve()
        )
    return rows


def build_dataset(rows, output: str | Path, *, parameters=None, shard_size=1000):
    """Fail on bad input and existing output; publish only after all checks pass."""
    parameters = parameters or PocketParameters()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists; choose a new directory: {output}")
    if (
        not rows
        or not isinstance(shard_size, int)
        or isinstance(shard_size, bool)
        or shard_size < 1
    ):
        raise ValueError("nonempty input and positive shard_size required")
    required = {
        "sample_id",
        "source_pdb_id",
        "split",
        "raw_path",
        "receptor_chain_id",
        "peptide_chain_id",
    }
    for row in rows:
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError(f"each input needs {sorted(required)}")
        if any(not isinstance(row[k], str) or not row[k] for k in required):
            raise ValueError("input fields must be nonempty strings")
        if not Path(row["raw_path"]).is_file():
            raise FileNotFoundError(row["raw_path"])
        index = row.get("model_index", 0)
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("model_index must be a nonnegative integer")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate sample_id across input records/splits")
    rows = sorted(rows, key=lambda r: (r["split"], r["sample_id"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.building-{uuid.uuid4().hex}"
    pocket = staging
    (pocket / "shards").mkdir(parents=True)
    pocket_env = None
    manifest, shards = [], []
    try:
        shard_name = None
        quality_counts = Counter()
        for i, row in enumerate(rows):
            if i % shard_size == 0:
                if pocket_env is not None:
                    pocket_env.close()
                    pocket_env = None
                    shards.append(
                        {
                            "shard_id": shard_name,
                            "data_mdb_sha256": sha256_file(
                                pocket / "shards" / shard_name / "data.mdb"
                            ),
                        }
                    )
                shard_name = f"shard-{i // shard_size:05d}.lmdb"
                pocket_env = lmdb.open(str(pocket / "shards" / shard_name), map_size=64 << 20)
            try:
                record, geometry = preprocess_complex(
                    row["raw_path"],
                    sample_id=row["sample_id"],
                    source_pdb_id=row["source_pdb_id"],
                    split=row["split"],
                    receptor_chain_id=row["receptor_chain_id"],
                    peptide_chain_id=row["peptide_chain_id"],
                    parameters=parameters,
                    model_index=row.get("model_index", 0),
                    expected_peptide_sequence=row.get("expected_peptide_sequence"),
                )
            except Exception as exc:
                raise ValueError(f"{row['sample_id']}: {exc}") from exc
            # Bind tensor contents to an unchanged source file, including during parsing.
            if sha256_file(row["raw_path"]) != record["raw_file_sha256"]:
                raise ValueError(f"source changed while processing {row['sample_id']}")
            quality_counts.update(record["structure_quality"]["warnings"])
            with (staging / "quality.jsonl").open("a") as report:
                report.write(
                    json.dumps(
                        dict(
                            sample_id=row["sample_id"],
                            raw_file_sha256=record["raw_file_sha256"],
                            **record["structure_quality"],
                        ),
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
            record["complex_schema_version"] = COMPLEX_RECORD_SCHEMA
            record["joint_v2_target"] = geometry
            _put(pocket_env, row["sample_id"], pack_record(record))
            entry = dict(
                schema_version="apexgen.manifest.v0",
                sample_id=row["sample_id"],
                source_pdb_id=row["source_pdb_id"],
                split=row["split"],
                raw_path=record["raw_path"],
                status="included",
                reason=None,
                shard_id=shard_name,
                record_index=i,
                tensor_key=row["sample_id"],
                peptide_length=record["peptide_length"],
                pocket_size=len(record["pocket_aatype"]),
                raw_file_sha256=record["raw_file_sha256"],
            )
            manifest.append(entry)
        pocket_env.close()
        pocket_env = None
        shards.append(
            {
                "shard_id": shard_name,
                "data_mdb_sha256": sha256_file(pocket / "shards" / shard_name / "data.mdb"),
            }
        )
        pq.write_table(pa.Table.from_pylist(manifest), pocket / "manifest.parquet")
        policy = dict(
            schema="apexgen.joint_v2.native_pdb_preprocessing.v4",
            **asdict(parameters),
            core="native peptide/receptor heavy atom distance <= cutoff",
            context="CB-to-any-core-CB distance <= radius; Gly CA; virtual CB fallback; always keep core",
            coordinate_system="subtract core CA centroid; no native chain idealization",
            joint_v2_contract_sha256=JOINT_V2_CONTRACT_SHA256,
            chemistry="explicit receptor normalization; canonical peptide; omit receptor nonprotein residues",
            peptide_completeness="reject sequence/geometry gaps and declared missing residues; unknown termini flagged when no metadata",
            backbone_link_cn_bounds_angstrom=[
                parameters.geometry.min_cn_angstrom,
                parameters.geometry.max_cn_angstrom,
            ],
            receptor_breaks="retain observed fragments; mask cross-break torsions",
            missing_atoms="positive occupancy only; require N/CA/C; mask other missing atoms; never fill",
            polymer_index="deposited label_seq_id - 1; unambiguous declared alignment; otherwise separate observed fragments; never renumber after cropping",
            chemical_qc="elements, pinned bond/angle bounds, observed stereocentres, explicit and plausible implicit covalent links",
            gemmi_version=gemmi.__version__,
            atom14_constants_sha256=ATOM14_CONSTANTS_SHA256,
            backbone_stereochemistry_sha256=BACKBONE_STEREOCHEMISTRY_SHA256,
        )
        _write_json(
            pocket / "metadata.json",
            dict(
                schema_version=COMPLEX_DATASET_SCHEMA,
                record_schema_version=COMPLEX_RECORD_SCHEMA,
                record_count=len(manifest),
                manifest_sha256=sha256_file(pocket / "manifest.parquet"),
                shards=shards,
                preprocessing=policy,
                preprocessing_config_sha256=canonical_sha256(policy),
            ),
        )
        identity = joint_v2_dataset_identity(pocket)
        raw_root = os.path.commonpath([str(Path(r["raw_path"]).resolve().parent) for r in rows])
        data_config = dict(
            schema_version="apexgen.data.joint_v2.v3",
            dataset_root=".",
            raw_structure_root=raw_root,
            train_split="train",
            validation_split="valid",
        )
        (staging / "data.yaml").write_text(yaml.safe_dump(data_config, sort_keys=False))
        # Bind all shared primitives and v2 modules after functional packaging.
        code_files = sorted(Path(__file__).resolve().parents[3].rglob("*.py"))
        summary = dict(
            formal=False,
            record_count=len(rows),
            split_counts=dict(Counter(r["split"] for r in rows)),
            preprocessing=policy,
            dataset_identity=identity,
            source_sha256={str(p.resolve()): sha256_file(p) for p in code_files},
            input_records=[
                {**r, "raw_file_sha256": m["raw_file_sha256"]} for r, m in zip(rows, manifest)
            ],
            pocket_sizes={r["sample_id"]: r["pocket_size"] for r in manifest},
            quality_warning_counts=dict(quality_counts),
            quality_report="quality.jsonl",
        )
        _write_json(staging / "build.json", summary)
        # A completed destination is never intentionally replaced.
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
        return summary
    finally:
        if pocket_env is not None:
            pocket_env.close()
        if staging.exists():
            shutil.rmtree(staging)
