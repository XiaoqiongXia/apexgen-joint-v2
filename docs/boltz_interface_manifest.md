# Boltz protein-chain interface inventory

Each row describes one **undirected protein chain pair** in an NPZ structure. A/B are
inventory sides ordered by source row (`chain_a_row < chain_b_row`), not literal chain
names. Training can expand one row into A→B and B→A without duplicating its metadata.

This inventory imposes no chain-length limit, target cropping, standard-AA requirement,
or training-adapter geometry filter. Chains with source `mask=False` remain visible for
task-specific filtering.

## Inventory built on 2026-09-20

Output: `artifacts/datasets/boltz_interface_inventory_20260920/`.
The local index covered 44,990 complete NPZ files, with zero processing failures.
Of these, 25,057 structures produced 625,317 distinct protein pairs, all verified by
actual heavy-atom contacts at ≤5 Å. Both source chain masks were true for 618,568 pairs;
source flags remain available for the other 6,749. These are interface candidates,
not samples that have passed training QC.

Thirteen automated tests passed. Six pairs from four real structures matched an independent
full distance-matrix calculation. Full-inventory checks verified pair uniqueness, index-list
lengths, residue-count bounds, contact distances, and agreement of key CSV/Parquet scalars;
see `verification.json`.

## Files

- `interfaces.csv`: scalar metadata for browsing/filtering.
- `interfaces.parquet`: the same records plus both interface-residue index lists.
- `preview.csv` / `preview.json`: the first 20 records.
- `structures.jsonl`: status and pair count for every input, including structures without protein interfaces.
- `errors.jsonl`: unreadable/unreliable inputs and reasons; failures are not silently discarded.
- `metadata.json`: contact definition, counts, source archive/inventory hashes, output hashes, and code hashes.
- `examples.csv`: six verified pairs from four real structures.
- `four_structure_audit.json`: independent distance-matrix checks for those structures.
- `verification.json`: full-inventory consistency/completeness checks.

## Fields

| Field | Meaning |
|---|---|
| `pair_id` | Pair identity, e.g. `5hga:chain0-chain2`; unique within this source archive |
| `source_archive` | Absolute path to the original tar archive |
| `source_member` | Complete NPZ member path inside the tar |
| `source_npz_filename` | Original filename, e.g. `5hga.npz` |
| `source_structure_id` | ID from the filename; PDB ID for this RCSB collection |
| `source_offset` / `source_size` | Byte offset of NPZ data and byte length for direct tar reads |
| `source_npz_sha256` | Hash of the complete NPZ bytes |
| `coordinate_source` | `atoms.coords` |
| `ensemble_model_count` | Source coordinate-model count; contacts use `atoms.coords`, not a model average |
| `contact_cutoff_angstrom` | Heavy-atom cutoff, 5 Å by default |
| `both_source_masks_true` | Whether both source Boltz chain masks pass |
| `contact_verified` | Whether recomputed coordinates still show contact at the selected cutoff |
| `minimum_contact_distance_angstrom` | Closest verified heavy-atom distance; null if no contact |

The following suffixes appear under both `chain_a_` and `chain_b_`:

| Suffix | Meaning |
|---|---|
| `row` | Zero-based row of the source `chains` table |
| `id` | Complete NPZ chain name, e.g. `A1`, `B2` |
| `asym_id` / `entity_id` / `sym_id` | Source instance, molecular entity, and symmetry-copy identifiers; distinct namespaces |
| `source_mask` | Original Boltz chain-validity flag |
| `length` | Number of residue entries in the full chain, including unobserved residues |
| `residue_table_start` | Chain start in the global `residues` table |
| `source_present_residue_count` | Number of residues with source `is_present=True` |
| `observed_residue_count` | Residues with at least one observed heavy atom; not a backbone-completeness count |
| `observed_heavy_atom_count` | Number of observed heavy atoms |
| `nonstandard_residue_count` | Source `is_standard=False` count; not a full model-AA compatibility check |
| `interface_residue_count` | Distinct residues contacting the partner chain |
| `interface_residue_indices` | Source `residues.res_idx`, usually zero-based chain polymer positions |
| `interface_residue_rows` | Global `residues` rows for the same interface residues |

The last two lists are stored in Parquet/JSON. Source NPZ files do not reliably retain
original author residue numbers or author-chain mappings. The inventory does not invent
these fields or strip assembly-copy suffixes from chain names.

## Contact definition

Recompute each protein–protein pair listed in NPZ `interfaces`. A residue participates if
at least one observed heavy atom is within **≤5 Å** of an observed heavy atom on the other
chain. In the legacy schema, these atoms have `is_present=True` and `element>1`.
Use all observed heavy atoms, not only CA/CB. The native BoltzGen schema without an
element column is handled by the schema adapter.

Count a residue once even if several of its atoms contact the partner. Side A and side B
usually have different counts. These counts are not atom-pair counts, residue-pair counts,
or buried surface area.

Missing-atom placeholders and hydrogens are excluded. Nonfinite observed heavy-atom
coordinates cause an explicit error. Duplicate and reversed source pairs collapse to
one row. Even with a larger custom cutoff, only pairs already listed in source `interfaces`
are recomputed; new pairs are not discovered.

The source parser uses `interfaces.chain_1/chain_2` as chain-table rows. Do not generally
interpret these as entity IDs, author chain IDs, or assume they equal `asym_id`.

## Read or build

```python
import pyarrow.parquet as pq

table = pq.read_table(
    "artifacts/datasets/boltz_interface_inventory_20260920/interfaces.parquet",
    columns=[
        "source_structure_id", "chain_a_id", "chain_b_id",
        "chain_a_length", "chain_b_length",
        "chain_a_interface_residue_count", "chain_b_interface_residue_count",
        "chain_a_interface_residue_indices", "chain_b_interface_residue_indices",
    ],
    filters=[("source_structure_id", "=", "5hga")],
)
print(table.to_pylist())
```

Original build command; the output directory must not exist:

```bash
source scripts/project_tmp_env.sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/data/build_boltz_interface_manifest.py \
  --archive /data2/xiaoqiong/python_project/BasinDiff/dataset/Structure/rcsb_processed_targets.tar \
  --members artifacts/tmp/boltz_inventory_20260920/rcsb_processed_targets_members.csv \
  --output artifacts/datasets/boltz_interface_inventory_20260920 \
  --workers 8 --cutoff 5.0
```

This is metadata before training-sample construction. Geometric contact does not establish
biological binding or complete model chemistry/geometry compatibility. The inventory covers
only RCSB members in the local complete-member index. The previously inventoried OpenFold
single-chain collection had no inter-protein-chain interfaces.
