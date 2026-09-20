# Bidirectional Boltz interface-fragment training data

The input is `interfaces.parquet`. Each row describes one undirected contact pair.
A/B are ordered by source chain-table row, not receptor/binder labels. Build A→B and
B→A independently. `peptide` is the model's generated-side role; the source chain need
not be named P or be a naturally occurring short peptide.

## Selection rules

1. Locate the original NPZ and verify byte length, SHA256, chain rows/names/lengths,
   and residue indices against the inventory.
2. For A→B, split the B-side interface residues into consecutive runs of source
   `residues.res_idx`. Adjacent positions must differ by exactly one.
3. Retain every run meeting the minimum length as an independent sample, ordered by
   source position. Enumerate A-side runs independently for B→A. Retain maximal runs;
   do not additionally enumerate overlapping windows inside a run.
4. Each sample contains the experimental sequence and coordinates of exactly one run.
   **Never concatenate separate runs or insert gap residues to join them.**
5. Require N/CA/C for every target residue, consecutive sequence positions, and acceptable
   C–N distances and connection angles. Record rejection per segment; do not delete
   residues or fill coordinates to make a segment pass. Other valid segments remain eligible.
6. There is no maximum length. The default minimum is **four residues**, implementing
   length greater than three: `--min-fragment-length 4`. Short segments are recorded as
   `skipped_short`, not extended or stitched. The parameter is an inclusive minimum;
   the model permits an explicit minimum of three, but the default excludes length three.

For example, `10,11,12,13,30,31,32,33,34,50,51,52` produces independent samples
`[10,14)` and `[30,35)`; the final three-residue run is skipped. The opposite chain
provides context separately for each retained sample. These are zero-based NPZ polymer
positions, not PDB author residue numbers.

## Context and supervision

For each **selected target segment**, recompute the opposite chain's heavy-atom contact
core at ≤5 Å. Retain the core and its context within 11 Å by CB distance, using CA for Gly.
Do not include the remainder of the target's source chain. Nearby unrepresented source
atoms are recorded in `nearby_excluded_atom_rows`.

Subtract the same context-core CA centroid from both sides to preserve their relative
pose. Target sequence/coordinates enter supervision and noisy training states, not the
static condition; context remains fixed. The crop is selected from the experimental
interface, so inference needs a specified site. A fragment extracted from a larger protein
has not thereby been validated to fold or bind independently.

Retain residue-name/token cross-checks, name-based atom14 mapping, missing-side-chain
masks, and per-atom audits against the original NPZ. `peptide_residue_keys` preserves
source positions. Target batch positions become 0..L−1 only after single-run continuity
has been established. Standard peptide bonds at fragment boundaries may be cut and are
recorded explicitly. No terminal groups or OXT are added, and coordinates are unchanged.
Unsupported covalent crosslinks are rejected.

## Build and load

For extracted official BoltzGen data, build the inventory and records directly from
selected NPZ files. Selection is explicit; the output directory must be new:

```bash
source scripts/project_tmp_env.sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/data/prepare_joint_v2_boltzgen.py \
  --structures-dir artifacts/datasets/boltzgen1_train_official_20260920/training_data/targets/structures \
  --structure-ids 5hga 5k9s 4ep3 7yxn \
  --output artifacts/datasets/boltzgen_native_four_20260920 \
  --model-smoke
```

The outer directory contains `interfaces.parquet`, `interfaces.csv`, `source_audits.json`,
`summary.json`, and automatically generated `sample_inventory.csv` / `sample_inventory.md`.
Each inventory row corresponds to one accepted model sample. Fields include source
structure/file/SHA256, sample ID, direction, context/target chain IDs, original chain lengths,
actual pocket/target lengths, target sequence, and source-index ranges. Discontinuous
pocket positions are listed as separate runs, never represented by a misleading envelope.
Ranges use zero-based `[start, stop)`; the CSV also provides one-based inclusive ranges.
Neither convention is PDB author numbering.

The CSV also stores `record_index`, `shard_id`, `tensor_key`, and `dataset_source_npz`.
A row is appended and flushed immediately after its LMDB transaction commits. Inventory
generation does not reread LMDB or collect all sample rows in memory. Before writing,
verify sample identity, lengths, chains, and sequence across the interface inventory,
training manifest, and record. Interface CSV/Parquet files are written per source-structure
batch before sample conversion: this remains a two-stage process. Deduplication sets,
the current structure, and the current batch still use memory; constant memory is not claimed.

Model-ready data resides in **`dataset/`**. For standalone NPZ inputs, `source_archive`
is the NPZ path itself and `source_offset=0`. A ZIP member's compressed header offset
must not be treated as a raw NPZ byte offset.

During construction, inspect `dataset.inprogress/sample_inventory.csv`, `directions.jsonl`,
and `build_status.json`. Each saved LMDB sample has one CSV row; shards default to 256
samples. The Parquet manifest is written by scan batch. Successful completion renames
`dataset.inprogress/` to `dataset/`, copies the CSV to the outer directory, and streams
Markdown from the CSV without rereading atom tensors.

Python exceptions or Ctrl-C preserve intermediate files and a failed/interrupted status.
Existing intermediate directories are not overwritten. LMDB and CSV do not share a
transaction: interruption can leave CSV behind LMDB and Parquet containing only previous
complete batches. Intermediate output is not a complete training set and must not simply
be renamed. Automatic recovery/resume is not implemented. CSV flush writes user-space
buffers; it is not a per-row fsync guarantee against power loss.

The indexed tar/Parquet route remains available:

```bash
source scripts/project_tmp_env.sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/data/prepare_joint_v2_boltz_chain_pairs.py \
  --inventory artifacts/datasets/boltz_interface_inventory_20260920/interfaces.parquet \
  --output artifacts/datasets/boltz_all_interface_fragments_four_20260920 \
  --structure-ids 5hga 5k9s 4ep3 7yxn --min-fragment-length 4 --model-smoke
```

Each source NPZ is copied once under its content hash. `--structure-ids` bounds input;
without it, this inventory converter processes the complete supplied inventory and checks
both directions independently. The default split is `smoke`. An explicit
`--split-map splits.json` can supply `{structure_id: "train"/"validation"/...}`; every
selected structure must be assigned. All pairs, directions, and fragments from one structure
share a split to prevent within-structure leakage. Homology clustering/deduplication is
not automatic and must be prepared separately for a formal split.

- `manifest.parquet`: accepted segments, source chains, direction, `fragment_index`, and
  target start/stop. `pair_id` retains the undirected pair identity;
  `sample_id` is `pair_id:direction:fragmentSTART-STOP`, avoiding collisions among segments.
- `shards/*.lmdb`: records readable by `JointV2Dataset`; 256 records per shard by default.
- `directions.jsonl`: the outcome of **every consecutive run**, including short runs,
  coverage, and rejection reasons.
- If a listed source pair has no contact on either side after recomputation, its direction
  is `skipped_no_contact`. It is not segmented and does not block other pairs. The
  corresponding `build.json` count counts these directions.
- `mapping.json`: AA and atom-slot mapping contract.
- `build.json` / `metadata.json`: counts, parameters, input/output identities, and source hashes.
- `model_smoke.json`: optional CPU verification summary, in batches of eight; detailed
  forward/backward, condition independence, fixed-context, and mapping checks are in
  `model_smoke_batch_*.json`.

```python
from apexgen.joint_v2.data.dataset import JointV2Dataset
from apexgen.joint_v2.data.batch import collate_joint_v2_records

dataset = JointV2Dataset(
    "artifacts/datasets/boltzgen_native_four_20260920/dataset", split="smoke"
)
try:
    batch = collate_joint_v2_records([dataset[0], dataset[1]])
    batch.condition.validate_model_input()
finally:
    dataset.close()
```

Sample coverage is its segment's interface-residue count divided by the original total
interface-residue count on that direction's target side. A direction may yield multiple
samples, each with its own recomputed context; their coverage fractions can be summed.
Context may contain nonconsecutive runs, preserving source positions and break masks
without imposing false peptide bonds.

The four-case smoke is development validation, not a formal train/validation split;
it does not start optimizer or GPU training. The converter uses mapping contract v2.
Old v1 Boltz records must be rebuilt: Dataset and collate reject mismatched mappings,
and changing hashes manually is not a valid migration.
