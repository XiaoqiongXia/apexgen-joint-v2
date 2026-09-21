# BoltzGen overfitting dataset: 24,576 samples

This overfitting dataset contains **three LMDB shards with 8,192 records each**,
for **24,576 samples** in total. Every record contains the pocket condition and
peptide labels with precomputed static geometry. The LMDB shards total 441.6 MiB;
the complete example is approximately 578 MiB, including 422 source NPZ files,
a CSV inventory, a Parquet manifest, and validation metadata.
LMDB and NPZ files are stored in **Git LFS**.

## Download and verify

From the repository root, with Git LFS installed and repository access configured:

```bash
git lfs install
git pull --ff-only
git lfs pull --include='examples/boltzgen_overfit/**' --exclude=''
source scripts/project_tmp_env.sh
python examples/boltzgen_overfit/verify.py
```

The verifier checks every file checksum, dataset identity, all 24,576 static
feature caches, CSV/sample correspondence, and strict/light collation on eight
samples. A remaining LFS pointer fails verification.

## Train

After installing the project dependencies:

```bash
source scripts/project_tmp_env.sh
apexgen-simplex train \
  --config configs/joint_v2/portable/simplex_tiny.yaml \
  --dataset examples/boltzgen_overfit/dataset --split smoke \
  --geometry-checks auto --steps 2 --device cpu \
  --output runs/boltzgen-overfit-smoke
```

`auto` verifies the dataset at startup and then assembles batches from cached
features. Use `--geometry-checks full` for strict per-batch geometry checks.
Custom loaders should use `prepare_training_collator(dataset)` as `collate_fn`.

The cache schema is `apexgen.joint_v2.static_features.v1`. Pocket features include
sequence/chain indices, backbone dihedrals, sidechain chi angles, and their masks.
Peptide features include observed backbone frames, backbone/sidechain angles,
and masks. Original atom coordinates, residue identities, masks and provenance
are retained. Padding, random flow states and learned features remain online.

## Inventory and provenance

`dataset/sample_inventory.csv` preserves the sample ID, source structure and
chain IDs, pocket/peptide lengths, residue index ranges, and LMDB key.
Historical absolute source paths are provenance; the loader resolves bundled
`dataset/sources/<sha256>.npz` files after relocation.

These are the first three closed shards of the full conversion, upgraded with
static features. The first shard is byte-for-byte identical to the previously
published 8,192-sample static-feature example; two new shards add 16,384 records.
The directory was renamed from `boltzgen8192_static` to `boltzgen_overfit`.
Its split remains **smoke**, with no random or homology-separated validation split.
Use this dataset for overfitting experiments and pipeline checks.

The source dataset card and license declaration are preserved in
`UPSTREAM_DATASET_CARD.md`. Upgrade and model checks are recorded in
`dataset/static_features_upgrade.json`, `dataset/export.json`, and
`dataset/model_smoke.json`. BoltzGen-processed coordinates do not retain every
field of the original PDB/mmCIF files.
