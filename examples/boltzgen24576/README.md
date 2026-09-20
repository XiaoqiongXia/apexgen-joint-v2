# BoltzGen preview: 24,576 samples

This portable training subset contains three complete LMDB shards (8,192 samples each),
422 source NPZ files, `sample_inventory.csv`, a complete `manifest.parquet`, and mapping/hash metadata.
It occupies approximately 515 MiB. Large LMDB/NPZ files are stored in **Git LFS**.

## Download

From the repository root, after installing Git LFS and authenticating for this private repository:

```bash
git lfs install
git pull --ff-only
git lfs pull --include='examples/boltzgen24576/**' --exclude=''
(cd examples/boltzgen24576 && sha256sum -c SHA256SUMS)
python examples/boltzgen24576/verify.py
```

If cloning for the first time, Git LFS must be installed to download the large objects automatically.
The checksum check detects pointer files when an LFS download has not completed.

## Train

Install the project's dependencies as described in the root README, then run:

```bash
apexgen-simplex train \
  --config configs/joint_v2/portable/simplex_tiny.yaml \
  --dataset examples/boltzgen24576/dataset --split smoke \
  --steps 2 --device cpu --output runs/boltzgen24576-smoke
```

For a longer run, use a new output directory and select the appropriate step count/device.
The loader reads samples on demand. The LMDB values use lossless zlib compression;
this repository's decoder supports both these values and the older uncompressed example.
No manual patches or preprocessing are required after pulling this commit and its LFS objects.

Python access uses `JointV2Dataset('examples/boltzgen24576/dataset', split='smoke')`
and `collate_joint_v2_records`. Original source paths inside records are historical provenance;
the loader resolves bundled `dataset/sources/<sha256>.npz` files after relocation.

## Scope and provenance

These are the first three closed shards of the ongoing full BoltzGen conversion, not a
random or homology-separated benchmark. The preserved label is **smoke**, not train.
They can be used for exploratory training and pipeline checks, but do not supply an independent
validation set. No formal-training configuration or approval is implied by this data export.

All 24,576 records were decoded and checked against the CSV. Shard hashes were checked before
and after copying; the first and last record of every shard passed model-input collation.
See `dataset/export.json` for exact hashes and the source-run snapshot.

The data derives from the official BoltzGen training structures. The upstream dataset card
is preserved in `UPSTREAM_DATASET_CARD.md` (MIT declaration). The selection/export does not
change the upstream data license. Tensor coordinates are BoltzGen-processed observations,
not a lossless copy of all original PDB/mmCIF metadata.
