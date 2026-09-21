# BoltzGen 8,192-sample static-feature dataset

One closed shard exported from the running full conversion. Every record includes
precomputed pocket/peptide static geometry. Sample IDs and CSV provenance are preserved.

Load this directory with `JointV2Dataset(path, split="smoke")`, then use
`collate_joint_v2_records`. Requires the current static-feature-aware loader.

This is the first shard, not a random or homology-separated train/validation split.
See `export.json` for source identity, validation and model smoke results.
